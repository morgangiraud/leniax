import os
import psutil
import time
import pickle
import json
import math
import matplotlib.pyplot as plt
from multiprocessing import get_context
from absl import logging as absl_logging
from typing import Dict, Callable, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from ribs.archives import ArchiveBase
from ribs.archives import GridArchive, CVTArchive
from ribs.visualize import grid_archive_heatmap, cvt_archive_heatmap, parallel_axes_plot
from ribs.optimizers import Optimizer

from . import utils as leniax_utils
from . import loader as leniax_loader
from . import runner as leniax_runner
from . import helpers as leniax_helpers
from . import initializations as leniax_init
from .lenia import LeniaIndividual
from .statistics import build_compute_stats_fn
from .kernels import get_kernels_and_mapping

QDMetrics = Dict[str, Dict[str, list]]


def build_eval_lenia_config_mem_optimized_fn(qd_config: Dict, fitness_coef: float = 1., fft: bool = True) -> Callable:
    max_run_iter = qd_config['run_params']['max_run_iter']
    world_params = qd_config['world_params']
    R = world_params['R']

    render_params = qd_config['render_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    kernels_params = qd_config['kernels_params']

    K, mapping = get_kernels_and_mapping(
        kernels_params, render_params['world_size'], world_params['nb_channels'], world_params['R'], fft
    )
    update_fn = leniax_helpers.build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = build_compute_stats_fn(world_params, render_params)

    def eval_lenia_config_mem_optimized(lenia_sols: List[LeniaIndividual]) -> List[LeniaIndividual]:
        qd_config = lenia_sols[0].qd_config

        _, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(qd_config, lenia_sols, fft)

        stats, _ = leniax_runner.run_scan_mem_optimized(
            *run_scan_mem_optimized_parameters, max_run_iter, R, update_fn, compute_stats_fn
        )
        stats['N'].block_until_ready()

        results = leniax_helpers.update_individuals(lenia_sols, stats, fitness_coef)

        return results

    return eval_lenia_config_mem_optimized


def run_qd_search(
    rng_key: jax.random.KeyArray,
    qd_config: Dict,
    optimizer: Optimizer,
    fitness_domain: Tuple[int, int],
    eval_fn: Callable,
    log_freq: int = 1,
    n_workers: int = -1
) -> Tuple[jax.random.KeyArray, QDMetrics]:
    """Run a Quality-diveristy search

    .. Warning::
        n_workers == -1 means that your evaluation functions handles parallelism
        n_workers == 0 means that you want to use a sinple python loop function
        n_workers > 0 means that you want to use python spawn mechanism

    Args:
        rng_key: jax PRNGKey
        qd_config: QD configuration
        optimizer: pyribs Optimizer
        fitness_domain: a 2-tuple of ints representing the fitness bounds
        eval_fn: The evaluation function
        log_freq: Logging frequency
        n_workers: Number of workers used to eval a set of candidate solutions

    Returns:
        A 2-tuple representing a jax PRNGkey and search metrics
    """
    DEBUG = False
    if DEBUG is True:
        print('!!!! DEBUGGING MODE !!!!')
        if n_workers >= 0:
            print('!!!! n_workers set to 0 !!!!')
            n_workers == 0

    metrics: QDMetrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Max Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Size": {
            "x": [0],
            "y": [0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
    }
    print(f"{'iter':>6}{'coverage':>30}{'mean':>20}{'std':>20}{'min':>16}{'max':>16}{'QD Score':>20}{'Duration':>20}")

    nb_total_bins = optimizer.archive._bins
    nb_iter = qd_config['algo']['budget'] // (qd_config['algo']['batch_size'] * len(optimizer._emitters))
    for itr in range(1, nb_iter + 1):
        t0 = time.time()
        # Request models from the optimizer.
        sols = optimizer.ask()
        rng_key, *subkeys = jax.random.split(rng_key, 1 + len(sols))
        lenia_sols = [LeniaIndividual(qd_config, subkey, params) for subkey, params in zip(subkeys, sols)]

        # Evaluate the models and record the objectives and BCs.
        results: List
        if n_workers == -1:
            # Beware in this case, we expect the evaluation function to handle a list of solutions
            results = eval_fn(lenia_sols)
        elif n_workers == 0:
            results = list(map(eval_fn, lenia_sols))
        else:
            with get_context("spawn").Pool(processes=n_workers) as pool:
                results = pool.map(eval_fn, lenia_sols)

        # Send the results back to the optimizer.
        fits, bcs, metadata = [], [], []
        for _, ind in enumerate(results):
            fits.append(ind.fitness)
            bcs.append(ind.features)
            metadata.append(ind.get_config())
        optimizer.tell(fits, bcs, metadata)

        # Log statistics
        if itr % log_freq == 0 or itr == nb_iter:
            df = optimizer.archive.as_pandas(include_solutions=False)
            metrics["QD Score"]["x"].append(itr)
            qd_score = (df['objective'] - fitness_domain[0]).sum() / (fitness_domain[1] - fitness_domain[0])
            qd_score /= nb_total_bins
            metrics["QD Score"]["y"].append(qd_score)

            metrics["Max Score"]["x"].append(itr)
            min_score = df["objective"].min()
            max_score = df["objective"].max()
            mean_score = df["objective"].mean()
            std_score = df["objective"].std()
            metrics["Max Score"]["y"].append(max_score)

            metrics["Archive Size"]["x"].append(itr)
            metrics["Archive Size"]["y"].append(len(df))
            metrics["Archive Coverage"]["x"].append(itr)
            coverage = len(df) / nb_total_bins * 100
            metrics["Archive Coverage"]["y"].append(coverage)

            save_all(itr, optimizer, fitness_domain, sols, fits, bcs)

            nb_best = 0
            for idx in optimizer.archive._occupied_indices:
                if optimizer.archive._objective_values[idx] >= fitness_domain[1]:
                    nb_best += 1
            nb_best_str = f"({nb_best}) - "
            print(
                f"{itr:>4}/{nb_iter:<4} {nb_best_str + str(len(df)):>22}/{nb_total_bins:<6}{mean_score:>20.4f}"
                f"{std_score:>20.4f}{min_score:>16.4f}{max_score:>16.4f}{qd_score:>16.4f}"
                f"{time.time() - t0:>20.4f}"
            )

    return rng_key, metrics


###
# QD Utils
###
def load_qd_grid_and_config(grid_fullpath):
    with open(grid_fullpath, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, ArchiveBase):
            grid = data
            try:
                qd_config = grid.qd_config
            except Exception:
                # backward compatibility
                qd_config = grid.base_config
        else:
            if 'container' in data:
                grid = data['container']
            elif 'algorithms' in data:
                grid = data['algorithms'][0].container
            else:
                raise ValueError('Unknown data')
            qd_config = grid[0].qd_config

    return grid, qd_config


def dump_best(grid: ArchiveBase, fitness_threshold: float):
    qd_config = grid.qd_config
    seed = qd_config['run_params']['seed']
    rng_key = leniax_utils.seed_everything(seed)

    real_bests = []
    for idx in grid._occupied_indices:
        if abs(grid._objective_values[idx]) >= fitness_threshold:
            rng_key, subkey = jax.random.split(rng_key)
            lenia = LeniaIndividual(grid._metadata[idx], subkey, grid._solutions[idx])
            real_bests.append(lenia)

    print(f"Found {len(real_bests)} beast!")

    nb_cpus = psutil.cpu_count(logical=False) - 1
    with get_context("spawn").Pool(processes=nb_cpus) as pool:
        pool.map(render_found_lenia, enumerate(real_bests))
    # for lenia_tuple in enumerate(real_bests):
    #     render_found_lenia(lenia_tuple)


def render_found_lenia(enum_lenia: Tuple[int, LeniaIndividual]):
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    id_lenia, lenia = enum_lenia
    padded_id = str(id_lenia).zfill(4)
    config = lenia.get_config()

    save_dir = os.path.join(os.getcwd(), f"c-{padded_id}")  # changed by hydra
    # if os.path.isdir(save_dir):
    #     print(f"Folder for id: {padded_id}, alreaady exist. Passing")
    #     # If the lenia folder already exists at that path, we consider the work alreayd done
    #     return
    leniax_utils.check_dir(save_dir)

    if 'init_rng_key' in config['algo']:
        init_slug = config['algo']['init_slug']
        _, noises = leniax_init.register[init_slug](
            jnp.array(config['algo']['init_rng_key']),
            config['run_params']['nb_init_search'] * config['world_params']['nb_channels'],
            config['render_params']['world_size'],
            config['world_params']['R'],
            config['kernels_params'][0]['k_params'],
            config['kernels_params'][0]['gf_params']
        )
        init_noises_shape = [config['run_params']['nb_init_search'], config['world_params']['nb_channels']
                             ] + config['render_params']['world_size']
        init_noises = noises.reshape(init_noises_shape)
        config['run_params']['init_cells'] = init_noises[config['algo']['best_init_idx']]

    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(config, use_init_cells=True, with_jit=True, fft=True)
    all_cells = all_cells[:int(stats_dict['N']), 0]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"[{padded_id}] - {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)

    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


###
# QD Viz
###


def save_ccdf(archive, fullpath):
    """Saves a CCDF showing the distribution of the archive's objective values.

    CCDF = Complementary Cumulative Distribution Function (see
    https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)).
    The CCDF plotted here is not normalized to the range (0,1). This may help
    when comparing CCDF's among archives with different amounts of coverage
    (i.e. when one archive has more cells filled).

    Args:
        archive (GridArchive): Archive with results from an experiment.
        fullpath (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        archive.as_pandas(include_solutions=False)["objective"],
        50,  # Number of bins.
        histtype="step",
        density=False,
        cumulative=False
    )  # CCDF rather than CDF.
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Num. Entries")
    ax.set_title("Distribution of Archive Objective Values")
    fig.savefig(fullpath)
    plt.close(fig)


def save_metrics(metrics, save_dir):
    # Plots.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        fig.savefig(os.path.join(save_dir, f"{metric}.png"))
        plt.close(fig)

    # JSON file.
    with open(os.path.join(save_dir, 'metrics.json'), "w") as file:
        json.dump(metrics, file, indent=2)


def save_heatmap(archive, fitness_domain, fullpath):
    if isinstance(archive, GridArchive):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid_archive_heatmap(archive, square=False, vmin=fitness_domain[0], vmax=fitness_domain[1], ax=ax)
    elif isinstance(archive, CVTArchive):
        fig, ax = plt.subplots(figsize=(8, 6))
        cvt_archive_heatmap(archive, vmin=fitness_domain[0], vmax=fitness_domain[1], ax=ax)

    ax.set_title("heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)


def save_parallel_axes_plot(archive, fitness_domain, fullpath):
    fig, ax = plt.subplots(figsize=(8, 6))

    parallel_axes_plot(archive, vmin=fitness_domain[0], vmax=fitness_domain[1], ax=ax, sort_archive=True)

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)


def save_emitter_samples(archive, fitness_domain, sols, fits, bcs, fullpath, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    bcs_jnp = jnp.array(bcs)

    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    t = ax.scatter(bcs_jnp.T[0], bcs_jnp.T[1], c=fits, cmap="magma", vmin=fitness_domain[0], vmax=fitness_domain[1])
    ax.figure.colorbar(t, ax=ax, pad=0.1)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)


def save_all(current_iter: int, optimizer, fitness_domain, sols: List, fits: List, bcs: List):
    save_dir = os.getcwd()
    prefix_fullpath = os.path.join(save_dir, f"{str(current_iter).zfill(4)}-")
    save_ccdf(optimizer.archive, f"{prefix_fullpath}archive_ccdf.png")
    save_heatmap(optimizer.archive, fitness_domain, f"{prefix_fullpath}archive_heatmap.png")

    pos = 0
    for emitter_id, emitter in enumerate(optimizer.emitters):
        end = pos + emitter.batch_size
        save_emitter_samples(
            optimizer.archive,
            fitness_domain,
            sols[pos:end],
            fits[pos:end],
            bcs[pos:end],
            f"{prefix_fullpath}emitter_{emitter_id}.png",
            type(emitter).__name__
        )
        pos = end

    # Save Archive
    with open(f"{prefix_fullpath}final.p", 'wb') as handle:
        pickle.dump(optimizer.archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if current_iter - 1 > 0:
        previous_prefix_fullpath = os.path.join(save_dir, f"{str(current_iter-1).zfill(4)}-")
        previous_fullpath = f"{previous_prefix_fullpath}final.p"
        if os.path.isfile(previous_fullpath):
            os.remove(previous_fullpath)


###
# QD Debug
###
def rastrigin(pos: jnp.ndarray, A: float = 10.):
    """Valu are expected to be between [0, 1]"""
    assert len(pos.shape) == 3
    assert pos.shape[-1] == 2

    pi2 = 2 * math.pi

    scaled_pos = 8 * pos - 4
    z = jnp.sum(scaled_pos**2 - A * np.cos(pi2 * scaled_pos), axis=-1)

    return 2 * A + z


def eval_debug(ind: LeniaIndividual, fitness_coef=1.):
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    fitness = fitness_coef * rastrigin(jnp.array(ind.params)[jnp.newaxis, jnp.newaxis, :]).squeeze()
    features = [ind.params[0] * 8 - 4, ind.params[1] * 8 - 4]

    ind.fitness = fitness
    ind.features = features

    return ind
