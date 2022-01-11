import os
import psutil
import time
import pickle
import json
import math
import matplotlib.pyplot as plt
from multiprocessing import get_context
from absl import logging as absl_logging
import random
from typing import Dict, Callable, List
import jax
import jax.numpy as jnp
import numpy as np

from ribs.archives import ArchiveBase
from ribs.archives import GridArchive, CVTArchive
from ribs.visualize import grid_archive_heatmap, cvt_archive_heatmap, parallel_axes_plot

from .lenia import LeniaIndividual
from . import utils as leniax_utils
from . import loader as leniax_loader
from . import runner as leniax_runner
from . import helpers as leniax_helpers
from .statistics import build_compute_stats_fn
from .kernels import get_kernels_and_mapping

QDMetrics = Dict[str, Dict[str, list]]


class genBaseIndividual(object):
    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        self.config = config
        self.rng_key = rng_key

        self.fitness = None
        self.features = None

    def __iter__(self):
        return self

    def __next__(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)

        return LeniaIndividual(self.config, subkey)

    def __call__(self):
        while (True):
            yield self.__next__()


def rastrigin(pos: jnp.ndarray, A: float = 10.):
    """Valu are expected to be between [0, 1]"""
    assert len(pos.shape) == 3
    assert pos.shape[-1] == 2

    pi2 = 2 * math.pi

    scaled_pos = 8 * pos - 4
    z = jnp.sum(scaled_pos**2 - A * np.cos(pi2 * scaled_pos), axis=-1)

    return 2 * A + z


def eval_debug(ind: LeniaIndividual, neg_fitness=False):
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    if neg_fitness is True:
        fitness = -rastrigin(jnp.array(ind)[jnp.newaxis, jnp.newaxis, :]).squeeze()
    else:
        fitness = rastrigin(jnp.array(ind)[jnp.newaxis, jnp.newaxis, :]).squeeze()
    features = [ind[0] * 8 - 4, ind[1] * 8 - 4]

    ind.fitness = fitness
    ind.features = features

    return ind


def eval_lenia_config(ind: LeniaIndividual, neg_fitness: bool = False, fft: bool = True) -> LeniaIndividual:
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    config = ind.get_config()

    # We have a problem here for reproducibility
    # This function can be called in an other computer and os the for numpy and random
    # Whould be set again here.
    # We use the first element of the jax rng key to do so
    np.random.seed(ind.rng_key[0])
    random.seed(ind.rng_key[0])

    _, best, _ = leniax_helpers.search_for_init(ind.rng_key, config, fft)

    nb_steps = best['N']
    stats = best['all_stats']
    config['behaviours'] = {}
    for k in stats.keys():
        if k == 'N':
            continue
        config['behaviours'][k] = stats[k][-128:].mean()
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    final_cells = best['all_cells'][-1][:, 0, 0, ...]

    ind.set_cells(leniax_loader.compress_array(leniax_utils.crop_zero(final_cells)))
    ind.set_init_cells(leniax_loader.compress_array(init_cells))

    if neg_fitness is True:
        fitness = -nb_steps
    else:
        fitness = nb_steps
    features = [leniax_utils.get_param(config, key_string) for key_string in ind.qd_config['phenotype']]

    ind.fitness = fitness
    ind.features = features

    return ind


def build_eval_lenia_config_mem_optimized_fn(qd_config: Dict, neg_fitness: bool = False, fft: bool = True) -> Callable:
    max_run_iter = qd_config['run_params']['max_run_iter']
    world_params = qd_config['world_params']
    R = world_params['R']

    render_params = qd_config['render_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    kernels_params = qd_config['kernels_params']['k']
    K, mapping = get_kernels_and_mapping(
        kernels_params, render_params['world_size'], world_params['nb_channels'], world_params['R'], fft
    )

    update_fn_scale_1 = leniax_helpers.build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn_scale_1 = build_compute_stats_fn(world_params, render_params)

    def eval_lenia_config_mem_optimized(lenia_sols: List[LeniaIndividual]) -> List[LeniaIndividual]:
        qd_config = lenia_sols[0].qd_config
        _, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(qd_config, lenia_sols)

        stats, all_final_cells = leniax_runner.run_scan_mem_optimized(
            *run_scan_mem_optimized_parameters, max_run_iter, R, update_fn_scale_1, compute_stats_fn_scale_1
        )
        stats['N'].block_until_ready()
        all_final_cells.block_until_ready()

        # top 8
        # top8_compressible = leniax_loader.make_array_compressible(top8)
        # stats, all_final_cells = leniax_runner.run_scan_mem_optimized(
        #     *run_scan_mem_optimized_parameters, max_run_iter, R, update_fn_scale_2, compute_stats_fn_scale_2
        # )

        cells0s = run_scan_mem_optimized_parameters[0]
        results = leniax_helpers.update_individuals(lenia_sols, stats, cells0s, all_final_cells, neg_fitness)

        return results

    return eval_lenia_config_mem_optimized


def run_qd_search(
    eval_fn: Callable,
    nb_iter: int,
    lenia_generator,
    optimizer,
    fitness_domain,
    log_freq: int = 1,
    n_workers: int = -1
) -> QDMetrics:
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
    nb_total_bins = optimizer.archive._bins

    DEBUG = False
    if DEBUG is True:
        print('!!!! DEBUGGING MODE !!!!')
        if n_workers >= 0:
            print('!!!! n_workers set to 0 !!!!')
            n_workers == 0

    print(f"{'iter':>6}{'coverage':>30}{'mean':>20}{'std':>20}{'min':>16}{'max':>16}{'QD Score':>20}{'Duration':>20}")
    for itr in range(1, nb_iter + 1):
        t0 = time.time()
        # Request models from the optimizer.
        sols = optimizer.ask()
        lenia_sols = []
        for sol in sols:
            lenia = next(lenia_generator)
            lenia[:] = sol
            lenia_sols.append(lenia)

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

        fits, bcs, metadata = [], [], []
        for _, ind in enumerate(results):
            fits.append(ind.fitness)
            bcs.append(ind.features)
            metadata.append(ind.get_config())

        # Send the results back to the optimizer.
        optimizer.tell(fits, bcs, metadata)

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

    return metrics


# QD Utils
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


def dump_best(grid: ArchiveBase, fitness_threshold: float):
    qd_config = grid.qd_config
    seed = qd_config['run_params']['seed']
    rng_key = leniax_utils.seed_everything(seed)
    generator_builder = genBaseIndividual(qd_config, rng_key)
    lenia_generator = generator_builder()

    real_bests = []
    for idx in grid._occupied_indices:
        if abs(grid._objective_values[idx]) >= fitness_threshold:
            lenia = next(lenia_generator)
            lenia.qd_config = grid._metadata[idx]
            lenia[:] = grid._solutions[idx]
            real_bests.append(lenia)

    print(f"Found {len(real_bests)} beast!")

    nb_cpus = psutil.cpu_count(logical=False) - 1
    with get_context("spawn").Pool(processes=nb_cpus) as pool:
        pool.map(leniax_helpers.process_lenia, enumerate(real_bests))
    # for lenia_tuple in enumerate(real_bests):
    #     leniax_helpers.process_lenia(lenia_tuple)
