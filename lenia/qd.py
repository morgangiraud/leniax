import os
import time
import pickle
import json
import math
import matplotlib.pyplot as plt
from multiprocessing import get_context
from absl import logging
import random
import copy
from typing import Dict, Any, Tuple, Callable, List
import jax
import jax.numpy as jnp
import numpy as np
from alive_progress import alive_bar

from ribs.archives import ArchiveBase
from ribs.archives import GridArchive, CVTArchive
from ribs.visualize import grid_archive_heatmap, cvt_archive_heatmap, parallel_axes_plot

from . import utils as lenia_utils
from . import core as lenia_core
from . import video as lenia_video
from . import helpers as lenia_helpers
from .growth_functions import growth_fns
from .kernels import get_kernels_and_mapping, draw_kernels
from .statistics import build_compute_stats_fn

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


class LeniaIndividual(list):
    # Individual inherits from list
    # The raw parameters can then be set with ind[:] = [param1,  ...]
    #
    # The philosophy of the lib is to have parameters sampled from the same domain
    # And then scaled by custom functions before being used in the evaluation function
    # To sum up:
    #   - All parameters are generated in the sampling_domain
    #   - the dimension parameter is the number of parameter
    #   - in the eval function:
    #       1. You scale those parameters
    #       2. You create the configuration from those parameters
    #       3. You evaluate the configuration
    #       4. you set fitness and features

    fitness: float
    features: List[float]

    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        super().__init__()

        self.base_config = config

        self.rng_key = rng_key

    def set_init_cells(self, init_cells: str):
        self.base_config['run_params']['cells'] = init_cells

    def get_config(self) -> Dict:
        p_and_ds = self.get_genotype()
        raw_values = list(self)
        assert len(raw_values) == len(p_and_ds)

        to_update = get_update_config(p_and_ds, raw_values)
        config = update_config(self.base_config, to_update)

        return config

    def get_genotype(self):
        return self.base_config['genotype']


def get_param(dic: Dict, key_string: str) -> Any:
    keys = key_string.split('.')
    for key in keys:
        if key.isdigit():
            dic = dic[int(key)]
        else:
            dic = dic[key]

    if isinstance(dic, jnp.ndarray):
        val = float(jnp.mean(dic))
    else:
        val = dic

    return val


def linear_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] + (domain[1] - domain[0]) * raw_value

    return val


def log_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] * (domain[1] / domain[0])**raw_value

    return val


def update_dict(dic: Dict, key_string: str, value: Any):
    keys = key_string.split('.')
    for i, key in enumerate(keys[:-1]):
        if key.isdigit():
            if isinstance(dic, list):
                idx = int(key)
                if len(dic) < idx + 1:
                    for _ in range(idx + 1 - len(dic)):
                        dic.append({})
                dic = dic[idx]
            else:
                raise Exception('This key should be an array')
        else:
            if keys[i + 1].isdigit():
                dic = dic.setdefault(key, [{}])
            else:
                dic = dic.setdefault(key, {})

    dic[keys[-1]] = value


def get_update_config(genotype: Dict, raw_values: list) -> Dict:
    to_update: Dict = {}
    for p_and_d, raw_val in zip(genotype, raw_values):
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        val_type = p_and_d['type']
        if val_type == 'float':
            val = float(linear_scale(raw_val, domain))
        elif val_type == 'int':
            d = (domain[0], domain[1] + 1)
            val = int(linear_scale(raw_val, d) - 0.5)
        elif val_type == 'choice':
            d = (0, len(domain))
            val = domain[int(linear_scale(raw_val, d) - 0.5)]
        else:
            raise ValueError(f"type {val_type} unknown")

        update_dict(to_update, key_str, val)

    return to_update


def update_config(old_config, to_update):
    new_config = copy.deepcopy(old_config)
    if 'kernels_params' in to_update:
        for i, kernel in enumerate(to_update['kernels_params']['k']):
            new_config['kernels_params']['k'][i].update(kernel)
    if 'world_params' in to_update:
        new_config['world_params'].update(to_update['world_params'])

    return new_config


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
    logging.set_verbosity(logging.ERROR)

    if neg_fitness is True:
        fitness = -rastrigin(jnp.array(ind)[jnp.newaxis, jnp.newaxis, :]).squeeze()
    else:
        fitness = rastrigin(jnp.array(ind)[jnp.newaxis, jnp.newaxis, :]).squeeze()
    features = [ind[0] * 8 - 4, ind[1] * 8 - 4]

    ind.fitness = fitness
    ind.features = features

    return ind


def eval_lenia_config(ind: LeniaIndividual, neg_fitness=False) -> LeniaIndividual:
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    logging.set_verbosity(logging.ERROR)

    config = ind.get_config()

    # We have a problem here for reproducibility
    # This function can be called in an other computer and os the for numpy and random
    # Whould be set again here.
    # We use the first element of the jax rng key to do so
    np.random.seed(ind.rng_key[0])
    random.seed(ind.rng_key[0])

    _, best, _ = lenia_helpers.search_for_init(ind.rng_key, config)

    nb_steps = best['N']
    config['behaviours'] = best['all_stats']
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    ind.set_init_cells(lenia_utils.compress_array(init_cells))

    if neg_fitness is True:
        fitness = -nb_steps
    else:
        fitness = nb_steps
    features = [get_param(config, key_string) for key_string in ind.base_config['phenotype']]

    ind.fitness = fitness
    ind.features = features

    return ind


def eval_lenia_config_mem_optimized(lenia_sols: List[LeniaIndividual], neg_fitness=False) -> List[LeniaIndividual]:
    base_config = lenia_sols[0].base_config
    rng_key = lenia_sols[0].rng_key
    rng_key, run_scan_mem_optimized_parameters = lenia_helpers.get_mem_optimized_inputs(
        rng_key, base_config, lenia_sols
    )

    Ns = lenia_core.run_scan_mem_optimized(*run_scan_mem_optimized_parameters)
    Ns.block_until_ready()

    cells0s = run_scan_mem_optimized_parameters[0]
    results = lenia_helpers.update_individuals(base_config, lenia_sols, Ns, cells0s, neg_fitness)

    return results


def eval_lenia_init(ind: LeniaIndividual, neg_fitness=False) -> LeniaIndividual:
    # This function is usually called un sub-process, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    logging.set_verbosity(logging.ERROR)

    config = ind.get_config()
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    max_run_iter = run_params['max_run_iter']

    # We have a problem here for reproducibility
    # This function can be called in an other computer and os the for numpy and random
    # Whould be set again here.
    # We use the first element of the jax rng key to do so
    np.random.seed(ind.rng_key[0])
    random.seed(ind.rng_key[0])

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    update_fn = lenia_core.build_update_fn(world_params, K.shape, mapping)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    compute_stats_fn = build_compute_stats_fn(config['world_params'], config['render_params'])
    noise = jnp.array(ind).reshape(K.shape)
    cells_0 = lenia_core.init_cells(world_size, nb_channels, [noise])

    all_cells, _, _, all_stats = lenia_core.run(
        cells_0, K, gfn_params, kernels_weight_per_channel, max_run_iter, update_fn, compute_stats_fn
    )

    nb_steps = len(all_cells)
    config['behaviours'] = all_stats

    if neg_fitness is True:
        fitness = -nb_steps
    else:
        fitness = nb_steps
    features = [noise.mean(), noise.stddev()]

    ind.fitness = fitness
    ind.features = features

    return ind


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

    print(f"{'iter'}{'coverage':>32}{'mean':>20}{'std':>20}{'min':>16}{'max':>16}{'QD Score':>20}")
    with alive_bar(nb_iter) as update_bar:
        for itr in range(1, nb_iter + 1):
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
            for i, ind in enumerate(results):
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

                print(
                    f"{len(df):>23}/{nb_total_bins:<6}{mean_score:>20.6f}"
                    f"{std_score:>20.6f}{min_score:>16.6f}{max_score:>16.6f}{qd_score:>20.6f}"
                )

            # Logging.
            if not DEBUG:
                update_bar()

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

    with open(f"{prefix_fullpath}final.p", 'wb') as handle:
        pickle.dump(optimizer.archive, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_best(grid: ArchiveBase, fitness_threshold: float):
    base_config = grid.base_config
    seed = base_config['run_params']['seed']
    rng_key = lenia_utils.seed_everything(seed)
    generator_builder = genBaseIndividual(base_config, rng_key)
    lenia_generator = generator_builder()

    real_bests = []
    for idx in grid._occupied_indices:
        if abs(grid._objective_values[idx]) >= fitness_threshold:
            lenia = next(lenia_generator)
            lenia.base_config = grid._metadata[idx]
            lenia[:] = grid._solutions[idx]
            real_bests.append(lenia)

    print(f"Found {len(real_bests)} beast!")

    for id_best, best in enumerate(real_bests):
        config = best.get_config()

        render_params = config['render_params']

        start_time = time.time()
        all_cells, _, _, all_stats = lenia_helpers.init_and_run(config)
        total_time = time.time() - start_time

        nb_iter_done = len(all_cells)
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

        save_dir = os.path.join(os.getcwd(), str(id_best))  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        # print('Dumping cells')
        # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
        #     np.save(f, np.array(all_cells)[:, 0, 0, ...])

        print('Plotting stats and functions')
        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

        lenia_utils.plot_stats(save_dir, all_stats)
        _, K, _ = lenia_core.init(config)
        draw_kernels(K, save_dir, colormap)
        for i, kernel in enumerate(config['kernels_params']['k']):
            lenia_utils.plot_gfunction(
                save_dir, i, growth_fns[kernel['gf_id']], kernel['m'], kernel['s'], config['world_params']['T']
            )

        print('Dumping video')
        lenia_video.dump_video(all_cells, render_params, media_dir, colormap)
