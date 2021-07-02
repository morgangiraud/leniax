import json
import matplotlib.pyplot as plt
import os
from multiprocessing import get_context
from absl import logging
import random
import copy
from typing import Dict, Any, Tuple, Callable, List
import jax
import jax.numpy as jnp
import numpy as np
from alive_progress import alive_bar
from qdpy.phenotype import Individual

from ribs.archives import GridArchive, CVTArchive
from ribs.visualize import grid_archive_heatmap, cvt_archive_heatmap

from .api import search_for_init
from . import utils as lenia_utils
from .kernels import get_kernels_and_mapping
from .statistics import stats_list_to_dict, build_compute_stats_fn
from .core import build_update_fn, run, init_cells

QDMetrics = Dict[str, Dict[str, list]]


class genBaseIndividual(object):
    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        self.config = config
        self.rng_key = rng_key

    def __iter__(self):
        return self

    def __next__(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)

        return LeniaIndividual(self.config, subkey)

    def __call__(self):
        while (True):
            yield self.__next__()


class LeniaIndividual(Individual):
    # Individual inherits from list
    # The raw parameters can then be set with ind[:] = [param1,  ...]
    #
    # The philosophy of the lib is to have parameters sampled from the same domain
    # And then scaled by custom functions before being used in the evaluation function
    # To sum up:
    #   - All parameters are generated in the ind_domain
    #   - the dimension parameter is the number of parameter
    #   - in the eval function:
    #       1. You scale those parameters
    #       2. You create the configuration form those parameters
    #       3. You evaluate the configuration
    #       4. you set fitness and features
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


def eval_lenia_config(ind: LeniaIndividual, neg_fitness=False, qdpy=False) -> Tuple:
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

    _, runs = search_for_init(ind.rng_key, config, with_stats=True)

    best = runs[0]
    nb_steps = best['N']
    config['behaviours'] = stats_list_to_dict(best['all_stats'])
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    ind.set_init_cells(lenia_utils.compress_array(init_cells))

    if neg_fitness is True:
        fitness = -nb_steps
    else:
        fitness = nb_steps
    features = [get_param(config, key_string) for key_string in ind.base_config['phenotype']]

    if qdpy is True:
        ind.fitness.values = [fitness]
        ind.features.values = features
        return ind

    return fitness, features


def eval_lenia_init(ind: LeniaIndividual, neg_fitness=False) -> Tuple:
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
    update_fn = build_update_fn(world_params, K, mapping)
    compute_stats_fn = build_compute_stats_fn(config['world_params'], config['render_params'])
    noise = jnp.array(ind).reshape(K.shape)
    cells_0 = init_cells(world_size, nb_channels, [noise])
    all_cells, _, _, all_stats = run(cells_0, max_run_iter, update_fn, compute_stats_fn)

    nb_steps = len(all_cells)
    config['behaviours'] = stats_list_to_dict(all_stats)

    if neg_fitness is True:
        fitness = -nb_steps
    else:
        fitness = nb_steps
    features = [noise.mean(), noise.stddev()]

    return fitness, features


def run_qd_ribs_search(eval_fn: Callable, nb_iter: int, lenia_generator, optimizer, log_freq: int = 1) -> QDMetrics:
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
    cpu_count = os.cpu_count()
    if isinstance(cpu_count, int):
        n_workers = max(cpu_count - 1, 1)
    else:
        n_workers = 1
    DEBUG = False

    # ray_eval_fn = ray.remote(eval_fn)
    print(f"{'iter':16}{'coverage':32}{'mean':32}{'std':32}{'min':16}{'max':16}{'QD Score':32}")
    with alive_bar(nb_iter) as update_bar:
        for itr in range(1, nb_iter + 1):
            # Request models from the optimizer.
            sols = optimizer.ask()
            lenial_sols = []
            for sol in sols:
                lenia = next(lenia_generator)
                lenia[:] = sol
                lenial_sols.append(lenia)

            # Evaluate the models and record the objectives and BCs.
            fits, bcs = [], []
            results: List
            if DEBUG:
                results = list(map(eval_fn, lenial_sols))
            else:
                # result_ids = [ray_eval_fn.remote(sol) for sol in lenial_sols]
                # while len(result_ids):
                #     done_id, result_ids = ray.wait(result_ids)
                #     fit, feat = ray.get(done_id[0])
                #     fits.append(fit)
                #     bcs.append(feat)
                with get_context("spawn").Pool(processes=n_workers) as pool:
                    results = pool.map(eval_fn, lenial_sols)
            for fit, feat in results:
                fits.append(fit)
                bcs.append(feat)

            # Send the results back to the optimizer.
            optimizer.tell(fits, bcs)

            # Logging.
            if not DEBUG:
                update_bar()

            if itr % log_freq == 0 or itr == nb_iter:
                df = optimizer.archive.as_pandas(include_solutions=False)
                metrics["QD Score"]["x"].append(itr)
                qd_score = df['objective'].sum()
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

                print(
                    f"{len(df):12}/{nb_total_bins}{mean_score:32}{std_score:32}{min_score:16}{max_score:16}{qd_score:32}"
                )

    return metrics


# QD Utils
def save_ccdf(archive, filename):
    """Saves a CCDF showing the distribution of the archive's objective values.

    CCDF = Complementary Cumulative Distribution Function (see
    https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)).
    The CCDF plotted here is not normalized to the range (0,1). This may help
    when comparing CCDF's among archives with different amounts of coverage
    (i.e. when one archive has more cells filled).

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots()
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
    fig.savefig(filename)


def save_metrics(outdir, metrics):
    """Saves metrics to png plots and a JSON file.

    Args:
        outdir (Path): output directory for saving files.
        metrics (dict): Metrics as output by run_search.
    """
    # Plots.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        fig.savefig(f"{outdir}/{metric.lower().replace(' ', '_')}.png")

    # JSON file.
    with open(f"{outdir}/metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)


def save_heatmap(archive, fitness_domain, save_dir):
    if isinstance(archive, GridArchive):
        fig, ax = plt.subplots(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=fitness_domain[0], vmax=fitness_domain[1], ax=ax)
    elif isinstance(archive, CVTArchive):
        fig, ax = plt.subplots(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=fitness_domain[0], vmax=fitness_domain[1], ax=ax)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'heatmap.png'))
    plt.close(fig)
