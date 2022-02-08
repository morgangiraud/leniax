"""Leniax: Quality-diversity search for stable local patterns

This example shows how to use the pyribs library with Leniax to search for
stable local patterns.
We are using the CMA-ME algorithm.

For more information on how pyribs work, please check their documentation.

Usage:
    ``python examples/run.py -cn config_name -cp config_path``
"""
import os
import math
import pickle
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import jax.numpy as jnp

from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from leniax import qd as leniax_qd
from leniax import utils as leniax_utils
from leniax import video as leniax_video

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)

###
# We use hydra to load Leniax configurations.
# It alloes to do many things among which, we can override configuraiton parameters.
# For example to render a Lenia in a bigger size:
# python examples/run.py render_params.world_size='[512, 512]' world_params.scale=4
###
cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_name = "config_qd_cmame"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)
    leniax_utils.print_config(config)

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    seed = config['run_params']['seed']
    rng_key = leniax_utils.seed_everything(seed)

    # In QD algorithm we maintain an archive of solutions which we iteratively improve
    # So first, we have to define the archive
    fitness_domain = (0, config['run_params']['max_run_iter'])
    features_domain = config['grid']['features_domain']
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)
    if True:
        archive = GridArchive(grid_shape, features_domain, seed=seed)
    else:
        bins = math.prod(grid_shape)
        archive = CVTArchive(bins, features_domain, seed=seed, use_kd_tree=True)
    archive.qd_config = config

    # Then, we define the emitters. Emitters represent different strategy to sample
    # new candidate solutions.
    genotype_dims = len(config['genotype'])
    sampling_domain = config['algo']['sampling_domain']
    sampling_bounds = [sampling_domain for _ in range(genotype_dims)]
    initial_model = jnp.array([bounds[0] + 0.5 * (bounds[1] - bounds[0])
                               for bounds in sampling_bounds])  # start the CMA-ES algorithm
    batch_size = config['algo']['batch_size']
    mut_sigma0 = config['algo']['mut_sigma0']
    sigma0 = config['algo']['sigma0']
    emitters = [
        GaussianEmitter(
            archive, initial_model.flatten(), mut_sigma0, batch_size=batch_size, seed=seed + 1, bounds=sampling_bounds
        ),
        OptimizingEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 2, bounds=sampling_bounds
        ),
        RandomDirectionEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 3, bounds=sampling_bounds
        ),
        ImprovementEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 4, bounds=sampling_bounds
        )
    ]
    # We setup the QD optimizer
    optimizer = Optimizer(archive, emitters)
    # And build the evaluation function
    eval_fn = leniax_qd.build_eval_lenia_config_mem_optimized_fn(config)

    # Finally we can launch the search
    # This function returns a new JAX PRNG key and the search metrics.
    # !!! The solutions are updated in the archive directly and are not returned.
    rng_key, metrics = leniax_qd.run_qd_search(rng_key, config, optimizer, fitness_domain, eval_fn, log_freq=1, n_workers=-1)

    # We save the whole configuration and the archive
    with open(f"{save_dir}/final.p", 'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
    leniax_utils.save_config(save_dir, archive.qd_config)

    # Finally, we render a fee key visuals to interpret the QD search
    leniax_qd.save_ccdf(optimizer.archive, f"{save_dir}/archive_ccdf.png")
    leniax_qd.save_metrics(metrics, save_dir)
    leniax_qd.save_heatmap(optimizer.archive, fitness_domain, f"{save_dir}/archive_heatmap.png")
    leniax_qd.save_parallel_axes_plot(optimizer.archive, fitness_domain, f"{save_dir}/archive_parralel_plot.png")
    leniax_video.render_qd_search(os.path.join(save_dir, 'qd_search.mp4'))

    # Optionnaly, you can also render the best results of the QD search.
    # Best is defined by fitness_threshold
    if config['other']['render_bests'] is True:
        fitness_threshold = 0.7 * fitness_domain[1]
        leniax_qd.render_best(optimizer.archive, fitness_threshold)


if __name__ == '__main__':
    run()
