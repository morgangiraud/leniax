import os
import math
import pickle
from absl import logging
from omegaconf import DictConfig
import hydra
import jax.numpy as jnp

from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from leniax.utils import get_container
from leniax import qd as leniax_qd
from leniax import utils as leniax_utils
from leniax import video as leniax_video

# Disable JAX logging https://abseil.io/docs/python/guides/logging
logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_cmame")
# @hydra.main(config_path=config_path, config_name="config_qd_cmame_3c6k")
def run(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf, config_path)
    print(config)

    # Seed
    seed = config['run_params']['seed']
    rng_key = leniax_utils.seed_everything(seed)

    # Archive
    fitness_domain = [0, config['run_params']['max_run_iter']]
    features_domain = config['grid']['features_domain']
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)
    if True:
        archive = GridArchive(grid_shape, features_domain, seed=seed)
    else:
        bins = math.prod(grid_shape)
        archive = CVTArchive(bins, features_domain, seed=seed, use_kd_tree=True)
    archive.qd_config = config

    # Emitters
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
        ),
    ]

    # Optimizer
    optimizer = Optimizer(archive, emitters)

    # QD search
    eval_fn = leniax_qd.build_eval_lenia_config_mem_optimized_fn(config)
    nb_iter = config['algo']['budget'] // (batch_size * len(emitters))
    lenia_generator = leniax_qd.genBaseIndividual(config, rng_key)()
    log_freq = 1
    n_workers = -1
    metrics = leniax_qd.run_qd_search(eval_fn, nb_iter, lenia_generator, optimizer, fitness_domain, log_freq, n_workers)

    # Save results
    save_dir = os.getcwd()
    with open(f"{save_dir}/final.p", 'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
    leniax_utils.save_config(save_dir, archive.qd_config)

    leniax_qd.save_ccdf(optimizer.archive, f"{save_dir}/archive_ccdf.png")
    leniax_qd.save_metrics(metrics, save_dir)
    leniax_qd.save_heatmap(optimizer.archive, fitness_domain, f"{save_dir}/archive_heatmap.png")
    leniax_qd.save_parallel_axes_plot(optimizer.archive, fitness_domain, f"{save_dir}/archive_parralel_plot.png")
    leniax_video.dump_qd_ribs_result(os.path.join(save_dir, 'qd_search.mp4'))

    if config['other']['dump_bests'] is True:
        fitness_threshold = 0.7 * fitness_domain[1]
        leniax_qd.dump_best(optimizer.archive, fitness_threshold)


if __name__ == '__main__':
    run()
