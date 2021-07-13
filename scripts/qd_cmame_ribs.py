import os
from absl import logging
from omegaconf import DictConfig
import hydra
import jax.numpy as jnp
import pickle
# import ray

from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from lenia.api import get_container
from lenia import qd as lenia_qd
from lenia import utils as lenia_utils
from lenia import helpers as lenia_helpers

# We are not using matmul on huge matrix, so we can avoid parallelising every operation
# This allow us to increase the numbre of parallel process
# https://github.com/google/jax/issues/743
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# ray.init(num_cpus=1, _redis_password="", local_mode=False)


@hydra.main(config_path=config_path, config_name="config_qd_cmame")
def run(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)

    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    logging.set_verbosity(logging.ERROR)

    seed = config['run_params']['seed']
    rng_key = lenia_utils.seed_everything(seed)
    generator_builder = lenia_qd.genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

    fitness_domain = [0, config['run_params']['max_run_iter']]
    features_domain = config['grid']['features_domain']
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)
    if True:
        archive = GridArchive(grid_shape, features_domain, seed=seed)
    else:
        bins = 1_000
        archive = CVTArchive(bins, features_domain, seed=seed, use_kd_tree=True)
    archive.base_config = config

    genotype_dims = len(config['genotype'])
    sampling_domain = config['algo']['sampling_domain']
    sampling_bounds = [sampling_domain for _ in range(genotype_dims)]
    initial_model = jnp.array([bounds[0] + 0.5 * (bounds[1] - bounds[0])
                               for bounds in sampling_bounds])  # start the CMA-ES algorithm

    batch_size = config['algo']['batch_size']
    log_freq = 1
    sigma0 = config['algo']['sigma0']
    emitters = [
        GaussianEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 1, bounds=sampling_bounds
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
    optimizer = Optimizer(archive, emitters)

    nb_iter = config['algo']['budget'] // (batch_size * len(emitters))
    metrics = lenia_qd.run_qd_ribs_search(
        lenia_qd.eval_lenia_config, nb_iter, lenia_generator, optimizer, fitness_domain, log_freq
    )
    # metrics = lenia_qd.run_qd_ribs_search(
    #     lenia_qd.debug_eval, nb_iter, lenia_generator, optimizer, fitness_domain, log_freq
    # )

    # Save results
    save_dir = os.getcwd()
    with open(f"{save_dir}/final.p", 'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    lenia_qd.save_ccdf(optimizer.archive, str(f"{save_dir}/archive_ccdf.png"))
    lenia_qd.save_metrics(save_dir, metrics)
    lenia_qd.save_heatmap(optimizer.archive, fitness_domain, save_dir)

    if config['other']['dump_bests'] is True:
        lenia_helpers.dump_best(optimizer.archive, config['run_params']['max_run_iter'], lenia_generator)


if __name__ == '__main__':
    run()
