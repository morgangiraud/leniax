import os
from absl import logging
from omegaconf import DictConfig
import hydra
import jax.numpy as jnp
import jax
# import ray

from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from lenia.api import get_container
from lenia import utils as lenia_utils
from lenia import qd as lenia_qd
from lenia.kernels import get_kernels_and_mapping

# We are not using matmul and huge matrix, so we can avoid parallelising every operation
# This allow us to increase the numbre of parallel process
# https://github.com/google/jax/issues/743
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=2")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# ray.init(num_cpus=1, _redis_password="", local_mode=False)


@hydra.main(config_path=config_path, config_name="config_qd_init_search")
def run(omegaConf: DictConfig) -> None:
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    logging.set_verbosity(logging.ERROR)

    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 16
    config['run_params']['max_run_iter'] = 1024

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
    K, _ = get_kernels_and_mapping(
        config['kernels_params']['k'],
        config['render_params']['world_size'],
        config['world_params']['nb_channels'],
        config['world_params']['R'],
    )
    K = K[:, 0, 0, ...]
    _, subkey = jax.random.split(rng_key)
    genotype_dims = jnp.product(jnp.array(K.shape))

    initial_model = jax.random.uniform(subkey, K.shape)  # start the CMA-ES algorithm
    genotype_bounds = [(0., 1.) for _ in range(genotype_dims)]

    budget = 4
    log_freq = 1
    batch_size = 4
    sigma0 = config['algo']['sigma0']
    emitters = [
        GaussianEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 1, bounds=genotype_bounds
        ),
        ImprovementEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 2, bounds=genotype_bounds
        ),
        OptimizingEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 3, bounds=genotype_bounds
        ),
        RandomDirectionEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 4, bounds=genotype_bounds
        ),
    ]
    optimizer = Optimizer(archive, emitters)

    metrics = lenia_qd.run_qd_ribs_search(lenia_qd.eval_lenia_init, budget, lenia_generator, optimizer, log_freq)

    save_dir = os.getcwd()
    optimizer.archive.as_pandas().to_csv(f"{save_dir}/archive.csv")
    lenia_qd.save_ccdf(optimizer.archive, str(f"{save_dir}/archive_ccdf.png"))
    lenia_qd.save_metrics(save_dir, metrics)
    lenia_qd.save_heatmap(optimizer.archive, fitness_domain, save_dir)


if __name__ == '__main__':
    run()
