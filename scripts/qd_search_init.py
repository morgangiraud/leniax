import os
import pickle
from absl import logging
from omegaconf import DictConfig
import hydra
import jax.numpy as jnp
import jax

from ribs.archives import GridArchive, CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from lenia.helpers import get_container
from lenia.kernels import get_kernels_and_mapping
from lenia import utils as lenia_utils
from lenia import qd as lenia_qd
from lenia import video as lenia_video

# Disable JAX logging https://abseil.io/docs/python/guides/logging
logging.set_verbosity(logging.ERROR)

# # We are not using matmul and huge matrix, so we can avoid parallelising every operation
# # This allow us to increase the numbre of parallel process
# # https://github.com/google/jax/issues/743
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=2")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_init_search")
def run(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 16
    config['run_params']['max_run_iter'] = 1024
    print(config)

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
    sampling_bounds = [(0., 1.) for _ in range(genotype_dims)]

    budget = 4
    log_freq = 1
    batch_size = 4
    mut_sigma0 = config['algo']['mut_sigma0']
    sigma0 = config['algo']['sigma0']
    emitters = [
        GaussianEmitter(
            archive, initial_model.flatten(), mut_sigma0, batch_size=batch_size, seed=seed + 1, bounds=sampling_bounds
        ),
        ImprovementEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 2, bounds=sampling_bounds
        ),
        OptimizingEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 3, bounds=sampling_bounds
        ),
        RandomDirectionEmitter(
            archive, initial_model.flatten(), sigma0, batch_size=batch_size, seed=seed + 4, bounds=sampling_bounds
        ),
    ]
    optimizer = Optimizer(archive, emitters)

    eval_fn = lenia_qd.eval_lenia_init
    metrics = lenia_qd.run_qd_search(eval_fn, budget, lenia_generator, optimizer, fitness_domain, log_freq)

    save_dir = os.getcwd()
    with open(f"{save_dir}/final.p", 'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lenia_utils.save_config(save_dir, archive.base_config)

    lenia_qd.save_ccdf(optimizer.archive, f"{save_dir}/archive_ccdf.png")
    lenia_qd.save_metrics(metrics, save_dir)
    lenia_qd.save_heatmap(optimizer.archive, fitness_domain, f"{save_dir}/archive_heatmap.png")
    lenia_qd.save_parallel_axes_plot(optimizer.archive, fitness_domain, f"{save_dir}/archive_parralel_plot.png")
    lenia_video.dump_qd_ribs_result(os.path.join(save_dir, 'qd_search.mp4'))


if __name__ == '__main__':
    run()
