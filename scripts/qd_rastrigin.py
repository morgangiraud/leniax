import os
import psutil
from absl import logging
from omegaconf import DictConfig
import hydra
import jax
import jax.numpy as jnp
import pickle
import math

from ribs.archives import CVTArchive
from ribs.emitters import GaussianEmitter, ImprovementEmitter, OptimizingEmitter, RandomDirectionEmitter
from ribs.optimizers import Optimizer

from lenia.helpers import get_container
from lenia import qd as lenia_qd
from lenia import utils as lenia_utils
from lenia import video as lenia_video

# Disable JAX logging https://abseil.io/docs/python/guides/logging
logging.set_verbosity(logging.ERROR)

# # We are not using matmul on huge matrix, so we can avoid parallelising every operation
# # This allow us to increase the numbre of parallel process
# # https://github.com/google/jax/issues/743
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_cmame")
def run(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)
    config['algo']['budget'] *= 2
    print(config)

    seed = config['run_params']['seed']
    rng_key = lenia_utils.seed_everything(seed)
    generator_builder = lenia_qd.genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

    fitness_domain = [-65, 0]  # negative rastrigin domain in(to maximise)
    features_domain = [[-4, 4], [-4, 4]]
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)

    bins = math.prod(grid_shape)
    archive = CVTArchive(bins, features_domain, seed=seed, use_kd_tree=True)
    archive.qd_config = config

    genotype_dims = len(config['genotype'])
    sampling_domain = config['algo']['sampling_domain']
    sampling_bounds = [sampling_domain for _ in range(genotype_dims)]
    initial_model = jnp.array([bounds[0] + 0.5 * (bounds[1] - bounds[0])
                               for bounds in sampling_bounds])  # start the CMA-ES algorithm

    batch_size = config['algo']['batch_size']
    log_freq = 1
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
    optimizer = Optimizer(archive, emitters)

    # See https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores
    n_workers = psutil.cpu_count(logical=False) - 1
    nb_iter = config['algo']['budget'] // (batch_size * len(emitters))
    eval_fn = jax.partial(lenia_qd.eval_debug, neg_fitness=True)
    metrics = lenia_qd.run_qd_search(eval_fn, nb_iter, lenia_generator, optimizer, fitness_domain, log_freq, n_workers)

    # Save results
    save_dir = os.getcwd()
    with open(f"{save_dir}/final.p", 'wb') as handle:
        pickle.dump(archive, handle, protocol=pickle.HIGHEST_PROTOCOL)
    lenia_utils.save_config(save_dir, archive.qd_config)

    lenia_qd.save_ccdf(optimizer.archive, f"{save_dir}/archive_ccdf.png")
    lenia_qd.save_metrics(metrics, save_dir)
    lenia_qd.save_heatmap(optimizer.archive, fitness_domain, f"{save_dir}/archive_heatmap.png")
    lenia_qd.save_parallel_axes_plot(optimizer.archive, fitness_domain, f"{save_dir}/archive_parralel_plot.png")
    lenia_video.dump_qd_ribs_result(os.path.join(save_dir, 'qd_search.mp4'))


if __name__ == '__main__':
    run()
