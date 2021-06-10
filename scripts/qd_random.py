import os
import logging
from omegaconf import DictConfig
import hydra
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager
from qdpy.algorithms.search import Sobol

from lenia.api import get_container
from lenia.qd import genBaseIndividual, eval_fn
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_random")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    save_dir = os.getcwd()  # changed by hydra
    # media_dir = os.path.join(save_dir, 'media')

    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 32
    config['run_params']['max_run_iter'] = 512

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])
    generator_builder = genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

    fitness_domain = [(0, config['run_params']['max_run_iter'])]
    # features_domain = [(0., 20000), (0., 20000), (0., 20000), (0., 20000)]
    features_domain = [(0., 1.), (0., 1.)]
    grid_shape = [20, 100]
    grid = containers.Grid(
        shape=grid_shape, max_items_per_bin=1, fitness_domain=fitness_domain, features_domain=features_domain
    )

    # GOAL: for the simple case where we need to fill 20 * 100 niches
    # We would like to be able to check at lease 20k individuals
    # currently:
    # - a run takes <2.5 seconds
    # => search for init < 1280 seconds (for 512 init)
    # I can efficiently parallelise up to <= nb_cores // 2, on my mac, it means 8
    # => look for 20k individuals < 40 jours...
    #
    # TODO: vmap and jit the search for init (hopefully winning a 10 time here)
    cpu_count = os.cpu_count()
    batch_size = 8
    if isinstance(cpu_count, int):
        max_workers = cpu_count // 2
    else:
        max_workers = 1
    dimension = len(config['params_and_domains'])
    algo = Sobol(
        container=grid,
        budget=2**8,  # Nb of generated individuals
        ind_domain=config['algo']['ind_domain'],
        batch_size=batch_size,  # how many to batch together
        dimension=dimension,  # Number of parameters that can be updated
        nb_objectives=None,  # With None, use the container fitness domain
        optimisation_task="max",
        base_ind_gen=lenia_generator,
        tell_container_at_init=False,
        add_default_logger=False,
        name="lenia-random",
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=max_workers) as pMgr:
        _ = algo.optimise(eval_fn, executor=pMgr.executor, batch_mode=False)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)
    # breakpoint()


if __name__ == '__main__':
    run()
