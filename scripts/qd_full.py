import os
import copy
import logging
import hydra
from functools import partial
from omegaconf import DictConfig
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager
from qdpy.algorithms.base import Sq
from qdpy.algorithms.search import Sobol
from qdpy.algorithms.evolution import RandomSearchMutPolyBounded, CMAES
import jax

from lenia.api import get_container
from lenia.qd import genBaseIndividual, eval_fn
from lenia import utils as lenia_utils
from lenia import helpers as lenia_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_full")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    save_dir = os.getcwd()  # changed by hydra
    # media_dir = os.path.join(save_dir, 'media')

    config = get_container(omegaConf)
    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    config_algo1 = copy.deepcopy(config)
    config_algo1['run_params']['nb_init_search'] = 32
    config_algo1['run_params']['max_run_iter'] = 512
    rng_key, subkey = jax.random.split(rng_key)
    lenia_generator_algo1 = genBaseIndividual(config_algo1, subkey)()

    config_algo2 = copy.deepcopy(config)
    config_algo2['run_params']['nb_init_search'] = 128
    config_algo2['run_params']['max_run_iter'] = 512
    rng_key, subkey = jax.random.split(rng_key)
    lenia_generator_algo2 = genBaseIndividual(config_algo2, subkey)()

    config_algo3 = copy.deepcopy(config)
    config_algo3['run_params']['nb_init_search'] = 512
    config_algo3['run_params']['max_run_iter'] = 512
    rng_key, subkey = jax.random.split(rng_key)
    lenia_generator_algo3 = genBaseIndividual(config_algo3, subkey)()

    # CMAES algorithm can only minimize
    fitness_domain = [(-config['run_params']['max_run_iter'], 0)]
    features_domain = config['grid']['features_domain']
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)
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
    batch_size = 2**4
    if isinstance(cpu_count, int):
        max_workers = cpu_count // 2
    else:
        max_workers = 1
    dimension = len(config['params_and_domains'])
    optimisation_task = 'min'
    algos = []
    algos.append(
        Sobol(
            container=grid,
            budget=2**7,  # Nb of generated individuals
            ind_domain=config['algo']['ind_domain'],
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator_algo1,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-sobol",
        )
    )
    algos.append(
        RandomSearchMutPolyBounded(
            container=grid,
            budget=2**7,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sel_pb=config['algo']['sel_pb'],
            init_pb=1 - config['algo']['sel_pb'],
            mut_pb=config['algo']['mut_pb'],
            eta=config['algo']['eta'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator_algo2,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-randomsearchmutpolybounded",
        )
    )
    algos.append(
        CMAES(
            container=grid,
            budget=2**4,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator_algo3,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmaes",
        )
    )
    algo = Sq(algos, tell_container_when_switching=True)

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=max_workers) as pMgr:
        _ = algo.optimise(partial(eval_fn, neg_fitness=True), executor=pMgr.executor, batch_mode=False)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)

    lenia_helpers.dump_best(grid, config_algo1['run_params']['max_run_iter'])


if __name__ == '__main__':
    run()
