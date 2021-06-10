import os
import logging
import math
from functools import partial
from omegaconf import DictConfig
import hydra
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager
from qdpy.algorithms.evolution import RandomSearchMutPolyBounded
from qdpy.algorithms.cmame import (
    MEMAPElitesUCB1, CMAMEOptimizingEmitter, CMAMEImprovementEmitter, CMAMERandomDirectionEmitter
)

from lenia.api import get_container
from lenia.qd import genBaseIndividual, eval_fn
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_memapelitesucb")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    save_dir = os.getcwd()  # changed by hydra
    # media_dir = os.path.join(save_dir, 'media')

    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 128
    config['run_params']['max_run_iter'] = 512

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])
    generator_builder = genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

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
    batch_size = 8
    if isinstance(cpu_count, int):
        max_workers = cpu_count // 2
    else:
        max_workers = 1
    dimension = len(config['params_and_domains'])
    optimisation_task = 'min'
    algos = []
    algos.append(
        RandomSearchMutPolyBounded(
            container=grid,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sel_pb=config['algo']['sel_pb'],
            init_pb=1 - config['algo']['sel_pb'],
            mut_pb=config['algo']['mut_pb'],
            eta=config['algo']['eta'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-randomsearchmutpolybounded",
        )
    )
    algos.append(
        CMAMEOptimizingEmitter(
            container=grid,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmameoptimizingemitter",
        )
    )
    algos.append(
        CMAMERandomDirectionEmitter(
            container=grid,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmamerandomdirectionemitter",
        )
    )
    algos.append(
        CMAMEImprovementEmitter(
            container=grid,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=config['algo']['ind_domain'],
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmameimprovementemitter",
        )
    )
    algo = MEMAPElitesUCB1(
        algos,
        budget=2**9,
        zeta=config['algo']['zeta'],
        initial_expected_rwds=config['algo']['initial_expected_rwds'],
        nb_active_emitters=len(algos),
        shuffle_emitters=True,
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=max_workers) as pMgr:
        _ = algo.optimise(partial(eval_fn, neg_fitness=True), executor=pMgr.executor, batch_mode=False)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)
    # breakpoint()


if __name__ == '__main__':
    run()
