import os
import logging
from omegaconf import DictConfig
import hydra
from functools import partial
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager
from qdpy.algorithms.evolution import CMAES

from lenia.api import get_container
from lenia.qd import genBaseIndividual, eval_fn
from lenia import utils as lenia_utils
from lenia import helpers as lenia_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_cmaes")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    save_dir = os.getcwd()  # changed by hydra
    # media_dir = os.path.join(save_dir, 'media')

    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 256
    config['run_params']['max_run_iter'] = 512

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])
    generator_builder = genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

    # cma-es algorithm can only minimize
    # BUT
    # qdpy negate fitness values line 305: _pop_fitness_vals += [-1. * x for x in individual.fitness.values]
    # -> it works well for maximisation
    # -> does not work at all for minimisation
    fitness_domain = [(0, config['run_params']['max_run_iter'])]
    features_domain = config['grid']['features_domain']  # Phenotype
    grid_shape = config['grid']['shape']  # Phenotype bins
    assert len(grid_shape) == len(features_domain)
    grid = containers.Grid(
        shape=grid_shape, max_items_per_bin=1, fitness_domain=fitness_domain, features_domain=features_domain
    )

    cpu_count = os.cpu_count()
    batch_size = 8
    if isinstance(cpu_count, int):
        max_workers = cpu_count // 2 - 1
    else:
        max_workers = 1
    dimension = len(config['genotype'])  # Number of genes
    optimisation_task = 'max'
    algo = CMAES(
        container=grid,
        budget=config['algo']['budget'],  # Nb of generated individuals
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
        name="lenia-cmaes",
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=max_workers) as pMgr:
        _ = algo.optimise(partial(eval_fn, neg_fitness=False), executor=pMgr.executor, batch_mode=False)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)

    lenia_helpers.dump_best(grid, config['run_params']['max_run_iter'])


if __name__ == '__main__':
    run()
