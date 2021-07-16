import os
import logging
from omegaconf import DictConfig
import hydra
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager
from qdpy.algorithms.evolution import RandomSearchMutPolyBounded

from lenia.helpers import get_container
from lenia.qd import genBaseIndividual, eval_lenia_config
from lenia import utils as lenia_utils
from lenia import qd as lenia_qd

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_evo")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 512
    config['run_params']['max_run_iter'] = 1024

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])
    generator_builder = genBaseIndividual(config, rng_key)
    lenia_generator = generator_builder()

    fitness_domain = [(0, config['run_params']['max_run_iter'])]
    features_domain = config['grid']['features_domain']
    grid_shape = config['grid']['shape']
    assert len(grid_shape) == len(features_domain)
    grid = containers.Grid(
        shape=grid_shape, max_items_per_bin=1, fitness_domain=fitness_domain, features_domain=features_domain
    )

    cpu_count = os.cpu_count()
    batch_size = 8
    if isinstance(cpu_count, int):
        max_workers = max(cpu_count // 2 - 1, 1)
    else:
        max_workers = 1
    dimension = len(config['genotype'])
    optimisation_task = 'max'
    algo = RandomSearchMutPolyBounded(
        container=grid,
        budget=config['algo']['budget'],  # Nb of generated individuals
        ind_domain=config['algo']['sampling_domain'],
        batch_size=batch_size,  # how many to batch together
        dimension=dimension,  # Number of parameters that can be updated, we don't use it
        nb_objectives=None,  # With None, use the container fitness domain
        optimisation_task=optimisation_task,
        base_ind_gen=lenia_generator,
        tell_container_at_init=False,
        add_default_logger=False,
        name="lenia-randomsearchmutpolybounded",
        # RandomSearchMutPolyBounded parameters
        sel_pb=config['algo']['sel_pb'],
        init_pb=1 - config['algo']['sel_pb'],
        mut_pb=config['algo']['mut_pb'],
        eta=config['algo']['eta'],
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=max_workers) as pMgr:
        _ = algo.optimise(
            eval_lenia_config,
            executor=pMgr.executor,
            batch_mode=False  # Calling the optimisation loop per batch if True, else calling it once with total budget
        )

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    save_dir = os.getcwd()
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)

    if config['other']['dump_bests'] is True:
        lenia_qd.dump_best(grid, config['run_params']['max_run_iter'])


if __name__ == '__main__':
    run()
