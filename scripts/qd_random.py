import os
import random
import logging
from omegaconf import DictConfig
import hydra
from qdpy import algorithms, containers
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager

from lenia.api import get_container
from lenia.qd import genBaseIndividual, LeniaRandomUniform, eval_fn

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_random")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    config = get_container(omegaConf)

    # TODO: clean those seeds machinery
    random.seed(config['run_params']['seed'])
    save_dir = os.getcwd()  # changed by hydra
    # media_dir = os.path.join(save_dir, 'media')

    generator_builder = genBaseIndividual(config)
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
    # => search for init < 1280 seconds
    # I can efficiently parallelise up to <= nb_cores // 2, on my mac, it means 8
    # => look for 20k individuals < 40 jours...
    #
    # TODO: vmap and jit the search for init (hopefully winning a 10 time here)
    cpu_count = os.cpu_count()
    if isinstance(cpu_count, int):
        batch_size = cpu_count // 2
    else:
        batch_size = 1
    algo = LeniaRandomUniform(
        container=grid,
        budget=256,  # Nb of generated individuals
        batch_size=batch_size,  # how many to batch together
        # dimension=3,  # Number of parameters that can be updated, we don't use it
        nb_objectives=None,  # With None, use the container fitness domain
        optimisation_task="max",
        base_ind_gen=lenia_generator,
        tell_container_at_init=False,
        add_default_logger=False,
        name="lenia-random",
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=batch_size) as pMgr:
        _ = algo.optimise(eval_fn, executor=pMgr.executor, batch_mode=False)

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)
    # breakpoint()


if __name__ == '__main__':
    run()
