import os
import random
import logging
from omegaconf import DictConfig
import hydra
from qdpy import containers, algorithms
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
    random.seed(1)

    generator_builder = genBaseIndividual(config)
    lenia_generator = generator_builder()
    fitness_domain = [(0, config['run_params']['max_run_iter'])]
    features_domain = [(0., 20000), (0., 20000), (0., 20000), (0., 20000)]
    grid_shape = [5] * len(features_domain)

    grid = containers.Grid(
        shape=grid_shape, max_items_per_bin=1, fitness_domain=fitness_domain, features_domain=features_domain
    )

    algo = LeniaRandomUniform(
        container=grid,
        budget=10,  # Nb of generated individuals
        batch_size=1,  # how many to batch together
        # dimension=3,  # Number of parameters that can be updated, we don't use it
        nb_objectives=None,  # With None, use the container fitness domain
        optimisation_task="max",
        base_ind_gen=lenia_generator,
        tell_container_at_init=False,
        add_default_logger=False,
        name="lenia-random",
    )

    _ = algorithms.TQDMAlgorithmLogger(algo)

    with ParallelismManager("none") as pMgr:
        _ = algo.optimise(eval_fn, executor=pMgr.executor, batch_mode=False)

    print(f"Number of species found: {len(grid)}")
    breakpoint()


if __name__ == '__main__':
    run()
