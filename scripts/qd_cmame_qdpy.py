import os
from functools import partial
from absl import logging
from omegaconf import DictConfig
import hydra
import math

from qdpy.containers import Grid, CVTGrid
from qdpy.algorithms.cmame import CMAMEOptimizingEmitter, CMAMEImprovementEmitter, CMAMERandomDirectionEmitter
from qdpy.algorithms.evolution import RandomSearchMutPolyBounded
from qdpy.algorithms.cmame import MEMAPElitesUCB1
from qdpy import algorithms
from qdpy import plots as qdpy_plots
from qdpy.base import ParallelismManager

from lenia.api import get_container
from lenia import qd as lenia_qd
from lenia import utils as lenia_utils
from lenia import helpers as lenia_helpers

# We are not using matmul on huge matrix, so we can avoid parallelising every operation
# This allow us to increase the numbre of parallel process
# https://github.com/google/jax/issues/743
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_qd_cmame")
def run(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)

    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    logging.set_verbosity(logging.ERROR)

    seed = config['run_params']['seed']
    rng_key = lenia_utils.seed_everything(seed)
    generator_builder = lenia_qd.genBaseIndividual(config, rng_key)
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
    if True:
        archive = Grid(
            shape=grid_shape, max_items_per_bin=1, fitness_domain=fitness_domain, features_domain=features_domain
        )
    else:
        bins = 1_000
        archive = CVTGrid(bins, features_domain, seed=seed, use_kd_tree=True)

    batch_size = config['algo']['batch_size']
    dimension = len(config['genotype'])  # Number of genes
    optimisation_task = 'max'
    # Domain for genetic parameters, to be used in conjunction with a projecting function to reach phenotype domain
    sampling_domain = config['algo']['sampling_domain']
    emitters = [
        RandomSearchMutPolyBounded(
            container=archive,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=sampling_domain,
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
        ),
        CMAMEOptimizingEmitter(
            container=archive,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=sampling_domain,
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmameoptimizingemitter",
        ),
        CMAMERandomDirectionEmitter(
            container=archive,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=sampling_domain,
            sigma0=config['algo']['sigma0'],
            separable_cma=config['algo']['separable_cma'],
            ignore_if_not_added_to_container=config['algo']['ignore_if_not_added_to_container'],
            nb_objectives=None,  # With None, use the container fitness domain
            optimisation_task=optimisation_task,
            base_ind_gen=lenia_generator,
            tell_container_at_init=False,
            add_default_logger=False,
            name="lenia-cmamerandomdirectionemitter",
        ),
        CMAMEImprovementEmitter(
            container=archive,
            budget=math.inf,  # Nb of generated individuals
            batch_size=batch_size,  # how many to batch together
            dimension=dimension,  # Number of parameters that can be updated, we don't use it
            ind_domain=sampling_domain,
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
    ]
    budget = config['algo']['budget'] // (batch_size * len(emitters)) * (batch_size * len(emitters))
    algo = MEMAPElitesUCB1(
        emitters,
        budget=budget,  # Nb of generated individuals
        batch_size=batch_size * len(emitters),
        zeta=config['algo']['zeta'],
        initial_expected_rwds=config['algo']['initial_expected_rwds'],
        nb_active_emitters=len(emitters),  # number of emitters active per global batch,
        # If nb_active_emitters equal to the number of emitters, all emitters are used the same number of times,
        # set to 1 in the original paper
        shuffle_emitters=False,
        batch_mode=True,  # If true, switch algo only after batch_size suggestions, else after every suggestions
    )

    logger = algorithms.TQDMAlgorithmLogger(algo)

    cpu_count = os.cpu_count()
    if isinstance(cpu_count, int):
        n_workers = max(cpu_count - 1, 1)
    else:
        n_workers = 1
    eval_fn = partial(lenia_qd.eval_lenia_config, neg_fitness=False, qdpy=True)
    # with ParallelismManager("none") as pMgr:
    with ParallelismManager("multiprocessing", max_workers=n_workers) as pMgr:
        _ = algo.optimise(
            eval_fn,
            executor=pMgr.executor,
            batch_mode=False  # Calling the optimisation loop per batch if True, else calling it once with total budget
        )

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    save_dir = os.getcwd()
    qdpy_plots.default_plots_grid(logger, output_dir=save_dir)

    if config['other']['dump_bests'] is True:
        lenia_helpers.dump_best(archive, config['run_params']['max_run_iter'])


if __name__ == '__main__':
    run()
