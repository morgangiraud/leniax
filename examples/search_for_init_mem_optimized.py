"""Leniax: Randomly searching for an interesting initial state.

Given a configuration, this examples sample different initial states unitl:
- The budget is exausted
- A local stable pattern has been found

We are using a memory optimized version of the basic ``search_init`` function.
In this version, we do not store temporary states, we only store statistics.

Usage:
    ``python examples/run.py -cn config_name -cp config_path``
"""
import time
import os
import logging
from absl import logging as absl_logging
import jax
from omegaconf import DictConfig
import hydra

from leniax.lenia import LeniaIndividual
import leniax.utils as leniax_utils
import leniax.loader as leniax_loader
import leniax.helpers as leniax_helpers
import leniax.qd as leniax_qd

# This is used to remove JAX logging
absl_logging.set_verbosity(absl_logging.ERROR)

###
# We use hydra to load Leniax configurations.
# It alloes to do many things among which, we can override configuraiton parameters.
# For example to render a Lenia in a bigger size:
# python examples/run.py render_params.world_size='[512, 512]' world_params.scale=4
###
cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "config_init_search"


@hydra.main(config_path=config_path, config_name=config_name)
def search(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)
    leniax_utils.print_config(config)

    parent_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(parent_dir)
    logging.info(f"Output directory: {parent_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    # The memory optimized function is very generic and can run multiple simulations
    # with multiple initial states in parallel.
    # In this examples we are only looking for 1 configuration with multiple initial state
    nb_confs = 1
    rng_key, *subkeys = jax.random.split(rng_key, 1 + nb_confs)
    leniax_confs = [LeniaIndividual(config, subkey, []) for subkey in subkeys]

    # This is the main calls which runs and returns data of the simulation
    # Only one parameter needed: the configuration
    # This function returns a new a list of statics for each intial state per configurations.
    # For more information check the documentaton.
    logging.info("Search: start.")
    start_time = time.time()
    # First, the evaluation function is built dynamically
    eval_fn = leniax_qd.build_eval_lenia_config_mem_optimized_fn(config)
    # Then we evaluate all of our configurations in parallel
    results = eval_fn(leniax_confs)
    logging.info(f"Search: stop. {len(results)} configurations tested in {time.time() - start_time:.2f}.")

    # We iterate over the configurations and render the best initial state.
    for id_best, best in enumerate(results):
        logging.info(f"Search: best run length: {best.fitness}")
        config = best.get_config()
        save_dir = os.path.join(parent_dir, f"{str(id_best).zfill(4)}")  # changed by hydra
        leniax_utils.check_dir(save_dir)

        # Since we do not store intermediary states, we need to re-simulate the best configurations
        logging.info("Simulation: start.")
        start_time = time.time()
        all_cells, _, _, stats_dict = leniax_helpers.init_and_run(config, with_jit=True, stat_trunc=True)
        all_cells = all_cells[:, 0]
        total_time = time.time() - start_time
        nb_iter_done = len(all_cells)
        logging.info(
            f"Simulation: stop. {nb_iter_done} states computed in {total_time:.2f} seconds, {nb_iter_done / total_time:.2f} fps."
        )

        logging.info("Compression: start")
        start_time = time.time()
        config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
        config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop(all_cells[-1]))
        leniax_utils.save_config(save_dir, config)
        total_time = time.time() - start_time
        logging.info(f"Compression: stop. Done in {total_time:.2f} seconds.")

        logging.info("Assets production: start")
        start_time = time.time()
        leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)
        total_time = time.time() - start_time
        logging.info(f"Assets production: stop. Done in {total_time:.2f} seconds.")


if __name__ == '__main__':
    search()
