"""Leniax: Randomly searching for an interesting initial state.

Given a configuration, this examples sample different initial states unitl:
- The budget is exausted
- A local stable pattern has been found

Usage:
    ``python examples/run.py -cn config_name -cp config_path``
"""
import time
import os
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra

import leniax.utils as leniax_utils
import leniax.loader as leniax_loader
import leniax.helpers as leniax_helpers

# Disable JAX logging https://abseil.io/docs/python/guides/logging
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

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    # This is the main call which runs and returns data of the simulation
    # the JAX PRNG key and the configuration parameters are mandatory.
    # In this case:
    #   - We are using the fft optimization
    # This function returns a new rng_key, a dictionnary containing the best run
    # and the number of initial state tested.
    # For more information check the documentaton.
    logging.info("Search: start.")
    start_time = time.time()
    _, (best, nb_init_done) = leniax_helpers.search_for_init(rng_key, config, fft=True)
    # We truncate the best run up to its final interesting state
    all_cells = best['all_cells'][:int(best['N']), 0]
    stats_dict = {k: v.squeeze() for k, v in best['all_stats'].items()}
    logging.info(f"Search: stop. {nb_init_done} initial state tested in {time.time() - start_time:.2f}.")
    logging.info(f"Search: best run length: {best['N']}")

    # We then saved the initial and final states.
    logging.info("Compression: start")
    start_time = time.time()
    config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)
    total_time = time.time() - start_time
    logging.info(f"Compression: stop. Done in {total_time:.2f} seconds.")

    # Finally, we can render our Lenia and other different assets like statistics charts etc.
    # See the documentation for more information.
    logging.info("Assets production: start")
    start_time = time.time()
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)
    total_time = time.time() - start_time
    logging.info(f"Assets production: stop. Done in {total_time:.2f} seconds.")


if __name__ == '__main__':
    search()
