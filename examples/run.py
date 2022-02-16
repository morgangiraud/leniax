"""Leniax: A simple simulation example.

This is a simple example on how to use Leniax to render a Lenia simulation.

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
import leniax.helpers as leniax_helpers
import leniax.loader as leniax_loader
from leniax import colormaps as leniax_colormaps

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
config_name = "orbium"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)
    leniax_utils.print_config(config)

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    # This is the main call which runs and returns data of the simulation
    # Ony the configuration parameter is mandatory.
    # In this case:
    #   - We are using the field "init_cells" as a the simulation initialization
    #   - We are not using JAX jit functions
    #   - We are using the fft optimization
    #   - We will truncate the computed statistics directory up to final interesting state (more info on this in the documentation.)
    # This function returns all the different states, potentials and fields + the statistic dictionnary
    # All the arrays are of shape [nb_max_iter, N, C, world_dims...]
    logging.info("Simulation: start.")
    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
        rng_key,
        config,
        use_init_cells=False,
        with_jit=False,
        fft=True,
        stat_trunc=True,
    )
    # In our case, we only ran 1 simulation so N=1
    all_cells = all_cells[:, 0]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells)
    logging.info(
        f"Simulation: stop. {nb_iter_done} states computed in {total_time:.2f} seconds, {nb_iter_done / total_time:.2f} fps."
    )

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
    colormaps = [leniax_colormaps.get(cmap_name) for cmap_name in config['render_params']['colormaps']]
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict, colormaps)
    for colormap in colormaps:
        leniax_helpers.dump_frame(save_dir, f'last_frame_cropped_{colormap.name}', all_cells[-1], True, colormap)
        leniax_helpers.dump_frame(save_dir, f'last_frame_{colormap.name}', all_cells[-1], False, colormap)
    total_time = time.time() - start_time
    logging.info(f"Assets production: stop. Done in {total_time:.2f} seconds.")


if __name__ == '__main__':
    run()
