import time
import os
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
# from hydra.core.config_store import ConfigStore

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.loader as leniax_loader
from leniax import colormaps as leniax_colormaps
# from leniax.structured_config import LeniaxConfig

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "orbium"
# config_name = "wanderer"
# config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '2c-4k')
# config_name = "orbium-0"
# config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '3c-9k')
# config_name = "orbium-0-0"
# config_path = os.path.join(cdir, '..', 'conf', 'species', '3d', '1c-1k')
# config_name = "prototype"

# config_path = os.path.join(cdir, '..', 'outputs', 'tmp')
# config_name = "config"


# cs = ConfigStore.instance()
# cs.store(name=config_name, node=LeniaxConfig)
@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)

    # config['render_params']['pixel_size_power2'] = 0
    config['render_params']['pixel_size'] = 1
    # config['render_params']['size_power2'] = 7
    # config['render_params']['world_size'] = [256, 256]
    # config['world_params']['scale'] = 2.
    config['run_params']['max_run_iter'] = 512

    leniax_utils.print_config(config)

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    print("Rendering: start")
    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
        config,
        use_init_cells=False,
        with_jit=False,
        fft=True,
        stat_trunc=True,
        # override=override
    )  # [nb_max_iter, N=1, C, world_dims...]
    all_cells = all_cells[:, 0]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells)
    print(f"Rendering: {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    print("Compressing")
    start_time = time.time()
    config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)
    total_time = time.time() - start_time
    print(f"Compressing in {total_time} seconds")

    print("Dumping assets")
    transparent_bg = False
    colormaps = [
        # leniax_colormaps.get('alizarin'),
        # leniax_colormaps.get('black-white'),
        # leniax_colormaps.get('carmine-blue'),
        # leniax_colormaps.get('cinnamon'),
        # leniax_colormaps.get('city'),
        # leniax_colormaps.get('golden'),
        # leniax_colormaps.get('laurel'),
        leniax_colormaps.get('msdos'),
        # leniax_colormaps.get('pink-beach'),
        # leniax_colormaps.get('rainbow'),
        # leniax_colormaps.get('rainbow_transparent'),
        # leniax_colormaps.get('river-Leaf'),
        # leniax_colormaps.get('salvia'),
        # leniax_colormaps.get('summer'),
        # leniax_colormaps.get('white-black'),
        # leniax_colormaps.Hilbert2d3dColormap('hilbert-2d3d'),
        # leniax_colormaps.ExtendedColormap('extended', transparent_bg=transparent_bg),
    ]
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict, colormaps, transparent_bg)
    for colormap in colormaps:
        leniax_helpers.dump_frame(save_dir, f'last_frame_cropped_{colormap.name}', all_cells[-1], False, colormap)
        leniax_helpers.dump_frame(save_dir, f'last_frame_{colormap.name}', all_cells[-1], True, colormap)


if __name__ == '__main__':
    run()
