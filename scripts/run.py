import time
import os
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
# from hydra.core.config_store import ConfigStore

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
# from leniax.structured_config import LeniaxConfig

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_name = "orbium"
# config_name = "vibratium"

# config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')
# config_name = "wanderer"

# config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-2k')
# config_name = "squiggle"

# config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-3k')
# config_name = "fish"

# config_path = os.path.join(cdir, '..', 'conf', 'species', '3c-15k')

config_path = os.path.join(cdir, '..', 'outputs', 'collection-01', '05-maelstrom', '0189')
# config_path = os.path.join(cdir, '..', 'outputs', 'tmp')
config_name = "config"

# config_path = os.path.join(cdir, '..', 'tests', 'fixtures')
# config_name = "aquarium-test.yaml"

stat_trunc = True
to_hd = False


# cs = ConfigStore.instance()
# cs.store(name=config_name, node=LeniaxConfig)
@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_helpers.get_container(omegaConf, config_path)

    config['render_params']['pixel_size_power2'] = 0
    config['render_params']['pixel_size'] = 1
    config['render_params']['size_power2'] = 7
    config['render_params']['world_size'] = [128, 128]
    config['world_params']['scale'] = 1.
    config['run_params']['max_run_iter'] = 1024

    if to_hd is True:
        config = leniax_utils.update_config_to_hd(config)
        use_init_cells = False
    else:
        use_init_cells = True

    leniax_utils.print_config(config)

    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
        config, with_jit=True, fft=True, use_init_cells=use_init_cells
    )  # [nb_max_iter, N=1, C, world_dims...]
    if stat_trunc is True:
        all_cells = all_cells[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]
    else:
        all_cells = all_cells[:, 0]  # [nb_iter, C, world_dims...]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    config['run_params']['init_cells'] = leniax_utils.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)

    print("Dumping assets")
    # import matplotlib.pyplot as plt
    # colormaps = [plt.get_cmap(name) for name in ['plasma', 'turbo', 'viridis']]
    # leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict, colormaps)
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    run()
