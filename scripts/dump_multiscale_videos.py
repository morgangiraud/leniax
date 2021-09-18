import time
import os
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import copy

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.colormaps as leniax_colormaps

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_name = "orbium"

stat_trunc = True
resolutions = [
    [128, 128],  # Default Lenia world
    [512, 512],  # Default Lenia NFT size
    [720, 1280],  # HD
    [1080, 1920],  # FULLHD
]
scales = [
    1,
    4,
    8,
    12,
]


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    ori_config = leniax_helpers.get_container(omegaConf, config_path)

    for res, scale in zip(resolutions, scales):
        config = copy.deepcopy(ori_config)
        config["render_params"]["world_size"] = res
        config["render_params"]["pixel_size"] = 1
        config['world_params']['scale'] = scale

        print('Initialiazing and running', config)
        start_time = time.time()
        all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
            config, with_jit=False, fft=True, use_init_cells=False
        )  # [nb_max_iter, N=1, C, world_dims...]
        if stat_trunc is True:
            all_cells = all_cells[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]
        else:
            all_cells = all_cells[:, 0]  # [nb_iter, C, world_dims...]
        total_time = time.time() - start_time

        nb_iter_done = len(all_cells)
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

        save_dir = os.path.join(os.getcwd(), '-'.join([str(a) for a in res]))  # changed by hydra
        leniax_utils.check_dir(save_dir)

        config['run_params']['init_cells'] = leniax_utils.compress_array(
            leniax_utils.center_and_crop_cells(all_cells[0])
        )
        config['run_params']['cells'] = leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
        leniax_utils.save_config(save_dir, config)

        print("Dumping assets")
        colormaps = [
            leniax_colormaps.colormaps['blackwhite'],
            leniax_colormaps.colormaps['rainbow'],
            leniax_colormaps.colormaps['carmine-blue'],
            leniax_colormaps.colormaps['carmine-green'],
            leniax_colormaps.colormaps['red'],
            leniax_colormaps.colormaps['vistoris-violet'],
            leniax_colormaps.colormaps['vistoris-green'],
            leniax_colormaps.colormaps['cinnamon'],
            leniax_colormaps.colormaps['golden'],
            leniax_colormaps.colormaps['ochraceous'],
            leniax_colormaps.colormaps['lemon-turquoise'],
            leniax_colormaps.colormaps['lemon-venice'],
            leniax_colormaps.colormaps['salvia'],
        ]
        leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict, colormaps)


if __name__ == '__main__':
    run()
