import time
import os
import logging
import psutil
from multiprocessing import get_context
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import copy

import leniax.utils as leniax_utils
import leniax.video as leniax_video
import leniax.helpers as leniax_helpers
import leniax.colormaps as leniax_colormaps

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))


def render(tuple_input):
    res, scale, ori_config = tuple_input
    config = copy.deepcopy(ori_config)
    config["render_params"]["world_size"] = res
    config["render_params"]["pixel_size"] = 1
    config['world_params']['scale'] = scale
    # config['run_params']['max_run_iter'] = 300

    logging.info('Initialiazing and running', config)
    start_time = time.time()
    all_cells, _, _, _ = leniax_helpers.init_and_run(
        config, use_init_cells=False, with_jit=False, fft=True, stat_trunc=stat_trunc
    )  # [nb_max_iter, N=1, C, world_dims...]
    all_cells = all_cells[:, 0]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    logging.info(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.path.join(os.getcwd(), '-'.join([str(a) for a in res]))  # changed by hydra
    leniax_utils.check_dir(save_dir)

    logging.info("Compressing")
    config['run_params']['init_cells'] = leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[0]))
    config['run_params']['cells'] = leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)
    logging.info(f"Compressing done in {total_time} seconds")

    if 'dry_run' in config['other'] and config['other']['dry_run'] is True:
        exit()
    logging.info("Dumping assets")
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
    ]
    leniax_video.dump_video(save_dir, all_cells, config["render_params"], colormaps, 'birth')
    # leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict, colormaps)
    leniax_helpers.dump_frame(save_dir, f'creature_scale{scale}', all_cells[-1], True, colormaps[0])


config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_name = "orbium"

stat_trunc = True
resolutions = [
    [128, 128],  # Optim Lenia world
    [256, 256],
    [512, 512],  # Default Lenia NFT size
    [768, 768],  # Twitter
    [720, 1280],  # HD landscape
    [1280, 720],  # HD
    [1000, 1000],  # Tablette
    [1600, 800],  # Mobile
    [1080, 1920],  # FULLHD landscape
    [1920, 1080],  # FULLHD
]
scales = [
    1,
    2,
    4,
    6,
    6,
    6,
    8,
    8,
    8,
    8,
]


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    ori_config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(ori_config)

    if 'dry_run' in ori_config['other'] and ori_config['other']['dry_run'] is True:
        resolutions = [[128, 128], [256, 256]]
        scales = [1, 2]

    configs = [copy.deepcopy(ori_config) for _ in range(len(scales))]

    nb_cpus = max(1, psutil.cpu_count(logical=False) - 2)
    with get_context("spawn").Pool(processes=nb_cpus) as pool:
        pool.map(render, zip(resolutions, scales, configs))


if __name__ == '__main__':
    run()
