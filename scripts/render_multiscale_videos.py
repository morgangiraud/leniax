import time
import os
import logging
from absl import logging as absl_logging
import psutil
from multiprocessing import get_context
from omegaconf import DictConfig
import hydra
import copy

import leniax.utils as leniax_utils
import leniax.video as leniax_video
import leniax.helpers as leniax_helpers
import leniax.colormaps as leniax_colormaps
import leniax.loader as leniax_loader

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))


def render(tuple_input):
    res, scale, ori_config = tuple_input
    leniax_utils.set_log_level(ori_config)

    config = copy.deepcopy(ori_config)
    config["render_params"]["world_size"] = res
    config["render_params"]["pixel_size"] = 1
    config['world_params']['scale'] = scale
    # config['run_params']['max_run_iter'] = 300

    leniax_utils.print_config(config)

    # Hydra change automatically the working directory for each run.
    save_dir = os.path.join(os.getcwd(), '-'.join([str(a) for a in res]))
    leniax_utils.check_dir(save_dir)

    logging.info("Simulation: start.")
    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
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
    leniax_video.render_video(save_dir, all_cells, config["render_params"], colormaps, 'birth')
    for colormap in colormaps:
        leniax_helpers.dump_frame(save_dir, f'last_frame_cropped_{colormap.name}', all_cells[-1], True, colormap)
        leniax_helpers.dump_frame(save_dir, f'last_frame_{colormap.name}', all_cells[-1], False, colormap)
    total_time = time.time() - start_time
    logging.info(f"Assets production: stop. Done in {total_time:.2f} seconds.")


config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_name = "orbium"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    ori_config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(ori_config)

    if 'dry_run' in ori_config['other'] and ori_config['other']['dry_run'] is True:
        resolutions = [[64, 64], [128, 128]]
        scales = [1, 2]
    else:
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

    configs = [copy.deepcopy(ori_config) for _ in range(len(scales))]

    nb_cpus = max(1, psutil.cpu_count(logical=False) - 2)
    with get_context("spawn").Pool(processes=nb_cpus) as pool:
        pool.map(render, zip(resolutions, scales, configs))


if __name__ == '__main__':
    run()
