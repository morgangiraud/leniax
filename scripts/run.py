import time
import os
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np

from lenia.api import get_container
from lenia import utils as lenia_utils
from lenia.core import init_and_run

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species')


@hydra.main(config_path=config_path, config_name="orbium")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    config = get_container(omegaConf)

    render_params = config['render_params']

    start_time = time.time()
    all_cells, all_fields, all_potentials, all_stats = init_and_run(config)
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells) - 1  # all_cells contains cell_0
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.getcwd()  # changed by hydra
    media_dir = os.path.join(save_dir, 'media')
    lenia_utils.check_dir(media_dir)

    first_cells = all_cells[0][:, 0, 0, ...]
    config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
    lenia_utils.save_config(save_dir, config)

    lenia_utils.plot_stats(save_dir, all_stats)

    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    all_times = []
    for i in range(len(all_cells)):
        start_time = time.time()
        lenia_utils.save_images(
            media_dir,
            [all_cells[i][:, 0, 0, ...], all_fields[i][:, 0, 0, ...], all_potentials[i]],
            0,
            1,
            render_params['pixel_size'],
            render_params['pixel_border_size'],
            colormap,
            i,
            "",
        )
        all_times.append(time.time() - start_time)
    total_time = np.sum(all_times)
    mean_time = np.mean(all_times)
    print(f"{len(all_times)} images dumped in {total_time} seconds: {mean_time} fps")


if __name__ == '__main__':
    run()
