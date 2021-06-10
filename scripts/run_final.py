import time
import subprocess
import os
import logging
import pickle
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np

from lenia import utils as lenia_utils
from lenia.api import init_and_run

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
pickle_path = os.path.join(cdir, '..', 'outputs', '2021-06-10', '14-15-54', 'final.p')


@hydra.main(config_path=config_path, config_name="dummy")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    best_inds = data['container']._get_best_inds()
    real_bests = list(filter(lambda x: abs(x.fitness.values[0]) >= 512, best_inds))
    print(f"Found {len(real_bests)} beast!")

    for id_best, best in enumerate(real_bests):
        config = best.get_config()
        # print(config)

        render_params = config['render_params']

        start_time = time.time()
        all_cells, all_fields, all_potentials, all_stats = init_and_run(config)
        total_time = time.time() - start_time

        nb_iter_done = len(all_cells) - 1  # all_cells contains cell_0
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

        save_dir = os.path.join(os.getcwd(), str(id_best))  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        frames_dir = os.path.join(media_dir, 'frames')
        lenia_utils.check_dir(frames_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        lenia_utils.plot_stats(save_dir, all_stats)

        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        all_times = []
        for i in range(len(all_cells)):
            start_time = time.time()
            lenia_utils.save_images(
                frames_dir,
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

        cmd = [
            '/usr/local/bin/ffmpeg',
            '-y',
            '-r',
            '13',
            '-i',
            os.path.join(frames_dir, '%05d.png'),
            '-c:v',
            'libx264',
            '-vf',
            'fps=26',
            '-pix_fmt',
            'yuv420p',
            os.path.join(media_dir, 'beast.mp4')
        ]
        subprocess.call(cmd)


if __name__ == '__main__':
    run()
