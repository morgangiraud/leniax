import time
import ffmpeg
import os
import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np

from lenia.api import get_container
from lenia import utils as lenia_utils
from lenia.api import init_and_run

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

    print('Plotting stats')
    lenia_utils.plot_stats(save_dir, all_stats)

    print('Dumping cells')
    with open(os.path.join(save_dir, 'cells.p'), 'wb') as f: 
        np.save(f, np.array(all_cells)[:, 0, 0, ...])

    print('Dumping video')
    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    width = all_cells[0].shape[-1] * render_params['pixel_size']
    height = all_cells[0].shape[-2] * render_params['pixel_size']
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24',
                     s=f"{width}x{height}").output(os.path.join(media_dir, 'beast.mp4'),
                                                   pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    )
    all_times = []
    for i in range(len(all_cells)):
        start_time = time.time()
        img = lenia_utils.get_image(
            all_cells[i][:, 0, 0, ...], render_params['pixel_size'], render_params['pixel_border_size'], colormap
        )
        process.stdin.write(img.tobytes())

        all_times.append(time.time() - start_time)
    process.stdin.close()
    process.wait()
    total_time = np.sum(all_times)
    mean_time = np.mean(all_times)
    print(f"{len(all_times)} images dumped in {total_time} seconds: {mean_time} fps")


if __name__ == '__main__':
    run()
