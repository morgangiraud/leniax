import os
import time
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from qdpy.containers import Container
from ribs.archives import ArchiveBase
from typing import Union

from . import utils as lenia_utils
from .api import init_and_run
from .growth_functions import growth_fns
from .core import init


def dump_best(grid: Union[Container, ArchiveBase], fitness_threshold: float, lenia_generator=None):
    if isinstance(grid, Container):
        best_inds = grid._get_best_inds()
        real_bests = list(filter(lambda x: abs(x.fitness.values[0]) >= fitness_threshold, best_inds))
    else:
        real_bests = []
        for i in grid._occupied_indices:
            if abs(grid._objective_values[i]) >= fitness_threshold:
                lenia = next(lenia_generator)
                lenia[:] = grid._solutions[i]
                real_bests.append(lenia)
    print(f"Found {len(real_bests)} beast!")

    for id_best, best in enumerate(real_bests):
        config = best.get_config()
        # print(config)

        render_params = config['render_params']

        start_time = time.time()
        all_cells, all_fields, all_potentials, all_stats = init_and_run(config)
        total_time = time.time() - start_time

        nb_iter_done = len(all_cells)
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

        save_dir = os.path.join(os.getcwd(), str(id_best))  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        print('Dumping cells')
        with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
            np.save(f, np.array(all_cells)[:, 0, 0, ...])

        # print('Plotting stats and functions')
        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

        lenia_utils.plot_stats(save_dir, all_stats)
        _, K, _ = init(config)
        for i in range(K.shape[0]):
            current_k = K[i:i + 1, 0, 0]
            img = lenia_utils.get_image(
                lenia_utils.normalize(current_k, np.min(current_k), np.max(current_k)), 1, 0, colormap
            )
            with open(os.path.join(save_dir, f"kernel{i}.png"), 'wb') as f:
                img.save(f, format='png')
        for i, kernel in enumerate(config['kernels_params']['k']):
            lenia_utils.plot_gfunction(
                save_dir, i, growth_fns[kernel['gf_id']], kernel['m'], kernel['s'], config['world_params']['T']
            )

        width = all_cells[0].shape[-1] * render_params['pixel_size']
        height = all_cells[0].shape[-2] * render_params['pixel_size']
        process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24',
                         s=f"{width}x{height}").output(os.path.join(media_dir, 'beast.mp4'),
                                                       pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
        )
        all_times = []
        for i in range(nb_iter_done):
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
