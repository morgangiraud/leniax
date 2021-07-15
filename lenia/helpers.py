import os
import time
import numpy as np
import matplotlib.pyplot as plt
from qdpy.containers import Container
from ribs.archives import ArchiveBase
from typing import Union

from . import utils as lenia_utils
from .api import init_and_run, search_for_init
from .growth_functions import growth_fns
from .core import init
import lenia.video as lenia_video
import lenia.kernels as lenia_kernels


def dump_best(grid: Union[Container, ArchiveBase], fitness_threshold: float, lenia_generator=None):
    is_ribs_archive = False
    if isinstance(grid, Container):
        best_inds = grid._get_best_inds()
        real_bests = list(filter(lambda x: abs(x.fitness.values[0]) >= fitness_threshold, best_inds))
    else:
        is_ribs_archive = True
        real_bests = []
        i = 0
        for idx in grid._occupied_indices:
            if abs(grid._objective_values[idx]) >= fitness_threshold:
                lenia = next(lenia_generator)
                lenia[:] = grid._solutions[idx]
                # print(i, idx, grid._solutions[idx], lenia.get_config()["kernels_params"]["k"][0]["m"], lenia.get_config()["kernels_params"]["k"][0]["s"])
                real_bests.append(lenia)
                i += 1
    
    print(f"Found {len(real_bests)} beast!")

    for id_best, best in enumerate(real_bests):
        config = best.get_config()
        print(config)

        render_params = config['render_params']

        start_time = time.time()
        if is_ribs_archive is True:
            _, runs = search_for_init(best.rng_key, config, with_stats=True)
            best = runs[0]
            all_cells = best['all_cells']
            all_stats = best['all_stats']
        else:
            all_cells, _, _, all_stats = init_and_run(config)
        total_time = time.time() - start_time

        nb_iter_done = len(all_cells)
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

        save_dir = os.path.join(os.getcwd(), str(id_best))  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        # print('Dumping cells')
        # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
        #     np.save(f, np.array(all_cells)[:, 0, 0, ...])

        print('Plotting stats and functions')
        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

        lenia_utils.plot_stats(save_dir, all_stats)
        _, K, _ = init(config)
        lenia_kernels.draw_kernels(K, save_dir, colormap)
        for i, kernel in enumerate(config['kernels_params']['k']):
            lenia_utils.plot_gfunction(
                save_dir, i, growth_fns[kernel['gf_id']], kernel['m'], kernel['s'], config['world_params']['T']
            )

        print('Dumping video')
        lenia_video.dump_video(all_cells, render_params, media_dir, colormap)
