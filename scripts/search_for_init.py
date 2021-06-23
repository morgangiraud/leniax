import time
import ffmpeg
import numpy as np
import os
from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt

from lenia.api import search_for_init, get_container
# from lenia.api import search_for_init_parallel
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_init_search")
def launch(omegaConf: DictConfig) -> None:

    config = get_container(omegaConf)

    # config = {
    #     'world_params': {
    #         'nb_dims': 2, 'nb_channels': 1, 'R': 13, 'T': 10
    #     },
    #     'render_params': {
    #         'win_size_power2': 9,
    #         'pixel_size_power2': 2,
    #         'size_power2': 7,
    #         'pixel_border_size': 0,
    #         'world_size': [128, 128],
    #         'pixel_size': 4
    #     },
    #     'kernels_params': {
    #         'k': [{
    #             'r': 1, 'b': '1', 'm': 0.15, 's': 0.015, 'h': 1, 'k_id': 0, 'gf_id': 0, 'c_in': 0, 'c_out': 0
    #         }]
    #     },
    #     'run_params': {
    #         'code': '26f03e07-eb6c-4fcc-87bb-b4389793b63f',
    #         'seed': 1,
    #         'max_run_iter': 512,
    #         'nb_init_search': 400,
    #         'cells': 'MISSING'
    #     },
    #     'genotype': [{
    #         'key': 'kernels_params.k.0.m', 'domain': [0.0, 0.5], 'type': 'float'
    #     }, {
    #         'key': 'kernels_params.k.0.s', 'domain': [0.0, 0.5], 'type': 'float'
    #     }]
    # }
    print(config)

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, runs = search_for_init(rng_key, config, with_stats=True)
    # _, runs = search_for_init_parallel(rng_key, config)
    print(f"Init search done in {time.time() - t0}")

    if len(runs) > 0:
        best = runs[0]
        all_cells = best['all_cells']
        all_stats = best['all_stats']
        render_params = config['render_params']

        print(f"best run length: {best['N']}")

        save_dir = os.getcwd()  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        print('Plotting stats')
        lenia_utils.plot_stats(save_dir, all_stats)

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
    launch()
