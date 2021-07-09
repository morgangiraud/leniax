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

# We are not using matmul on huge matrix, so we can avoid parallelising every operation
# This allow us to increase the numbre of parallel process
# https://github.com/google/jax/issues/743
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1")

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')


@hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
def launch(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)

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
