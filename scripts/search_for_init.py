import time
import os
from absl import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

from lenia.helpers import search_for_init, get_container
from lenia import utils as lenia_utils
from lenia.growth_functions import growth_fns
from lenia.core import init
import lenia.video as lenia_video
import lenia.kernels as lenia_kernels

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')


@hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
def launch(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)
    config['run_params']['nb_init_search'] = 256
    config['run_params']['max_run_iter'] = 128
    print(config)

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, runs = search_for_init(rng_key, config)
    print(f"Init search done in {time.time() - t0} (nb_inits done: {len(runs) })")

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


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
