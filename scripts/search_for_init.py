import time
import os
from absl import logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

from lenia.helpers import search_for_init, get_container
from lenia import utils as lenia_utils
import lenia.video as lenia_video

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_path_1c1kv2 = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')


@hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
# @hydra.main(config_path=config_path_1c1kv2, config_name="wanderer")
def launch(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)
    # config['run_params']['nb_init_search'] = 16
    print(config)

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, best, nb_init_done = search_for_init(rng_key, config)
    print(f"Init search done in {time.time() - t0} (nb_inits done: {nb_init_done})")

    all_cells = best['all_cells'][:int(best['N']), 0]
    stats_dict = {k: v.squeeze() for k, v in best['all_stats'].items()}

    print(f"best run length: {best['N']}")

    save_dir = os.getcwd()  # changed by hydra
    media_dir = os.path.join(save_dir, 'media')
    lenia_utils.check_dir(media_dir)

    first_cells = all_cells[0]
    config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
    lenia_utils.save_config(save_dir, config)

    # print('Dumping cells')
    # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
    #     np.save(f, np.array(all_cells))

    # with open(os.path.join(save_dir, 'last_frame.p'), 'wb') as f:
    #     import pickle
    #     pickle.dump(np.array(all_cells[-1]), f)

    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    print('Plotting stats')
    lenia_utils.plot_stats(save_dir, stats_dict)

    print('Plotting kernels and functions')
    lenia_utils.plot_kernels(config, save_dir)

    print('Dumping video')
    render_params = config['render_params']
    lenia_video.dump_video(all_cells, render_params, media_dir, colormap)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
