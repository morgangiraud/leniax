import time
import os
import logging
from absl import logging as absl_logging
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np

from lenia.helpers import get_container, init_and_run
import lenia.utils as lenia_utils
from lenia.growth_functions import growth_fns
from lenia.core import init
import lenia.video as lenia_video
import lenia.kernels as lenia_kernels

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_path_1c1kv2 = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')
config_path_1c3k = os.path.join(cdir, '..', 'conf', 'species', '1c-3k')
config_path_3c15k = os.path.join(cdir, '..', 'conf', 'species', '3c-15k')
config_path_outputs = os.path.join(cdir, '..', 'outputs', '2021-08-09', '09-39-59')


# @hydra.main(config_path=config_path_1c1k, config_name="orbium")
# @hydra.main(config_path=config_path_1c3k, config_name="fish")
@hydra.main(config_path=config_path_3c15k, config_name="aquarium")
# @hydra.main(config_path=config_path_1c1kv2, config_name="wanderer")
# @hydra.main(config_path=config_path, config_name="orbium-scutium")
# @hydra.main(config_path=config_path_outputs, config_name="config")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    config = get_container(omegaConf)

    print('Initialiazing and running')
    start_time = time.time()
    all_cells, _, _, stats_dict = init_and_run(config, True)
    stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}
    all_cells = all_cells[:int(stats_dict['N'])]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.getcwd()  # changed by hydra
    media_dir = os.path.join(save_dir, 'media')
    lenia_utils.check_dir(media_dir)

    first_cells = all_cells[0, 0, ...]
    config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
    lenia_utils.save_config(save_dir, config)

    print('Dumping cells')
    with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
        np.save(f, np.array(all_cells)[:, 0, ...])

    # with open(os.path.join(save_dir, 'last_frame.p'), 'wb') as f:
    #     import pickle
    #     pickle.dump(np.array(all_cells)[127, 0, ...], f)

    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    print('Plotting stats')
    lenia_utils.plot_stats(save_dir, stats_dict)

    print('Plotting kernels and functions')
    _, K, _ = init(config)
    lenia_kernels.draw_kernels(K, save_dir, colormap)
    for i, kernel in enumerate(config['kernels_params']['k']):
        lenia_utils.plot_gfunction(
            save_dir, i, growth_fns[kernel['gf_id']], kernel['m'], kernel['s'], kernel['h'], config['world_params']['T']
        )

    print('Dumping video')
    render_params = config['render_params']
    lenia_video.dump_video(all_cells, render_params, media_dir, colormap)


if __name__ == '__main__':
    run()
