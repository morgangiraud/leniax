import time
import os
from absl import logging
import matplotlib.pyplot as plt
import jax
from omegaconf import DictConfig
import hydra

from lenia.lenia import LeniaIndividual
from lenia.growth_functions import growth_fns
import lenia.utils as lenia_utils
import lenia.core as lenia_core
import lenia.helpers as lenia_helpers
import lenia.video as lenia_video
import lenia.kernels as lenia_kernels
import lenia.qd as lenia_qd

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')


@hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
def launch(omegaConf: DictConfig) -> None:
    config = lenia_helpers.get_container(omegaConf)
    # config['run_params']['nb_init_search'] = 16
    # config['run_params']['max_run_iter'] = 512
    print(config)

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])
    lenia_sols = []
    for _ in range(1):
        rng_key, subkey = jax.random.split(rng_key)
        lenia_sols.append(LeniaIndividual(config, subkey))

    eval_fn = lenia_qd.build_eval_lenia_config_mem_optimized_fn(config)

    t0 = time.time()
    results = eval_fn(lenia_sols)
    print(f"Init search done in {time.time() - t0}")

    for id_best, best in enumerate(results):
        print(f"best run length: {best.fitness}")
        config = best.get_config()

        all_cells, _, _, stats_dict = lenia_helpers.init_and_run(config, True)
        stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}
        all_cells = all_cells[:int(stats_dict['N'])]

        save_dir = os.getcwd()  # changed by hydra
        save_dir = f"{save_dir}/{id_best}"
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        # print('Dumping cells')
        # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
        #     np.save(f, np.array(all_cells)[:, 0, 0, ...])

        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        print('Plotting stats')
        lenia_utils.plot_stats(save_dir, stats_dict)

        print('Plotting kernels and functions')
        _, K, _ = lenia_core.init(config)
        lenia_kernels.draw_kernels(K, save_dir, colormap)
        for i, kernel in enumerate(config['kernels_params']['k']):
            lenia_utils.plot_gfunction(
                save_dir, i, growth_fns[kernel['gf_id']], kernel['m'], kernel['s'], config['world_params']['T']
            )

        print('Dumping video')
        render_params = config['render_params']
        lenia_video.dump_video(all_cells, render_params, media_dir, colormap)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
