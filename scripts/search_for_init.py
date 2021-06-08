# import time
import os
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt

from lenia.api import search_for_init, get_container
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_init_search")
def launch(omegaConf: DictConfig) -> None:

    config = get_container(omegaConf)
    print(OmegaConf.to_yaml(config))

    config = {
        'world_params': {
            'nb_dims': 2, 'nb_channels': 1, 'R': 13, 'T': 10
        },
        'render_params': {
            'win_size_power2': 9,
            'pixel_size_power2': 2,
            'size_power2': 7,
            'pixel_border_size': 0,
            'world_size': [128, 128],
            'pixel_size': 4
        },
        'kernels_params': {
            'k': [{
                'r': 1,
                'b': '1',
                'm': 0.30243306595743663,
                's': 0.13453207820961527,
                'h': 1,
                'k_id': 0,
                'gf_id': 0,
                'c_in': 0,
                'c_out': 0
            }]
        },
        'run_params': {
            'code': '26f03e07-eb6c-4fcc-87bb-b4389793b63f',
            'seed': 1,
            'max_run_iter': 512,
            'nb_init_search': 512,
            'cells': 'MISSING'
        },
        'params_and_domains': [{
            'key': 'kernels_params.k.0.m', 'domain': [0.0, 0.5], 'type': 'float'
        }, {
            'key': 'kernels_params.k.0.s', 'domain': [0.0, 0.5], 'type': 'float'
        }]
    }

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    _, runs = search_for_init(rng_key, config)

    if len(runs) > 0:
        best = runs[0]
        all_cells = best['all_cells']

        print(f"best run length: {best['N']}")

        save_dir = os.getcwd()  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        first_cells = all_cells[0][:, 0, 0, ...]
        config['run_params']['cells'] = lenia_utils.compress_array(first_cells)
        lenia_utils.save_config(save_dir, config)

        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        for i in range(len(all_cells)):
            lenia_utils.save_images(
                media_dir,
                [all_cells[i][:, 0, 0, ...]],
                0,
                1,
                config['render_params']['pixel_size'],
                config['render_params']['pixel_border_size'],
                colormap,
                i,
                "cell",
            )


if __name__ == '__main__':
    launch()
