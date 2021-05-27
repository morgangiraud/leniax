# import time
import os
import uuid
import logging
import matplotlib.pyplot as plt
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

import lenia
from lenia.core import run_init_search

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_init_search")
def search(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    omegaConf.run_params.code = str(uuid.uuid4())

    world_params = omegaConf.world_params
    render_params = omegaConf.render_params
    run_params = omegaConf.run_params

    pixel_size = 2**render_params.pixel_size_power2
    pixel_border_size = render_params.pixel_border_size
    world_size = [2**render_params['size_power2']] * world_params['nb_dims']
    omegaConf.render_params.world_size = world_size
    omegaConf.render_params.pixel_size = pixel_size

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    rng_key = jax.random.PRNGKey(run_params['seed'])
    np.random.seed(run_params['seed'])

    print(OmegaConf.to_yaml(config))

    rng_key, runs = run_init_search(rng_key, config)
    if len(runs) > 0:
        runs.sort(key=lambda run: run["N"], reverse=True)
        best = runs[0]
        all_cells = best['all_cells']

        print(f"best run length: {best['N']}")

        save_dir = os.getcwd()  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia.utils.check_dir(media_dir)

        config['run_params']['cells'] = all_cells[0]
        lenia.utils.save_config(save_dir, config)
        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        for i in range(len(all_cells)):
            lenia.utils.save_image(
                media_dir,
                all_cells[i][:, 0, 0, ...],
                0,
                1,
                pixel_size,
                pixel_border_size,
                colormap,
                i,
                "cell",
            )


if __name__ == '__main__':
    search()
