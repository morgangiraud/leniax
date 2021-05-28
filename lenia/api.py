import os
import logging
import uuid
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from lenia import utils as lenia_utils
from .core import run_init_search


def search_for_init(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    config = get_container(omegaConf)
    print(OmegaConf.to_yaml(config))

    run_params = config['run_params']
    render_params = config['render_params']

    rng_key = lenia_utils.seed_everything(run_params['seed'])

    rng_key, runs = run_init_search(rng_key, config)

    if len(runs) > 0:
        runs.sort(key=lambda run: run["N"], reverse=True)
        best = runs[0]
        all_cells = best['all_cells']

        print(f"best run length: {best['N']}")

        save_dir = os.getcwd()  # changed by hydra
        media_dir = os.path.join(save_dir, 'media')
        lenia_utils.check_dir(media_dir)

        config['run_params']['cells'] = all_cells[0]
        lenia_utils.save_config(save_dir, config)
        colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        for i in range(len(all_cells)):
            lenia_utils.save_image(
                media_dir,
                all_cells[i][:, 0, 0, ...],
                0,
                1,
                render_params['pixel_size'],
                render_params['pixel_border_size'],
                colormap,
                i,
                "cell",
            )


def get_container(omegaConf: DictConfig):

    omegaConf.run_params.code = str(uuid.uuid4())

    world_params = omegaConf.world_params
    render_params = omegaConf.render_params
    pixel_size = 2**render_params.pixel_size_power2
    world_size = [2**render_params['size_power2']] * world_params['nb_dims']
    omegaConf.render_params.world_size = world_size
    omegaConf.render_params.pixel_size = pixel_size

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    return config
