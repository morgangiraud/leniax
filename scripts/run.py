import time
import os
import logging
import uuid
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra

import lenia
from lenia.core import init_and_run

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species')


@hydra.main(config_path=config_path, config_name="orbium-scutium")
def run(omegaConf: DictConfig) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    omegaConf.run_params.code = str(uuid.uuid4())

    world_params = omegaConf.world_params
    render_params = omegaConf.render_params
    pixel_size = 2**render_params.pixel_size_power2
    pixel_border_size = render_params.pixel_border_size
    world_size = [2**render_params['size_power2']] * world_params['nb_dims']
    omegaConf.render_params.world_size = world_size
    omegaConf.render_params.pixel_size = pixel_size

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    start_time = time.time()
    all_cells, all_fields, all_potentials = init_and_run(config)
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells) - 1  # all_cells contains cell_0
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.getcwd()  # changed by hydra
    media_dir = os.path.join(save_dir, 'media')
    lenia.utils.check_dir(media_dir)

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
        lenia.utils.save_image(
            media_dir,
            all_fields[i][:, 0, 0, ...],
            0,
            1,
            pixel_size,
            pixel_border_size,
            colormap,
            i,
            "field",
        )
        lenia.utils.save_image(
            media_dir,
            all_potentials[i],
            0,
            1,
            pixel_size,
            pixel_border_size,
            colormap,
            i,
            "potential",
        )


if __name__ == '__main__':
    run()
