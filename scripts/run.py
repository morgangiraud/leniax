import time
import os
import json
import matplotlib.pyplot as plt
from jax import jit

import lenia
from lenia.parser import get_default_parser
from lenia.growth_functions import growth_fns

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')

animal_file = os.path.join(cdir, 'orbium.json')

if __name__ == '__main__':
    lenia.utils.check_dir(save_dir)

    parser = get_default_parser()
    args = parser.parse_args()
    nb_dims = args.dims
    nb_channels = args.channels
    pixel_size_power2 = args.P if args.P is not None else args.dims
    pixel_border_size = args.B
    if args.S is not None:
        world_size_power2 = args.S
    else:
        world_size_power2 = [win_power2 - pixel_size_power2 for win_power2 in args.W]

    if len(world_size_power2) < nb_dims:
        world_size_power2 += [world_size_power2[-1]] * (nb_dims - len(world_size_power2))

    world_size = [2**size_power2 for size_power2 in world_size_power2]
    pixel_size = 2**pixel_size_power2

    print("World_size: ", world_size)

    # Load an animal
    if args.animal_filename:
        animal_file = os.path.join(cdir, args.animal_filename)
    with open(animal_file) as f:
        animal_conf = json.load(f)

    world_params = animal_conf['world_params']
    nb_frames = 150
    vmin = 0
    vmax = 1
    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    cells, K, mapping = lenia.init(animal_conf, world_size, nb_channels)

    for idx, gf_id in enumerate(mapping['cin_growth_fns']['gf_id']):
        lenia.utils.plot_function(
            save_dir, idx, growth_fns[gf_id], mapping['cin_growth_fns']['m'][idx], mapping['cin_growth_fns']['s'][idx]
        )

    update_fn = jit(lenia.build_update_fn(world_params, K, mapping))
    # update_fn = lenia.build_update_fn(world_params, K, mapping)

    lenia.utils.save_image(
        save_dir,
        cells[:, 0, 0, ...],
        vmin,
        vmax,
        pixel_size,
        pixel_border_size,
        colormap,
        animal_conf,
        0,
        "cells",
    )
    start_time = time.time()
    for i in range(1, nb_frames + 1):
        cells, field, potential = update_fn(cells)

        lenia.utils.save_image(
            save_dir,
            cells[:, 0, 0, ...],
            vmin,
            vmax,
            pixel_size,
            pixel_border_size,
            colormap,
            animal_conf,
            i,
            "cells",
        )
        lenia.utils.save_image(
            save_dir,
            field[:, 0, 0, ...],
            vmin,
            vmax,
            pixel_size,
            pixel_border_size,
            colormap,
            animal_conf,
            i,
            "field",
        )
        lenia.utils.save_image(
            save_dir,
            potential,
            vmin,
            vmax,
            pixel_size,
            pixel_border_size,
            colormap,
            animal_conf,
            i,
            "potential",
        )

    total_time = time.time() - start_time
    print(f"{nb_frames} frames made in {total_time} seconds: {nb_frames / total_time} fps")
