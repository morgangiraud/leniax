import time
import os
import copy
import json
import matplotlib.pyplot as plt
from jax import jit
import jax.numpy as jnp

import lenia
from lenia.parser import get_default_parser

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')

orbium_file = os.path.join(cdir, 'orbium.json')

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
    with open(orbium_file) as f:
        animal_conf = json.load(f)

    nb_frames = 150
    vmin = 0
    vmax = 1
    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    kernel_mode = lenia.kernels.KERNEL_MODE_ONE

    params, cells, growth_fn, kernel, kernels_fft = lenia.init(animal_conf, world_size, nb_channels, kernel_mode)
    cells_fft = jnp.asarray(copy.deepcopy(cells))

    update_fn = jit(lenia.build_update_fn(params, growth_fn, kernel, lenia.kernels.KERNEL_MODE_ONE))
    # fft_update_fn = jit(lenia.build_update_fn(params, growth_fn, kernels_fft, lenia.kernels.KERNEL_MODE_ONE_FFT))

    start_time = time.time()
    for i in range(nb_frames):
        cells, field, potential = update_fn(cells)
        lenia.utils.save_image(
            save_dir,
            cells,
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
            field,
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

        # cells_fft, field_fft, potential_fft = fft_update_fn(cells_fft)
        # lenia.utils.save_image(
        #     save_dir, cells_fft, vmin, vmax, pixel_size, pixel_border_size, colormap, animal_conf, i, "fft"
        # )

    total_time = time.time() - start_time
    print(f"{nb_frames} frames made in {total_time} seconds: {nb_frames / total_time} fps")
