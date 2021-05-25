import time
import os
import json
import matplotlib.pyplot as plt

import lenia
from lenia.parser import get_default_parser
from lenia.core import init_and_run

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save2')

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
    max_iter = 256

    start_time = time.time()

    all_cells, all_field, all_potential = init_and_run(world_params, world_size, animal_conf, max_iter)

    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    for i, cells in enumerate(all_cells):
        lenia.utils.save_image(
            save_dir,
            cells[:, 0, 0, ...],
            0,
            1,
            pixel_size,
            pixel_border_size,
            colormap,
            animal_conf,
            i,
            "cells",
        )
