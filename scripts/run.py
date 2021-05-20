import os
import copy
import argparse
import json
import jax.numpy as jnp

import lenia

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')

orbium_file = os.path.join(cdir, 'orbium.json')

if __name__ == '__main__':

    lenia.utils.check_dir(save_dir)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='''Lenia jax by Morgan Giraud

        recommanded settings: (2D) -d2 -p2, (wide) -d2 -p0 -w 10 9, (3D) -d3 -p3, (4D) -d4 -p4'''
    )
    parser.add_argument(
        '-d', '--dim', dest='D', default=2, action='store', type=int, help='number of dimensions (default 2D)'
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-w',
        '--win',
        dest='W',
        default=[9],
        action='store',
        type=int,
        nargs='+',
        help='window size = 2^W (apply to all sides if only one value, default 2^9 = 512)'
    )
    group.add_argument(
        '-s',
        '--size',
        dest='S',
        default=None,
        action='store',
        type=int,
        nargs='+',
        help='array size = 2^S (apply to all sides if only one value, default 2^(W-P) = 128)'
    )
    parser.add_argument(
        '-p', '--pixel', dest='P', default=None, action='store', type=int, help='pixel size = 2^P (default 2^D)'
    )
    parser.add_argument(
        '-b', '--border', dest='B', default=0, action='store', type=int, help='pixel border (default 0)'
    )
    args = parser.parse_args()

    nb_dims = args.D
    pixel_size_power2 = args.P if args.P is not None else args.D
    pixel_border_size = args.B
    if args.S is not None:
        world_size_power2 = args.S
    else:
        world_size_power2 = [win_power2 - pixel_size_power2 for win_power2 in args.W]

    if len(world_size_power2) < nb_dims:
        world_size_power2 += [world_size_power2[-1]] * (nb_dims - len(world_size_power2))

    world_size = [2**size_power2 for size_power2 in world_size_power2]
    pixel_size = 2**pixel_size_power2

    with open(orbium_file) as f:
        animal_conf = json.load(f)

    vmin = 0
    vmax = 1
    colormap = lenia.utils.create_colormap(
        jnp.asarray([[0, 0, 4], [0, 0, 8], [0, 4, 8], [0, 8, 8], [4, 8, 4], [8, 8, 0], [8, 4, 0], [8, 0, 0], [4, 0, 0]])
    )

    params, cells, gfunc, kernel, kernel_FFT = lenia.init(animal_conf, world_size, nb_dims)

    cells_fft = jnp.asarray(copy.deepcopy(cells))
    for i in range(240):
        cells = lenia.update(params, cells, gfunc, kernel)
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
            is_fft=False,
        )

        cells_fft = lenia.update_fft(params, cells_fft, gfunc, kernel_FFT)
        lenia.utils.save_image(
            save_dir, cells_fft, vmin, vmax, pixel_size, pixel_border_size, colormap, animal_conf, i, is_fft=True
        )
