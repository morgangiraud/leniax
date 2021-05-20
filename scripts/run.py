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

    DIM = args.D
    DIM_DELIM = {0: '', 1: '$', 2: '%', 3: '#', 4: '@A', 5: '@B', 6: '@C', 7: '@D', 8: '@E', 9: '@F'}
    X_AXIS, Y_AXIS, Z_AXIS = -1, -2, -3

    PIXEL_2 = args.P if args.P is not None else args.D
    PIXEL_BORDER = args.B
    if args.S is not None:
        SIZE_2 = args.S  # X, Y, Z, S...
    else:
        SIZE_2 = [win_2 - PIXEL_2 for win_2 in args.W]
    if len(SIZE_2) < DIM:
        SIZE_2 += [SIZE_2[-1]] * (DIM - len(SIZE_2))
    # GoL 9,9,3,1   Lenia Lo 9,9,2,0  Hi 9,9,0,0   1<<9=512

    SIZE = [1 << size_2 for size_2 in SIZE_2]
    PIXEL = 1 << PIXEL_2
    MID = [int(size / 2) for size in SIZE]
    SIZEX, SIZEY = SIZE[0], SIZE[1]
    MIDX, MIDY = MID[0], MID[1]

    SIZER, SIZETH, SIZEF = min(MIDX, MIDY), SIZEX, MIDX
    DEF_R = int(jnp.power(2.0, min(SIZE_2) - 6) * DIM * 5)
    RAND_R1 = int(jnp.power(2.0, min(SIZE_2) - 7) * DIM * 5)
    RAND_R2 = int(jnp.power(2.0, min(SIZE_2) - 5) * DIM * 5)

    EPSILON = 1e-10
    ROUND = 10

    with open(orbium_file) as f:
        animal_conf = json.load(f)
    vmin = 0
    vmax = 1
    colormap = lenia.utils.create_colormap(
        jnp.asarray([[0, 0, 4], [0, 0, 8], [0, 4, 8], [0, 8, 8], [4, 8, 4], [8, 8, 0], [8, 4, 0], [8, 0, 0], [4, 0, 0]])
    )

    params, cells, gfunc, kernel, kernel_FFT = lenia.init(animal_conf, SIZE, DIM, DIM_DELIM, MID)

    cells_fft = jnp.asarray(copy.deepcopy(cells))
    for i in range(240):
        cells = lenia.update(params, cells, gfunc, kernel)
        lenia.utils.save_image(
            save_dir,
            cells,
            vmin,
            vmax,
            PIXEL,
            PIXEL_BORDER,
            colormap,
            animal_conf,
            i,
            is_fft=False,
        )

        cells_fft = lenia.update_fft(params, cells_fft, gfunc, kernel_FFT)
        lenia.utils.save_image(save_dir, cells, vmin, vmax, PIXEL, PIXEL_BORDER, colormap, animal_conf, i, is_fft=True)
