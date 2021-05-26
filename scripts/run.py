import time
import os
import json
import logging
import uuid
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import lenia
from lenia.parser import get_default_parser
from lenia.core import init_and_run

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()

    # Load a configuration
    if args.config:
        config_fullpath = os.path.join(cdir, args.config)
        with open(config_fullpath) as f:
            config = json.load(f)

        # Render params
        pixel_size_power2 = args.P if args.P is not None else args.nb_dims
        pixel_border_size = args.B
        if args.S is not None:
            world_size_power2 = args.S
        else:
            world_size_power2 = [win_power2 - pixel_size_power2 for win_power2 in args.W]

        if len(world_size_power2) < args.nb_dims:
            world_size_power2 += [world_size_power2[-1]] * (args.nb_dims - len(world_size_power2))

        world_size = [2**size_power2 for size_power2 in world_size_power2]
        pixel_size = 2**pixel_size_power2

        render_params = {
            'world_size': world_size,
            'pixel_size': pixel_size,
            'pixel_border_size': pixel_border_size,
        }

        config['render_params'] = render_params
    else:
        # World params
        nb_dims = args.nb_dims
        nb_channels = args.nb_channels
        R = args.R
        T = args.T

        world_params = {
            'nb_dims': nb_dims,
            'nb_channels': nb_channels,
            'R': R,
            'T': T,
        }

        # Render params
        pixel_size_power2 = args.P if args.P is not None else args.nb_dims
        pixel_border_size = args.B
        if args.S is not None:
            world_size_power2 = args.S
        else:
            world_size_power2 = [win_power2 - pixel_size_power2 for win_power2 in args.W]

        if len(world_size_power2) < args.nb_dims:
            world_size_power2 += [world_size_power2[-1]] * (args.nb_dims - len(world_size_power2))

        world_size = [2**size_power2 for size_power2 in world_size_power2]
        pixel_size = 2**pixel_size_power2

        render_params = {
            'world_size': world_size,
            'pixel_size': pixel_size,
            'pixel_border_size': pixel_border_size,
        }

        # Kernel params
        kernels_params = [{
            "r": 1, "b": "1", "m": 0.15, "s": 0.015, "h": 1, "k_id": 0, "gf_id": 0, "c_in": 0, "c_out": 0
        }]

        # Other
        seed = args.seed
        rng = jax.random.PRNGKey(args.seed)
        init_size = jax.random.randint(rng, [2], 2**3, 2**6)
        init_shape = jnp.concatenate([jnp.array([nb_channels]), init_size])
        init_cells = jax.random.uniform(rng, init_shape, minval=0, maxval=1.)

        config = {
            'code': str(uuid.uuid4()),
            'world_params': world_params,
            'render_params': render_params,
            'kernels_params': kernels_params,
            'max_run_iter': args.max_run_iter,
            'cells': init_cells,
            'seed': seed
        }

    logging.info("config:", config)

    start_time = time.time()
    all_cells, all_fields, all_potentials = init_and_run(config)
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells) - 1  # all_cells contains cell_0
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.path.join(save_dir, config['code'])
    lenia.utils.check_dir(save_dir)
    lenia.utils.save_config(save_dir, config)
    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    for i in range(len(all_cells)):
        lenia.utils.save_image(
            save_dir,
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
            save_dir,
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
            save_dir,
            all_potentials[i],
            0,
            1,
            pixel_size,
            pixel_border_size,
            colormap,
            i,
            "potential",
        )
