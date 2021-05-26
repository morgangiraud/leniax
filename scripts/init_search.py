import time
import os
import uuid
import logging
import matplotlib.pyplot as plt
import jax
from jax import jit, random
import jax.numpy as jnp

import lenia
from lenia.parser import get_default_parser
from lenia.core import run, init_cells, build_update_fn
from lenia.kernels import get_kernels_and_mapping

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save_init_search')

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()

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
    kernels_params = [{"r": 1, "b": "1", "m": 0.17, "s": 0.015, "h": 1, "k_id": 0, "gf_id": 0, "c_in": 0, "c_out": 0}]

    # Other
    seed = args.seed
    rng = jax.random.PRNGKey(args.seed)

    max_run_iter = args.max_run_iter
    nb_init_search = 2**8
    config = {
        'code': str(uuid.uuid4()),
        'world_params': world_params,
        'render_params': render_params,
        'kernels_params': kernels_params,
        'max_run_iter': max_run_iter,
        'nb_init_search': nb_init_search,
        'seed': seed
    }

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    update_fn = jit(build_update_fn(world_params, K, mapping))

    logging.info("config:", config)

    # For loop for init
    runs = []
    noise_size = jnp.hstack([
        jnp.array([nb_channels] * nb_init_search)[:, jnp.newaxis], random.randint(rng, [nb_init_search, 2], 2**3, 2**6)
    ])
    for i in range(nb_init_search):
        start_time = time.time()

        # Init cells randomly,
        # This is hell slow, we need to generate all the noise before launching the loop
        rng, subkey = random.split(rng)
        noise = random.uniform(subkey, noise_size[i], minval=0, maxval=1.)
        cells_0 = init_cells(world_size, nb_channels, [noise])

        all_cells, all_fields, all_potentials = run(cells_0, update_fn, max_run_iter)
        nb_iter_done = len(all_cells)

        jnp.vstack([cells_0] + all_cells)
        if nb_iter_done > 30:
            runs.append({"N": nb_iter_done, "all_cells": all_cells})

        total_time = time.time() - start_time
        print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    runs.sort(key=lambda run: run["N"], reverse=True)
    best = runs[0]

    all_cells = best['all_cells']
    config['cells'] = all_cells[0]
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
