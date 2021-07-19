import time
import uuid
import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from omegaconf import DictConfig, OmegaConf

import lenia.initializations as initializations
import lenia.statistics as lenia_stat
import lenia.core as lenia_core

import lenia.kernels as lenia_kernels


def get_container(omegaConf: DictConfig) -> Dict:
    # TODO: Need to check if missing first
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


def search_for_init(rng_key: jnp.ndarray, config: Dict) -> Tuple[jnp.ndarray, Dict, int]:
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = lenia_kernels.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    gfn_params = mapping.get_gfn_params()
    update_fn = lenia_core.build_update_fn(world_params, K.shape, mapping)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    rng_key, noises = initializations.perlin(rng_key, nb_init_search, world_size, R, kernels_params[0])
    all_cells_0 = []
    for i in range(nb_init_search):
        cells_0 = lenia_core.init_cells(world_size, nb_channels, [noises[i]])
        all_cells_0.append(cells_0)
    all_cells_0_jnp = jnp.array(all_cells_0)

    best_run = {}
    current_max = 0
    for i in range(nb_init_search):
        t0 = time.time()

        all_cells, _, _, all_stats = lenia_core.run_scan(
            all_cells_0_jnp[i], K, gfn_params, max_run_iter, update_fn, compute_stats_fn
        )
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        all_stats['N'].block_until_ready()

        t1 = time.time()

        nb_iter_done = all_stats['N']
        if current_max < nb_iter_done:
            current_max = nb_iter_done
            best_run = {"N": nb_iter_done, "all_cells": all_cells, "all_stats": all_stats}

        if nb_iter_done >= max_run_iter:
            break

        print(t1 - t0, nb_iter_done)

    return rng_key, best_run, i


def init_and_run(config: Dict, with_jit: bool = False) -> Tuple:
    cells, K, mapping = lenia_core.init(config)
    gfn_params = mapping.get_gfn_params()

    update_fn = lenia_core.build_update_fn(config['world_params'], K.shape, mapping)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    max_run_iter = config['run_params']['max_run_iter']
    if with_jit is True:
        outputs = lenia_core.run_scan(cells, K, gfn_params, max_run_iter, update_fn, compute_stats_fn)
    else:
        outputs = lenia_core.run(cells, K, gfn_params, max_run_iter, update_fn, compute_stats_fn)

    return outputs


def slow_init_search(rng_key: jnp.ndarray, config: Dict) -> Tuple[jnp.ndarray, Dict]:
    """
        Slow_init_search is a debugging fuction used to explore in a grid-like fashion some initialization scheme

        So far, it has revealed that different size of random_uniform is not a good way to sample the init space
        but random_uniform on the whole world with different max value is a much better strategy.

    """
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    max_run_iter = run_params['max_run_iter']

    K, mapping = lenia_core.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    gfn_params = mapping.get_gfn_params()

    update_fn = lenia_core.build_update_fn(world_params, K.shape, mapping)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    runs: Dict[str, list] = {}
    min_bound = 128
    max_bound = 128
    max_search_iter = 1
    max_sampling_iter = 32
    for i in range(max_search_iter):
        noise_h = min_bound + i // max_bound
        noise_w = min_bound + i % max_bound
        runs[f"{noise_h},{noise_w}"] = []
        for _ in range(max_sampling_iter):
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.uniform(subkey, (1, noise_h, noise_w), minval=0, maxval=1.)

            cells_0 = lenia_core.init_cells(world_size, nb_channels, [noise])

            all_cells, _, _, all_stats = lenia_core.run(
                cells_0, K, gfn_params, max_run_iter, update_fn, compute_stats_fn
            )
            nb_iter_done = len(all_cells)

            runs[f"{noise_h},{noise_w}"].append(nb_iter_done >= max_run_iter)
            print(f"{noise_h},{noise_w} : {nb_iter_done} / {max_run_iter} -> {nb_iter_done >= max_run_iter}")

    return rng_key, runs
