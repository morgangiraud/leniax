import logging
import uuid
from typing import Dict, Tuple, List, Optional, Callable
from omegaconf import DictConfig, OmegaConf
import jax.numpy as jnp

from .core import run_init_search, run_init_search_parralel, init, run, build_update_fn, build_compute_stats_fn


def search_for_init(rng_key: jnp.ndarray, config: Dict, with_stats: bool = True) -> Tuple[jnp.ndarray, List[Dict]]:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    rng_key, runs = run_init_search(rng_key, config, with_stats)

    if len(runs) > 0:
        runs.sort(key=lambda run: run["N"], reverse=True)

    return rng_key, runs


def search_for_init_parallel(rng_key: jnp.ndarray, config: Dict) -> Tuple[jnp.ndarray, List[Dict]]:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    rng_key, best_all_cells = run_init_search_parralel(rng_key, config)
    runs = [{"all_cells": best_all_cells}]

    # if len(runs) > 0:
    #     runs.sort(key=lambda run: run["N"], reverse=True)

    return rng_key, runs


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


def init_and_run(config: Dict, with_stats: bool = True) -> Tuple:
    cells, K, mapping = init(config)
    update_fn = build_update_fn(config['world_params'], K, mapping)
    compute_stats_fn: Optional[Callable]
    if with_stats:
        compute_stats_fn = build_compute_stats_fn(config['world_params'], config['render_params'])
    else:
        compute_stats_fn = None

    max_run_iter = config['run_params']['max_run_iter']
    outputs = run(cells, max_run_iter, update_fn, compute_stats_fn)

    return outputs
