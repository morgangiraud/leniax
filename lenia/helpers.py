# import time
import uuid
import jax.numpy as jnp
from typing import Dict, Tuple, List
from omegaconf import DictConfig, OmegaConf

from .lenia import LeniaIndividual
from . import initializations as initializations
from . import statistics as lenia_stat
from . import core as lenia_core
from . import utils as lenia_utils

from . import kernels as lenia_kernels


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
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = lenia_kernels.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    update_fn = lenia_core.build_update_fn(world_params, K.shape, mapping, update_fn_version)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if update_fn_version == 'v1':
        rng_key, noises = initializations.perlin(rng_key, nb_init_search, world_size, R, kernels_params[0])
    elif update_fn_version == 'v2':
        rng_key, noises = initializations.perlin_local(rng_key, nb_init_search, world_size, R, kernels_params[0])
    else:
        raise ValueError('update_fn_version {update_fn_version} does not exist')

    all_cells_0 = []
    for i in range(nb_init_search):
        cells_0 = lenia_core.init_cells(world_size, nb_channels, [noises[i]])
        all_cells_0.append(cells_0)
    all_cells_0_jnp = jnp.array(all_cells_0)

    best_run = {}
    current_max = 0
    for i in range(nb_init_search):
        # t0 = time.time()

        all_cells, _, _, all_stats = lenia_core.run_scan(
            all_cells_0_jnp[i], K, gfn_params, kernels_weight_per_channel, max_run_iter, update_fn, compute_stats_fn
        )
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        all_stats['N'].block_until_ready()

        # t1 = time.time()

        nb_iter_done = all_stats['N']
        if current_max < nb_iter_done:
            current_max = nb_iter_done
            best_run = {"N": nb_iter_done, "all_cells": all_cells, "all_stats": all_stats}

        if nb_iter_done >= max_run_iter:
            break

        # print(t1 - t0, nb_iter_done)

    return rng_key, best_run, i


def init_and_run(config: Dict, with_jit: bool = False) -> Tuple:
    cells, K, mapping = lenia_core.init(config)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    world_params = config['world_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'

    max_run_iter = config['run_params']['max_run_iter']

    update_fn = lenia_core.build_update_fn(config['world_params'], K.shape, mapping, update_fn_version)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if with_jit is True:
        outputs = lenia_core.run_scan(
            cells, K, gfn_params, kernels_weight_per_channel, max_run_iter, update_fn, compute_stats_fn
        )
    else:
        outputs = lenia_core.run(
            cells, K, gfn_params, kernels_weight_per_channel, max_run_iter, update_fn, compute_stats_fn
        )

    return outputs


def get_mem_optimized_inputs(qd_config: Dict, lenia_sols: List[LeniaIndividual]):
    """
        All critical parameters are taken from the qd_config.
        All non-critical parameters are taken from each potential lenia solutions
    """
    world_params = qd_config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'

    render_params = qd_config['render_params']
    world_size = render_params['world_size']

    run_params = qd_config['run_params']
    nb_init_search = run_params['nb_init_search']

    all_cells_0 = []
    all_Ks = []
    all_gfn_params = []
    all_kernels_weight_per_channel = []
    for ind in lenia_sols:
        config = ind.get_config()
        kernels_params = config['kernels_params']['k']

        K, mapping = lenia_core.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
        gfn_params = mapping.get_gfn_params()
        kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

        if update_fn_version == 'v1':
            rng_key, noises = initializations.perlin(ind.rng_key, nb_init_search, world_size, R, kernels_params[0])
        elif update_fn_version == 'v2':
            rng_key, noises = initializations.perlin_local(
                ind.rng_key, nb_init_search, world_size, R, kernels_params[0]
            )
        else:
            raise ValueError('update_fn_version {update_fn_version} does not exist')

        all_cells_0.append(noises)
        all_Ks.append(K)
        all_gfn_params.append(gfn_params)
        all_kernels_weight_per_channel.append(kernels_weight_per_channel)

    all_cells_0_jnp = jnp.stack(all_cells_0)  # add a dimension
    all_Ks_jnp = jnp.stack(all_Ks)
    all_gfn_params_jnp = jnp.stack(all_gfn_params)
    all_kernels_weight_per_channel_jnp = jnp.stack(all_kernels_weight_per_channel)

    run_scan_mem_optimized_parameters = (
        all_cells_0_jnp,
        all_Ks_jnp,
        all_gfn_params_jnp,
        all_kernels_weight_per_channel_jnp,
    )

    return rng_key, run_scan_mem_optimized_parameters


def update_individuals(inds: List[LeniaIndividual],
                       Ns: jnp.ndarray,
                       cells0s: jnp.ndarray,
                       neg_fitness=False) -> List[LeniaIndividual]:
    """
        Args:
            - inds:         List,                       Evaluated Lenia individuals
            - Ns:           jnp.ndarray, [len(inds)]    Fitness of each lenias
            - cells0s:      jnp.ndarray, [len(inds), world_size...]
    """
    for i, ind in enumerate(inds):
        config = ind.get_config()

        current_Ns = Ns[i]
        current_cells0s = cells0s[i]

        max_idx = jnp.argmax(current_Ns, axis=0)

        nb_steps = current_Ns[max_idx]
        cells0 = current_cells0s[max_idx]
        # print(i, current_Ns, max_idx)

        # config['behaviours'] = best['all_stats']
        ind.set_init_cells(lenia_utils.compress_array(cells0))

        if neg_fitness is True:
            fitness = -nb_steps
        else:
            fitness = nb_steps
        ind.fitness = fitness

        if 'phenotype' in ind.qd_config:
            features = [lenia_utils.get_param(config, key_string) for key_string in ind.qd_config['phenotype']]
            ind.features = features

    return inds
