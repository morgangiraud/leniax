import time
import os
import json
import pickle
import uuid
import numpy as np
import jax.numpy as jnp
from shutil import copy2
from typing import Dict, Tuple, List
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from absl import logging as absl_logging

from .constant import EPSILON
from .lenia import LeniaIndividual
from . import initializations as initializations
from . import statistics as leniax_stat
from . import core as leniax_core
from . import utils as leniax_utils
from . import kernels as leniax_kernels
from . import growth_functions as leniax_gf
from . import video as leniax_video


def get_container(omegaConf: DictConfig) -> Dict:
    main_path = ''
    # We try to get the Hydra main config if we can
    try:
        for source in HydraConfig.get().runtime.config_sources:
            if source.provider == 'main':
                main_path = source.path
    except ValueError:
        pass

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
    config['main_path'] = main_path

    return config


def search_for_init(rng_key: jnp.ndarray, config: Dict, fft: bool = True) -> Tuple[jnp.ndarray, Dict, int]:
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    R = world_params['R']
    T = jnp.array(world_params['T'])

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = leniax_kernels.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    update_fn = leniax_core.build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if update_fn_version == 'v1':
        rng_key, noises = initializations.perlin(rng_key, nb_init_search, world_size, R, kernels_params[0])
    elif update_fn_version == 'v2':
        rng_key, noises = initializations.perlin_local(rng_key, nb_init_search, world_size, R, kernels_params[0])
    else:
        raise ValueError('update_fn_version {update_fn_version} does not exist')

    all_cells_0 = []
    for i in range(nb_init_search):
        cells_0 = leniax_core.init_cells(world_size, nb_channels, [noises[i]])
        all_cells_0.append(cells_0)
    all_cells_0_jnp = jnp.array(all_cells_0)

    best_run = {}
    current_max = 0
    for i in range(nb_init_search):
        # t0 = time.time()

        all_cells, _, _, all_stats = leniax_core.run_scan(
            all_cells_0_jnp[i],
            K,
            gfn_params,
            kernels_weight_per_channel,
            T,
            max_run_iter,
            R,
            update_fn,
            compute_stats_fn,
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


def init_and_run(config: Dict, scale: float = 1., with_jit: bool = False, fft: bool = True) -> Tuple:
    cells, K, mapping = leniax_core.init(config, scale=scale, fft=fft)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

    world_params = config['world_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    R = config['world_params']['R']
    T = jnp.array(config['world_params']['T'])

    max_run_iter = config['run_params']['max_run_iter']

    update_fn = leniax_core.build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if with_jit is True:
        all_cells, all_fields, all_potentials, stats_dict = leniax_core.run_scan(
            cells, K, gfn_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn
        )
    else:
        all_cells, all_fields, all_potentials, stats_dict = leniax_core.run(
            cells, K, gfn_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn
        )
    stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}

    return all_cells, all_fields, all_potentials, stats_dict


def get_mem_optimized_inputs(qd_config: Dict, lenia_sols: List[LeniaIndividual], fft: bool = True):
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
    all_Ts = []
    for ind in lenia_sols:
        config = ind.get_config()
        kernels_params = config['kernels_params']['k']

        K, mapping = leniax_core.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)
        gfn_params = mapping.get_gfn_params()
        kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

        nb_channels_to_init = nb_channels * nb_init_search
        if update_fn_version == 'v1':
            rng_key, noises = initializations.perlin(ind.rng_key, nb_channels_to_init, world_size, R, kernels_params[0])
        elif update_fn_version == 'v2':
            rng_key, noises = initializations.perlin_local(
                ind.rng_key, nb_channels_to_init, world_size, R, kernels_params[0]
            )
        else:
            raise ValueError('update_fn_version {update_fn_version} does not exist')

        all_cells_0.append(noises.reshape([nb_init_search, nb_channels] + world_size))
        all_Ks.append(K)
        all_gfn_params.append(gfn_params)
        all_kernels_weight_per_channel.append(kernels_weight_per_channel)
        all_Ts.append(config['world_params']['T'])

    all_cells_0_jnp = jnp.stack(all_cells_0)  # add a dimension
    all_Ks_jnp = jnp.stack(all_Ks)
    all_gfn_params_jnp = jnp.stack(all_gfn_params)
    all_kernels_weight_per_channel_jnp = jnp.stack(all_kernels_weight_per_channel)
    all_Ts_jnp = jnp.stack(all_Ts)

    run_scan_mem_optimized_parameters = (
        all_cells_0_jnp,
        all_Ks_jnp,
        all_gfn_params_jnp,
        all_kernels_weight_per_channel_jnp,
        all_Ts_jnp,
    )

    return rng_key, run_scan_mem_optimized_parameters


def update_individuals(
    inds: List[LeniaIndividual],
    stats: Dict[str, jnp.ndarray],
    cells0s: jnp.ndarray,
    neg_fitness=False
) -> List[LeniaIndividual]:
    """
        Args:
            - inds:         List,                       Evaluated Lenia individuals
            - Ns:           jnp.ndarray, [len(inds)]    Fitness of each lenias
            - cells0s:      jnp.ndarray, [len(inds), world_size...]
    """
    for i, ind in enumerate(inds):
        config = ind.get_config()

        current_Ns = stats['N'][i]
        current_cells0s = cells0s[i]

        max_idx = jnp.argmax(current_Ns, axis=0)

        nb_steps = current_Ns[max_idx]
        cells0 = current_cells0s[max_idx]
        # print(i, current_Ns, max_idx)

        config['behaviours'] = {}
        for k in stats.keys():
            if k == 'N':
                continue
            truncated_stat = stats[k][i, :int(nb_steps), max_idx]
            config['behaviours'][k] = truncated_stat[-128:].mean()
        ind.set_init_cells(leniax_utils.compress_array(cells0))

        if neg_fitness is True:
            fitness = -nb_steps
        else:
            fitness = nb_steps
        ind.fitness = fitness

        if 'phenotype' in ind.qd_config:
            features = [leniax_utils.get_param(config, key_string) for key_string in ind.qd_config['phenotype']]
            ind.features = features

    return inds


def process_lenia(enum_lenia: Tuple[int, LeniaIndividual]):
    # This function is usually called in forked processes, before launching any JAX code
    # We silent it
    # Disable JAX logging https://abseil.io/docs/python/guides/logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    id_lenia, lenia = enum_lenia
    padded_id = str(id_lenia).zfill(4)
    config = lenia.get_config()

    save_dir = os.path.join(os.getcwd(), f"c-{padded_id}")  # changed by hydra
    if os.path.isdir(save_dir):
        # If the lenia folder already exists at that path, we consider the work alreayd done
        return
    leniax_utils.check_dir(save_dir)

    start_time = time.time()
    all_cells, _, _, stats_dict = init_and_run(config, with_jit=True)
    all_cells = all_cells[:int(stats_dict['N']), 0]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"[{padded_id}] - {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    first_cells = all_cells[0]
    config['run_params']['cells'] = leniax_utils.compress_array(first_cells)
    leniax_utils.save_config(save_dir, config)

    print(f"[{padded_id}] - Dumping assets")
    dump_assets(save_dir, config, all_cells, stats_dict)


def dump_last_frame(save_dir: str, all_cells: jnp.ndarray, raw: bool = False):
    """
        Args:
            - save_dir: str
            - all_cells: jnp.ndarray[nb_iter, C, world_dims...]
    """
    last_frame = all_cells[-1]

    if raw is False:
        world_size = last_frame.shape[1:]
        axes = tuple(range(-len(world_size), 0, 1))

        midpoint = jnp.asarray([size // 2 for size in world_size])[:, jnp.newaxis, jnp.newaxis]
        coords = jnp.indices(world_size)
        centered_coords = coords - midpoint

        m_00 = last_frame.sum()
        MX = [(last_frame * coord).sum() for coord in centered_coords]
        mass_centroid = jnp.array(MX) / (m_00 + EPSILON)

        shift_idx = mass_centroid.astype(int).T
        centered_last_frame, _, _ = leniax_utils.center_world(
            last_frame[jnp.newaxis],
            last_frame[jnp.newaxis],
            last_frame[jnp.newaxis],
            shift_idx[jnp.newaxis],
            axes,
        )

        last_frame = leniax_utils.crop_zero(centered_last_frame[0])

    with open(os.path.join(save_dir, 'last_frame.p'), 'wb') as f:
        pickle.dump(np.array(last_frame), f)

    colormap = plt.get_cmap('plasma')
    img = leniax_utils.get_image(last_frame, 1, 0, colormap)
    with open(os.path.join(save_dir, "last_frame.png"), 'wb') as f:
        img.save(f, format='png')


###
# Viz
###
def dump_assets(save_dir: str, config: Dict, all_cells: jnp.ndarray, stats_dict: Dict):
    colormap = plt.get_cmap('plasma')  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    leniax_utils.plot_stats(save_dir, stats_dict)

    plot_kernels(save_dir, config)

    with open(os.path.join(save_dir, 'stats_dict.p'), 'wb') as f:
        pickle.dump(stats_dict, f)

    # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
    #     np.save(f, np.array(all_cells))

    dump_last_frame(save_dir, all_cells)

    viz_data: Dict = {'stats': {}}
    for k, v in stats_dict.items():
        if k == 'N':
            continue
        viz_data['stats'][k + '_mean'] = round(float(v[-128:].mean()), 5)
        viz_data['stats'][k + '_std'] = round(float(v[-128:].std()), 5)
    viz_data['img_url'] = 'last_frame.png'
    viz_data['k'] = config['kernels_params']['k']
    viz_data['R'] = config['world_params']['R']
    viz_data['T'] = config['world_params']['T']
    with open(os.path.join(save_dir, 'viz_data.json'), 'w') as fviz:
        json.dump(viz_data, fviz)

    render_params = config['render_params']
    leniax_video.dump_video(save_dir, all_cells, render_params, colormap)


def plot_kernels(save_dir, config):
    world_size = config['render_params']['world_size']
    R = config['world_params']['R']

    x = jnp.linspace(0, 1, 1000)
    all_ks = []
    all_gs = []
    for k in config['kernels_params']['k']:
        all_ks.append(leniax_kernels.get_kernel(k, world_size, R, True))
        all_gs.append(leniax_gf.growth_fns[k['gf_id']](x, k['m'], k['s']) * k['h'])
    Ks = leniax_utils.crop_zero(jnp.vstack(all_ks))
    K_size = Ks.shape[-1]
    K_mid = K_size // 2

    fullpath = f"{save_dir}/Ks.png"
    fig, ax = plt.subplots(1, 2, figsize=(10, 2))

    ax[0].plot(range(K_size), Ks[:, K_mid, :].T)
    ax[0].title.set_text('Ks cross-sections')
    ax[0].set_xlim([K_mid - R - 3, K_mid + R + 3])

    ax[1].plot(x, jnp.asarray(all_gs).T)
    ax[1].axhline(y=0, color='grey', linestyle='dotted')
    ax[1].title.set_text('growths Gs')

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)


def gather_viz_data(main_dir: str):
    viz_dir = os.path.join(main_dir, 'viz_data')
    leniax_utils.check_dir(viz_dir)

    all_viz_data: List[Dict] = []
    for (subdir, _, _) in os.walk(main_dir):
        viz_data_filename = os.path.join(subdir, 'viz_data.json')
        if not os.path.isfile(viz_data_filename):
            continue

        with open(viz_data_filename, 'r') as f:
            current_viz_data = json.load(f)

        last_frame_fullpath = os.path.join(subdir, current_viz_data['img_url'])
        last_frame_new_name = str(len(all_viz_data)) + '-' + current_viz_data['img_url']
        last_frame_newpath = os.path.join(viz_dir, last_frame_new_name)
        copy2(last_frame_fullpath, last_frame_newpath)

        current_viz_data['img_url'] = last_frame_new_name
        all_viz_data.append(current_viz_data)

    with open(os.path.join(viz_dir, 'all_viz_data.json'), 'w') as f:
        json.dump(all_viz_data, f)
