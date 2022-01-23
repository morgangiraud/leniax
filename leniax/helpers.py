import time
import os
import copy
import json
import pickle
import functools
import numpy as np
import scipy
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, List, Union, Any
import matplotlib.pyplot as plt
from absl import logging as absl_logging

from .lenia import LeniaIndividual
from . import core as leniax_core
from . import runner as leniax_runner
from . import initializations as initializations
from . import statistics as leniax_stat
from . import loader as leniax_loader
from . import utils as leniax_utils
from . import growth_functions as leniax_gf
from . import video as leniax_video
from . import colormaps as leniax_colormaps
from .growth_functions import growth_fns
from .kernels import KernelMapping, get_kernels_and_mapping, get_kernel

cdir = os.path.dirname(os.path.realpath(__file__))


def init(config: Dict, use_init_cells: bool = True, fft: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, KernelMapping]:
    nb_dims = config['world_params']['nb_dims']
    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    kernels_params = config['kernels_params']['k']
    R = config['world_params']['R']
    assert len(world_size) == nb_dims
    assert nb_channels > 0

    raw_cells = leniax_loader.load_raw_cells(config, use_init_cells)

    scale = config['world_params']['scale']
    if scale != 1.:
        raw_cells = jnp.array([scipy.ndimage.zoom(raw_cells[i], scale, order=0) for i in range(nb_channels)],
                              dtype=jnp.float32)
        config['world_params']['R'] *= scale

    # assert cells.shape[1] * 2.2 < config['render_params']['world_size'][0]
    # assert cells.shape[2] * 2.2 < config['render_params']['world_size'][1]
    # on = round(max(cells.shape[1] * 1.25, cells.shape[2] * 1.25) / 2.)
    # init_cells = create_init_cells(world_size, nb_channels, [
    #     jnp.rot90(cells, k=2, axes=(1, 2)),
    #     jnp.rot90(cells, k=1, axes=(1, 2)),
    #     jnp.rot90(cells, k=0, axes=(1, 2)),
    #     jnp.rot90(cells, k=-1, axes=(1, 2)),
    # ], [
    #     [0, -on, -on],
    #     [0, -on, on],
    #     [0, on, on],
    #     [0, on, -on],
    # ])
    if len(raw_cells.shape) > 1 + nb_dims:
        init_cells = create_init_cells(world_size, nb_channels, raw_cells)
    else:
        init_cells = create_init_cells(world_size, nb_channels, [raw_cells])
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)

    return init_cells, K, mapping


def create_init_cells(
    world_size: List[int],
    nb_channels: int,
    other_cells: Union[jnp.ndarray, List[jnp.ndarray]] = [],
    offsets: List[List[int]] = [],
) -> jnp.ndarray:
    """
        Outputs:
            - cells: jnp.ndarray[N=1, C, world_dims...]
    """
    world_shape = [nb_channels] + world_size  # [C, world_dims...]
    cells = jnp.zeros(world_shape)
    if isinstance(other_cells, list):
        if len(offsets) == len(other_cells):
            for c, offset in zip(other_cells, offsets):
                if len(c) != 0:
                    cells = leniax_utils.merge_cells(cells, c, offset)
        else:
            for c in other_cells:
                if len(c) != 0:
                    cells = leniax_utils.merge_cells(cells, c)

        cells = cells[jnp.newaxis, ...]
    elif isinstance(other_cells, jnp.ndarray) or isinstance(other_cells, np.ndarray):
        if isinstance(other_cells, np.ndarray):
            other_cells = jnp.array(other_cells, dtype=jnp.float32)
        cells = other_cells
    else:
        raise ValueError(f'Don\'t know how to handle {type(other_cells)}')

    return cells


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

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    update_fn = build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    nb_channels_to_init = nb_channels * nb_init_search
    if update_fn_version == 'v1':
        rng_key, noises = initializations.perlin(rng_key, nb_channels_to_init, world_size, R, kernels_params[0])
    elif update_fn_version == 'v2':
        rng_key, noises = initializations.perlin_local(rng_key, nb_channels_to_init, world_size, R, kernels_params[0])
    else:
        raise ValueError('update_fn_version {update_fn_version} does not exist')
    init_noises = noises.reshape([nb_init_search, nb_channels] + world_size)

    all_cells_0_jnp = jnp.array([
        create_init_cells(world_size, nb_channels, [init_noises[i]]) for i in range(nb_init_search)
    ])

    best_run = {}
    current_max = 0
    for i in range(nb_init_search):
        # t0 = time.time()

        all_cells, _, _, all_stats = leniax_runner.run_scan(
            all_cells_0_jnp[i],
            K,
            gfn_params,
            kernels_weight_per_channel,
            T,
            max_run_iter,
            R,
            update_fn,
            compute_stats_fn
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


MUTATION_RATE = 1e-5


def search_for_mutation(
    rng_key: jnp.ndarray,
    config: Dict,
    nb_scale_for_stability: int = 1,
    fft: bool = True,
    use_init_cells: bool = True
) -> Tuple[jnp.ndarray, Dict, int]:
    render_params = config['render_params']
    world_size = render_params['world_size']

    run_params = config['run_params']
    nb_mut_search = run_params['nb_mut_search']
    max_run_iter = run_params['max_run_iter']
    best_run = {}
    current_max = 0
    for i in range(nb_mut_search):
        copied_config = copy.deepcopy(config)
        for gene in config['genotype']:
            val = leniax_utils.get_param(copied_config, gene['key'])
            val += np.random.normal(0., MUTATION_RATE)
            leniax_utils.set_param(copied_config, gene['key'], val)

        total_iter_done = 0
        nb_iter_done = 0
        for scale_power in range(nb_scale_for_stability):
            # t0 = time.time()
            scaled_config = copy.deepcopy(copied_config)
            scaled_config['render_params']['world_size'] = [ws * 2**scale_power for ws in world_size]
            scaled_config['world_params']['scale'] = 2**scale_power

            all_cells, _, _, stats_dict = init_and_run(
                scaled_config, use_init_cells=use_init_cells, with_jit=True, fft=fft
            )
            all_cells = all_cells[:, 0]
            stats_dict['N'].block_until_ready()

            nb_iter_done = max(nb_iter_done, stats_dict['N'])
            total_iter_done += stats_dict['N']

        # Current_max at all scale
        # print(f'total_iter_done: {total_iter_done}')
        if current_max < total_iter_done:
            current_max = total_iter_done
            best_run = {
                "N": nb_iter_done,
                "all_cells": all_cells,
                "all_stats": stats_dict,
                "config": copied_config,
            }

        if total_iter_done >= max_run_iter * nb_scale_for_stability:
            break

        # print(t1 - t0, nb_iter_done)

    return rng_key, best_run, i


def init_and_run(
    config: Dict,
    use_init_cells: bool = True,
    with_jit: bool = True,
    fft: bool = True,
    stat_trunc: bool = False,
    override: Dict[str, Any] = {}
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Initialize and simulate a Lenia configuration

    To simulate a configuration with multiple initializations you must set
    ``with_jit`` to ``True`` so the function use the scan implementaton.
    Tou must also set ``stat_trunc`` to ``False``: multiple initializartions
    means different simulation length measured by the statistics.

    Args:
        config: The Lenia configuration
        use_init_cells: Boolean to choose if the `init_cells` field in the configuraiton will be used
        with_jit: Boolean to choose if the jitted function will be used
        fft: Boolean to choose if the fft implementation will be used
        stat_trunc: Boolean to choose if the outputs will be truncated using the simulation statistics

    Returns:
        A tuple of `[nb_iter, nb_init, nb_channels, world_dims...]` shaped
        cells, fields, potentials and statistics of the simulation
    """
    config = copy.deepcopy(config)

    cells, K, mapping = init(config, use_init_cells, fft)
    if 'cells' in override:
        cells = override['cells']
    if 'K' in override:
        K = override['K']
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

    world_params = config['world_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    R = config['world_params']['R']
    T = jnp.array(config['world_params']['T'])

    max_run_iter = config['run_params']['max_run_iter']

    update_fn = build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if with_jit is True:
        all_cells, all_fields, all_potentials, stats_dict = leniax_runner.run_scan(
            cells, K, gfn_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn
        )
    else:
        all_cells, all_fields, all_potentials, stats_dict = leniax_runner.run(
            cells, K, gfn_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn
        )  # [nb_iter, nb_init, C, world_dims...]
    stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}

    if stat_trunc is True:
        all_cells = all_cells[:int(stats_dict['N'])]  # [nb_iter, C, world_dims...]
        all_fields = all_fields[:int(stats_dict['N'])]
        all_potentials = all_potentials[:int(stats_dict['N'])]

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

        K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)
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
        cells_0 = noises.reshape([nb_init_search, nb_channels] + world_size)

        all_cells_0.append(cells_0)
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
    all_cells0: jnp.ndarray,
    all_final_cells: jnp.ndarray,
    neg_fitness=False
) -> List[LeniaIndividual]:
    """
        Args:
            - inds:         List,                       Evaluated Lenia individuals
            - cells0s:      jnp.ndarray, [len(inds), nb_init, world_size...]
    """
    for i, ind in enumerate(inds):
        config = ind.get_config()

        ind_Ns = stats['N'][i]
        ind_cells0s = all_cells0[i]
        ind_final_cells = all_final_cells[i]

        best_init_idx = jnp.argmax(ind_Ns, axis=0)

        nb_steps = ind_Ns[best_init_idx]
        cells0 = ind_cells0s[best_init_idx]
        final_cells = ind_final_cells[best_init_idx]
        # print(i, ind_Ns, max_idx)

        ind.set_init_cells(leniax_loader.compress_array(cells0))
        ind.set_cells(leniax_loader.compress_array(leniax_utils.center_and_crop_cells(final_cells)))

        if neg_fitness is True:
            fitness = -nb_steps
        else:
            fitness = nb_steps
        ind.fitness = fitness

        config['behaviours'] = {}
        for k in stats.keys():
            if k == 'N':
                continue
            truncated_stat = stats[k][i, :int(nb_steps), best_init_idx]
            config['behaviours'][k] = truncated_stat[-128:].mean()

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
        print(f"Folder for id: {padded_id}, alreaady exist. Passing")
        # If the lenia folder already exists at that path, we consider the work alreayd done
        return
    leniax_utils.check_dir(save_dir)

    start_time = time.time()
    all_cells, _, _, stats_dict = init_and_run(config, use_init_cells=True, with_jit=True, fft=True)
    all_cells = all_cells[:int(stats_dict['N']), 0]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"[{padded_id}] - {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)

    dump_assets(save_dir, config, all_cells, stats_dict)


def build_update_fn(
    K_shape: Tuple[int, ...],
    mapping: KernelMapping,
    update_fn_version: str = 'v1',
    average_weight: bool = True,
    fft: bool = True,
) -> Callable:
    get_potential_fn = build_get_potential_fn(K_shape, mapping.true_channels, fft)
    get_field_fn = build_get_field_fn(mapping.cin_growth_fns, average_weight)
    if update_fn_version == 'v1':
        update_fn = leniax_core.update_cells
    elif update_fn_version == 'v2':
        update_fn = leniax_core.update_cells_v2
    else:
        raise ValueError(f"version {update_fn_version} does not exist")

    return functools.partial(
        leniax_core.update, get_potential_fn=get_potential_fn, get_field_fn=get_field_fn, update_fn=update_fn
    )


def build_get_potential_fn(kernel_shape: Tuple[int, ...], true_channels: jnp.ndarray, fft: bool = True) -> Callable:
    # First 2 dimensions are for fake batch dim and nb_channels
    if fft is True:
        max_k_per_channel = kernel_shape[1]
        C = kernel_shape[2]
        nb_dims = len(kernel_shape) - 3
        wdims_axes = tuple(range(-1, -nb_dims - 1, -1))

        return functools.partial(
            leniax_core.get_potential_fft,
            true_channels=true_channels,
            max_k_per_channel=max_k_per_channel,
            C=C,
            wdims_axes=wdims_axes
        )
    else:
        padding = jnp.array([[0, 0]] * 2 + [[dim // 2, dim // 2] for dim in kernel_shape[2:]])

        return functools.partial(leniax_core.get_potential, padding=padding, true_channels=true_channels)


def build_get_field_fn(cin_growth_fns: List[List[int]], average: bool = True) -> Callable:
    growth_fn_l = []
    for growth_fns_per_channel in cin_growth_fns:
        for gf_id in growth_fns_per_channel:
            growth_fn = growth_fns[gf_id]
            growth_fn_l.append(growth_fn)

    growth_fn_t = tuple(growth_fn_l)

    if average:
        weighted_fn = leniax_core.weighted_mean
    else:
        weighted_fn = leniax_core.weighted_sum

    return functools.partial(leniax_core.get_field, growth_fn_t=growth_fn_t, weighted_fn=weighted_fn)


###
# Viz
###
def dump_assets(save_dir: str, config: Dict, all_cells: jnp.ndarray, stats_dict: Dict, colormaps=None):
    if colormaps is None:
        colormaps = [
            leniax_colormaps.ExtendedColormap('extended')
        ]  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    leniax_utils.plot_stats(save_dir, stats_dict)

    plot_kernels(save_dir, config)

    with open(os.path.join(save_dir, 'stats_dict.p'), 'wb') as f:
        pickle.dump(stats_dict, f)

    # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
    #     np.save(f, np.array(all_cells))

    dump_last_frame(save_dir, all_cells, True, colormaps[0])

    dump_viz_data(save_dir, stats_dict, config)

    render_params = config['render_params']
    all_outputs_fullpath = leniax_video.dump_video(save_dir, all_cells, render_params, colormaps)

    for output_fullpath in all_outputs_fullpath:
        leniax_video.dump_gif(output_fullpath)


def dump_last_frame(save_dir: str, all_cells: jnp.ndarray, raw: bool = False, colormap=None):
    """
        Args:
            - save_dir: str
            - all_cells: jnp.ndarray[nb_iter, C, world_dims...]
    """
    last_frame = all_cells[-1]
    dump_frame(save_dir, 'last_frame', last_frame, raw, colormap)


def dump_frame(save_dir: str, filename: str, cells: jnp.ndarray, raw: bool = False, colormap=None):
    if raw is False:
        cells = leniax_utils.center_and_crop_cells(cells)
    if colormap is None:
        colormap = plt.get_cmap('plasma')

    # with open(os.path.join(save_dir, f"{filename}.p"), 'wb') as f:
    #     pickle.dump(np.array(cells), f)

    img = leniax_utils.get_image(np.array(cells), 1, colormap)
    with open(os.path.join(save_dir, f"{filename}.png"), 'wb') as f:
        img.save(f, format='png')


def dump_viz_data(save_dir: str, stats_dict: Dict, config: Dict):
    viz_data: Dict = {'stats': {}}
    for k, v in stats_dict.items():
        if k == 'N':
            continue
        truncated_stat = v[:int(stats_dict['N'])]
        viz_data['stats'][k + '_mean'] = round(float(truncated_stat[-128:].mean()), 5)
        viz_data['stats'][k + '_std'] = round(float(truncated_stat[-128:].std()), 5)
    viz_data['img_url'] = 'last_frame.png'
    viz_data['k'] = config['kernels_params']['k']
    viz_data['R'] = config['world_params']['R']
    viz_data['T'] = config['world_params']['T']
    with open(os.path.join(save_dir, 'viz_data.json'), 'w') as fviz:
        json.dump(viz_data, fviz)


def plot_kernels(save_dir, config):
    world_size = config['render_params']['world_size']
    R = config['world_params']['R']

    x = jnp.linspace(0, 1, 1000)
    all_ks = []
    all_gs = []
    for k in config['kernels_params']['k']:
        all_ks.append(get_kernel(k, world_size, R, True))
        gfn_params = np.array([k['m'], k['s']])
        all_gs.append(leniax_gf.growth_fns[k['gf_id']](gfn_params, x) * k['h'])
    Ks = leniax_utils.crop_zero(jnp.vstack(all_ks))
    nb_Ks = Ks.shape[0]

    # Plot kernels image where color represent intensity
    rows = int(nb_Ks**0.5)
    cols = nb_Ks // rows
    axes = []
    fig = plt.figure(figsize=(6, 6))

    # We assume the kernel has the same size on all dimensions
    K_size = Ks.shape[-1]
    K_mid = K_size // 2
    vmax = Ks.max()
    fullpath = f"{save_dir}/Ks.png"
    if len(Ks.shape) == 3:
        # 2D kernels

        for i in range(nb_Ks):
            axes.append(fig.add_subplot(rows, cols, i + 1))
            axes[-1].title.set_text(f"kernel K{i}")
            plt.imshow(Ks[i], cmap='viridis', interpolation="nearest", vmin=0, vmax=vmax)
    elif len(Ks.shape) == 4:
        # 3D kernels
        for i in range(nb_Ks):
            K_to_plot = np.array(Ks[i] > 0)  # Need a real numpy array to work with the voxels function
            K_to_plot[K_mid:] = 0
            K_to_plot[:, :, K_mid:] = 0

            colors = plt.get_cmap('viridis')(Ks[i] / vmax)
            colors[:, :, :, 3] = 0.5

            axes.append(fig.add_subplot(rows, cols, i + 1, projection='3d'))
            axes[-1].title.set_text(f"kernel K{i}")
            axes[-1].voxels(K_to_plot, facecolors=colors)
    else:
        raise ValueError('We do not support plotting kernels containing more than 3 dimensions')
    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)

    # Plot Kernels and growth functions
    fullpath = f"{save_dir}/Ks_graph.png"
    fig, ax = plt.subplots(1, 2, figsize=(10, 2))

    if len(Ks.shape) == 3:
        ax[0].plot(range(K_size), Ks[:, K_mid, :].T)
    elif len(Ks.shape) == 4:
        ax[0].plot(range(K_size), Ks[:, K_mid, K_mid, :].T)
    ax[0].title.set_text('Ks cross-sections')
    ax[0].set_xlim([K_mid - R - 3, K_mid + R + 3])

    ax[1].plot(x, jnp.asarray(all_gs).T)
    ax[1].axhline(y=0, color='grey', linestyle='dotted')
    ax[1].title.set_text('growths Gs')

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)
