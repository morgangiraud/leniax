"""Leniax helper functions

Those functions are provided to ease the use of this library. See them as template
gluing core functions together to achieve common usages.
"""
import os
import copy
import logging
import time
import json
import pickle
import functools
import numpy as np
import scipy
import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, List, Union, Optional
import matplotlib.pyplot as plt

from . import core as leniax_core
from . import runner as leniax_runner
from . import initializations as leniax_init
from . import statistics as leniax_stat
from . import loader as leniax_loader
from . import utils as leniax_utils
from . import video as leniax_video
from . import colormaps as leniax_colormaps
from . import kernels as leniax_kernels
from .kernel_functions import register as kf_register
from .growth_functions import register as gf_register

cdir = os.path.dirname(os.path.realpath(__file__))


def init(
    config: Dict,
    use_init_cells: bool = True,
    fft: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, leniax_kernels.KernelMapping]:
    """Construct the initial state and metadata to run a simulation.

    Args:
        config: Leniax confguration
        use_init_cells: Set to ``True`` to use the ``init_cells`` configuration property.
        fft: Set to ``True`` to use FFT optimization

    Returns:
        A 3-tuple representing the initial state, Kernel and mapping data
    """
    nb_dims = config['world_params']['nb_dims']
    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    kernels_params = config['kernels_params']
    assert len(world_size) == nb_dims
    assert nb_channels > 0

    raw_cells = leniax_loader.load_raw_cells(config, use_init_cells)

    scale = config['world_params']['scale']
    if scale != 1.:
        raw_cells = jnp.array(
            [scipy.ndimage.zoom(raw_cells[i], scale, order=0) for i in range(nb_channels)],
            dtype=jnp.float32,
        )
        # We update the configuration here, the new R value will be used in the statistics
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
    K, mapping = leniax_kernels.get_kernels_and_mapping(kernels_params, world_size, nb_channels, config['world_params']['R'], fft)

    return init_cells, K, mapping


def create_init_cells(
    world_size: List[int],
    nb_channels: int,
    other_cells: Union[jnp.ndarray, List[jnp.ndarray]] = [],
    offsets: List[List[int]] = [],
) -> jnp.ndarray:
    """Construct the initial state

    Args:
        world_size: World size
        nb_channels: Number of world channels
        other_cells: Other initial states to merge
        offsets: Offsets used to merge other initial states

    Returns:
        The initial state
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


def init_and_run(
    rng_key: jax.random.KeyArray,
    config: Dict,
    use_init_cells: bool = True,
    with_jit: bool = True,
    fft: bool = True,
    stat_trunc: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Initialize and simulate a Lenia configuration

    To simulate a configuration with multiple initializations you must set:
    - ``with_jit=True`` so the function use the scan implementaton.
    - ``stat_trunc=False`` multiple initializations means different simulation length measured by the statistics.

    Args:
        rng_key: JAX PRNG key.
        config: Lenia configuration
        use_init_cells: Set to ``True`` to use the ``init_cells`` configuration property.
        with_jit: Set to ``True`` to use the jitted scan implementation
        fft: Set to ``True`` to use FFT optimization
        stat_trunc: Set to ``True`` to truncate run based on its statistics

    Returns:
        A tuple of `[nb_iter, nb_init, nb_channels, world_dims...]` shaped
        cells, fields, potentials and statistics of the simulation.
    """
    config = copy.deepcopy(config)

    cells, K, mapping = init(config, use_init_cells, fft)
    gf_params = mapping.get_gf_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

    world_params = config['world_params']
    get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    R = config['world_params']['R']
    T = jnp.array(config['world_params']['T'], dtype=jnp.float32)

    max_run_iter = config['run_params']['max_run_iter']

    update_fn = build_update_fn(K.shape, mapping, get_state_fn_slug, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if with_jit is True:
        all_cells, all_fields, all_potentials, stats_dict = leniax_runner.run_scan(
            rng_key, cells, K, gf_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn
        )
    else:
        all_cells, all_fields, all_potentials, stats_dict = leniax_runner.run(
            rng_key, cells, K, gf_params, kernels_weight_per_channel, T, max_run_iter, R, update_fn, compute_stats_fn, stat_trunc
        )  # [nb_iter, nb_init, C, world_dims...]
    stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}

    if stat_trunc is True:
        all_cells = all_cells[:int(stats_dict['N'])]  # [nb_iter, C, world_dims...]
        all_fields = all_fields[:int(stats_dict['N'])]
        all_potentials = all_potentials[:int(stats_dict['N'])]

    return all_cells, all_fields, all_potentials, stats_dict


def search_for_mutation(
    rng_key: jax.random.KeyArray,
    config: Dict,
    nb_scale_for_stability: int = 1,
    use_init_cells: bool = True,
    fft: bool = True,
    mutation_rate: float = 1e-5,
) -> Tuple[Dict, int]:
    """Search for a stable mutation

    Args:
        rng_key: JAX PRNG key.
        config: Lenia configuration.
        use_init_cells: Set to ``True`` to use the ``init_cells`` configuration property.
        fft: Set to ``True`` to use FFT optimization.
        nb_scale_for_stability: Number of time the configuration will be scaled and tested.
        mutation_rate: Mutation rate.

    Returns:
        A 2-tuple of a dictionnary with the best run data and
        the number of runs made to find it
    """
    render_params = config['render_params']
    world_size = render_params['world_size']

    run_params = config['run_params']
    nb_mut_search = run_params['nb_mut_search']
    max_run_iter = run_params['max_run_iter']

    best_run = {}
    current_max = 0
    nb_genes = len(config['genotype'])
    rng_key, *subkeys = jax.random.split(rng_key, nb_mut_search * nb_genes + 1)
    for i in range(nb_mut_search):
        copied_config = copy.deepcopy(config)
        for gene_i, gene in enumerate(config['genotype']):
            val = leniax_utils.get_param(copied_config, gene['key'])

            subkey = subkeys[i * nb_genes + gene_i]
            val += jax.random.normal(subkey, dtype=jnp.float32) * mutation_rate

            leniax_utils.set_param(copied_config, gene['key'], float(val))

        total_iter_done = 0
        nb_iter_done = 0
        for scale_power in range(nb_scale_for_stability):
            # t0 = time.time()
            scaled_config = copy.deepcopy(copied_config)
            scaled_config['render_params']['world_size'] = [ws * 2**scale_power for ws in world_size]
            scaled_config['world_params']['scale'] = 2**scale_power

            # We do not split the rng_key here, because we want to keep the exact same run
            # while just changing the rendering size
            all_cells, _, _, stats_dict = init_and_run(
                rng_key, scaled_config, use_init_cells=use_init_cells, with_jit=True, fft=fft
            )
            all_cells = all_cells[:, 0]
            stats_dict['N'].block_until_ready()

            nb_iter_done = max(nb_iter_done, stats_dict['N'])
            total_iter_done += stats_dict['N']

        # Current_max at all scale
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

    return best_run, i


def search_for_init(
    rng_key: jax.random.KeyArray,
    config: Dict,
    fft: bool = True,
) -> Tuple[Dict, int]:
    """Search for a stable initial state

    Args:
        rng_key: JAX PRNG key.
        config: Lenia configuration.
        fft: Set to ``True`` to use FFT optimization.

    Returns:
        A 2-tuple of a dictionnary with the best run data and
        the number of runs made to find it
    """
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    R = world_params['R']
    T = jnp.array(world_params['T'], dtype=jnp.float32)

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    init_slug = config['algo']['init_slug']

    K, mapping = leniax_kernels.get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)
    gf_params = mapping.get_gf_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    update_fn = build_update_fn(K.shape, mapping, get_state_fn_slug, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    nb_channels_to_init = nb_channels * nb_init_search
    rng_key, noises = leniax_init.register[init_slug](
        rng_key, nb_channels_to_init, world_size, R, kernels_params[0]['gf_params']
    )
    init_noises = noises.reshape([nb_init_search, nb_channels] + world_size)

    all_cells0_l = [create_init_cells(world_size, nb_channels, [init_noises[i]]) for i in range(nb_init_search)]
    all_cells0_jnp = jnp.array(all_cells0_l, dtype=jnp.float32)

    best_run = {}
    current_max = 0
    subkeys = jax.random.split(rng_key, nb_init_search)
    for i in range(nb_init_search):
        all_cells, _, _, all_stats = leniax_runner.run_scan(
            subkeys[i],
            all_cells0_jnp[i],
            K,
            gf_params,
            kernels_weight_per_channel,
            T,
            max_run_iter,
            R,
            update_fn,
            compute_stats_fn
        )
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        all_stats['N'].block_until_ready()

        nb_iter_done = all_stats['N']
        if current_max < nb_iter_done:
            current_max = nb_iter_done
            best_run = {"N": nb_iter_done, "all_cells": all_cells, "all_stats": all_stats}

        if nb_iter_done >= max_run_iter:
            break

    return best_run, i


###
# Pipeline dynamic builder
###
def build_update_fn(
    kernel_shape: Tuple[int, ...],
    mapping: leniax_kernels.KernelMapping,
    get_state_fn_slug: str = 'v1',
    average_weight: bool = True,
    fft: bool = True,
) -> Callable:
    """Construct an Leniax update function

    An update function allows one to update a simulation state.

    Args:
        kernel_shape: Kernel shape.
        mapping: Mapping data.
        get_state_fn_slug: Which version of Lenia should be run
        fft: Set to ``True`` to use FFT optimization

    Returns:
        A Leniax update function
    """
    get_potential_fn = build_get_potential_fn(kernel_shape, mapping.true_channels, fft)
    get_field_fn = build_get_field_fn(mapping.cin_gfs, average_weight)
    get_state_fn = leniax_core.register[get_state_fn_slug]

    return functools.partial(
        leniax_core.update, get_potential_fn=get_potential_fn, get_field_fn=get_field_fn, get_state_fn=get_state_fn
    )


def build_get_potential_fn(
    kernel_shape: Tuple[int, ...],
    true_channels: Optional[List[bool]] = None,
    fft: bool = True,
    channel_first: bool = True,
) -> Callable:
    """Construct an Leniax potential function

    A potential function allows one to compute the potential from a Lenia state.

    Args:
        kernel_shape: Kernel shape.
        true_channels: Boolean array indicating the true potential channels
        fft: Set to ``True`` to use FFT optimization

    Returns:
        A Leniax potential function
    """
    # First 2 dimensions are for fake batch dim and nb_channels
    if true_channels is not None:
        tc_indices_l = []
        for i in range(len(true_channels)):
            if true_channels[i] is True:
                tc_indices_l.append(i)
        tc_indices = tuple(tc_indices_l)
    else:
        tc_indices = None

    if fft is True:
        return functools.partial(
            leniax_core.get_potential_fft,
            tc_indices=tc_indices,
            channel_first=channel_first,
        )
    else:
        if channel_first is True:
            pad_l = [(0, 0), (0, 0)]
            for dim in kernel_shape[2:]:
                if dim % 2 == 0:
                    pad_l += [(dim // 2, dim // 2 - 1)]
                else:
                    pad_l += [(dim // 2, dim // 2)]
        else:
            pad_l = [(0, 0)]
            for dim in kernel_shape[:2]:
                if dim % 2 == 0:
                    pad_l += [(dim // 2, dim // 2 - 1)]
                else:
                    pad_l += [(dim // 2, dim // 2)]
            pad_l += [(0, 0)]

        padding = tuple(pad_l)

        return functools.partial(
            leniax_core.get_potential,
            tc_indices=tc_indices,
            padding=padding,
            channel_first=channel_first,
        )


def build_get_field_fn(cin_gfs: List[List[str]], average: bool = True) -> Callable:
    """Construct an Leniax field function

    A field function allows one to compute the field from a Lenia potential.

    Args:
        cin_gfs: List of growth functions per channel.
        average: Set to ``True`` to average instead of summing input channels

    Returns:
        A Leniax field function
    """
    growth_fn_l = []
    for growth_fns_per_channel in cin_gfs:
        for gf_slug in growth_fns_per_channel:
            growth_fn = gf_register[gf_slug]
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
def dump_assets(
    save_dir: str,
    config: Dict,
    all_cells: jnp.ndarray,
    stats_dict: Dict,
    colormaps: List = [],
    transparent_bg: bool = False
):
    """Dump a set of interesting assets.

        Those assets include:
            - Simulation statistics (plots and data)
            - Kernels and growth functions plots
            - Last frame
            - Video and Gif of the simulation

    Args:
        save_dir: directory used to save assets.
        config: Leniax configuration.
        all_cells: Simulation data of shape ``[nb_iter, C, world_dims...]``.
        stats_dict: Leniax statistics dictionnary.
        colormaps: A List of matplotlib compatible colormap.
        transparent_bg: Set to ``True`` to make the background transparent.
    """
    if len(colormaps) == 0:
        colormaps = [
            leniax_colormaps.ExtendedColormap('extended')
        ]  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    leniax_utils.plot_stats(save_dir, stats_dict)

    plot_kernels(save_dir, config)

    with open(os.path.join(save_dir, 'stats_dict.p'), 'wb') as f:
        pickle.dump(stats_dict, f)

    # with open(os.path.join(save_dir, 'cells.p'), 'wb') as f:
    #     np.save(f, np.array(all_cells))

    dump_last_frame(save_dir, all_cells, False, colormaps[0])

    dump_viz_data(save_dir, config, stats_dict)

    logging.info("Video rendering: start.")
    start_time = time.time()
    all_outputs_fullpath = leniax_video.render_video(
        save_dir, all_cells, config['render_params'], colormaps, '', transparent_bg
    )
    total_time = time.time() - start_time
    logging.info(
        f"Video rendering: stop. {len(all_cells)} states computed in {total_time:.2f} seconds, {len(all_cells) / total_time:.2f} fps."
    )

    logging.info("Gif rendering: start.")
    start_time = time.time()
    for output_fullpath in all_outputs_fullpath:
        leniax_video.render_gif(output_fullpath)
    total_time = time.time() - start_time
    logging.info(
        f"Gif rendering: stop. {len(all_cells)} states computed in {total_time:.2f} seconds, {len(all_cells) / total_time:.2f} fps."
    )


def dump_last_frame(
    save_dir: str,
    all_cells: jnp.ndarray,
    center_and_crop: bool = True,
    colormap=None,
):
    """Dump the last frame of the simulation

    The dumped last frame is called **last_frame.png**.

    Args:
        save_dir: directory used to save assets.
        all_cells: Simulation data of shape ``[nb_iter, C, world_dims...]``.
        center_and_crop: Set to ``True`` to center the stable pattern and crop the margin.
        colormap: A  matplotlib compatible colormap.
    """
    last_frame = all_cells[-1]
    dump_frame(save_dir, 'last_frame', last_frame, center_and_crop, colormap)


def dump_frame(
    save_dir: str,
    filename: str,
    cells: jnp.ndarray,
    center_and_crop: bool = True,
    colormap=None,
):
    """Dump a Lenia state as a image

    Args:
        save_dir: directory used to save assets.
        filename: File name.
        cells: A Lenia state of shape ``[C, world_dims...]``.
        center_and_crop: Set to ``True`` to center the stable pattern and crop the margin.
        colormap: A  matplotlib compatible colormap.
    """
    if center_and_crop is True:
        cells = leniax_utils.center_and_crop(cells)
    if colormap is None:
        colormap = plt.get_cmap('plasma')

    # with open(os.path.join(save_dir, f"{filename}.p"), 'wb') as f:
    #     pickle.dump(np.array(cells), f)

    img = leniax_utils.get_image(np.array(cells), 1, colormap)
    with open(os.path.join(save_dir, f"{filename}.png"), 'wb') as f:
        img.save(f, format='png')


def dump_viz_data(save_dir: str, config: Dict, stats_dict: Dict):
    """Dump vizualization data as JSON

    Args:
        save_dir: directory used to save assets.
        config: Leniax configuration.
        stats_dict: Leniax statistics dictionnary.
    """

    viz_data: Dict = {'stats': {}}
    for k, v in stats_dict.items():
        if k == 'N':
            continue
        truncated_stat = v[:int(stats_dict['N'])]
        viz_data['stats'][k + '_mean'] = round(float(truncated_stat[-128:].mean()), 5)
        viz_data['stats'][k + '_std'] = round(float(truncated_stat[-128:].std()), 5)
    viz_data['img_url'] = 'last_frame.png'
    viz_data['k'] = config['kernels_params']
    viz_data['R'] = config['world_params']['R']
    viz_data['T'] = config['world_params']['T']
    with open(os.path.join(save_dir, 'viz_data.json'), 'w') as fviz:
        json.dump(viz_data, fviz)


def plot_kernels(save_dir: str, config: Dict):
    """Plots kernels and growth functions

    Args:
        save_dir: directory used to save assets.
        config: Leniax configuration.
    """

    R = config['world_params']['R']
    scale = config['world_params']['scale']
    C = config['world_params']['nb_channels']

    x = jnp.linspace(0, 1, 1000)
    Ks, _ = leniax_kernels.get_kernels_and_mapping(
        config['kernels_params'], config['render_params']['world_size'], C, R, fft=False
    )
    Ks = Ks[:, 0]
    nb_Ks = Ks.shape[0]
    all_kfs = []
    all_gfs = []
    for param in config['kernels_params']:
        if param['kf_slug'] != '':
            all_kfs.append(kf_register[param['kf_slug']](param['kf_params'], x))
        all_gfs.append(gf_register[param['gf_slug']](param["gf_params"], x) * param['h'])

    # Plot kernels image where color represent intensity
    rows = int(nb_Ks**0.5)
    cols = nb_Ks // rows
    axes = []
    fig = plt.figure(figsize=(6, 6))

    # We assume the kernel has the same size on all dimensions
    K_size = Ks.shape[-1]
    K_mid = K_size // 2
    vmax = Ks.max()
    vmin = Ks.min()
    fullpath = f"{save_dir}/Ks.png"
    if len(Ks.shape) == 3:
        # 2D kernels
        for i in range(nb_Ks):
            axes.append(fig.add_subplot(rows, cols, i + 1))
            axes[-1].title.set_text(f"kernel K{i}")
            plt.imshow(Ks[i], cmap='viridis', interpolation="nearest", vmin=vmin, vmax=vmax)
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
    fig, ax = plt.subplots(1, 3, figsize=(10, 2))

    if len(all_kfs) > 0:
        ax[0].plot(x, jnp.asarray(all_kfs).T)
        ax[0].axhline(y=0, color='grey', linestyle='dotted')
        ax[0].title.set_text('Kernel functions')

    if len(Ks.shape) == 3:
        ax[1].plot(range(K_size), Ks[:, K_mid, :].T)
    elif len(Ks.shape) == 4:
        ax[1].plot(range(K_size), Ks[:, K_mid, K_mid, :].T)
    ax[1].title.set_text('Ks cross-sections')
    ax[1].set_xlim([K_mid - 3 - R * scale, K_mid + 3 + R * scale])

    ax[2].plot(x, jnp.asarray(all_gfs).T)
    ax[2].axhline(y=0, color='grey', linestyle='dotted')
    ax[2].title.set_text('Growth functions')

    plt.tight_layout()
    fig.savefig(fullpath)
    plt.close(fig)
