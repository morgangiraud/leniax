import os
import pickle
import functools
from jax import vmap, pmap, lax, jit
import jax.numpy as jnp
import scipy
from typing import Callable, List, Dict, Tuple

from . import utils
from . import statistics as leniax_stat
from .kernels import get_kernels_and_mapping, KernelMapping
from .constant import EPSILON, START_CHECK_STOP


def init(config: Dict, fft: bool = True, use_init_cells: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, KernelMapping]:
    nb_dims = config['world_params']['nb_dims']
    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    kernels_params = config['kernels_params']['k']
    assert len(world_size) == nb_dims
    assert nb_channels > 0

    cells = load_raw_cells(config, use_init_cells)

    scale = config['world_params']['scale']
    if scale != 1.:
        cells = jnp.array([scipy.ndimage.zoom(cells[i], scale, order=0) for i in range(nb_channels)], dtype=jnp.float32)
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
    init_cells = create_init_cells(world_size, nb_channels, [cells])
    R = config['world_params']['R']
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R, fft)

    return init_cells, K, mapping


def load_raw_cells(config: Dict, use_init_cells: bool = True) -> jnp.ndarray:
    nb_dims = config['world_params']['nb_dims']
    if use_init_cells is True:
        if 'init_cells' not in config['run_params']:
            # Backward compatibility
            cells = config['run_params']['cells']
        else:
            cells = config['run_params']['init_cells']
    else:
        cells = config['run_params']['cells']
    if type(cells) is str:
        if cells == 'MISSING':
            cells = jnp.array([])
        elif cells == 'last_frame.p':
            with open(os.path.join(config['main_path'], 'last_frame.p'), 'rb') as f:
                cells = jnp.array(pickle.load(f), dtype=jnp.float32)
        else:
            cells = utils.decompress_array(cells, nb_dims + 1)  # we add the channel dim
    elif type(cells) is list:
        cells = jnp.array(cells, dtype=jnp.float32)

    # We repair the missing channel in case of the single channel got squeezed out
    if len(cells.shape) == nb_dims and config['world_params']['nb_channels'] == 1:
        cells = jnp.expand_dims(cells, 0)

    return cells


def create_init_cells(
    world_size: List[int],
    nb_channels: int,
    other_cells: List[jnp.ndarray] = [],
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
                    cells = utils.merge_cells(cells, c, offset)
        else:
            for c in other_cells:
                if len(c) != 0:
                    cells = utils.merge_cells(cells, c)
    else:
        raise ValueError(f'Don\'t know how to handle {type(other_cells)}')

    cells = cells[jnp.newaxis, ...]

    return cells


def run(
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    assert max_run_iter > 0, f"max_run_iter must be positive, value given: {max_run_iter}"
    assert cells.shape[0] == 1
    # cells shape: [N=1, C, dims...]
    N = 1
    nb_world_dims = cells.ndim - 2

    all_cells = [cells]
    all_fields = []
    all_potentials = []
    all_stats = []
    init_mass = cells.sum()

    previous_mass = init_mass
    previous_sign = jnp.zeros(N)
    counters = leniax_stat.init_counters(N)
    should_continue = jnp.ones(N)
    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    mass_centroid = jnp.zeros([nb_world_dims, N])
    mass_angle = jnp.zeros([N])
    for current_iter in range(max_run_iter):
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel, 1. / T)

        stat_t, total_shift_idx, mass_centroid, mass_angle = compute_stats_fn(
            cells, field, potential, total_shift_idx, mass_centroid, mass_angle
        )

        cells = new_cells
        all_cells.append(cells)
        all_fields.append(field)
        all_potentials.append(potential)
        all_stats.append(stat_t)

        mass = stat_t['mass']
        cond = leniax_stat.min_mass_heuristic(EPSILON, mass)
        should_continue_cond = cond
        cond = leniax_stat.max_mass_heuristic(init_mass, mass)
        should_continue_cond *= cond

        sign = jnp.sign(mass - previous_mass)
        monotone_counter = counters['nb_monotone_step']
        cond, counters['nb_monotone_step'] = leniax_stat.monotonic_heuristic(sign, previous_sign, monotone_counter)
        should_continue_cond *= cond

        mass_volume = stat_t['mass_volume']
        mass_volume_counter = counters['nb_max_volume_step']
        cond, counters['nb_max_volume_step'] = leniax_stat.mass_volume_heuristic(mass_volume, mass_volume_counter, R)
        should_continue_cond *= cond

        should_continue *= should_continue_cond

        # We avoid dismissing a simulation during the init period
        if current_iter >= START_CHECK_STOP and should_continue == 0:
            break

    # To keep the same number of elements per array
    all_cells.pop()

    all_cells_jnp = jnp.array(all_cells)
    all_fields_jnp = jnp.array(all_fields)
    all_potentials_jnp = jnp.array(all_potentials)

    stats_dict = leniax_stat.stats_list_to_dict(all_stats)
    stats_dict['N'] = jnp.array(current_iter)

    return all_cells_jnp, all_fields_jnp, all_potentials_jnp, stats_dict


@functools.partial(jit, static_argnums=(5, 6, 7, 8))
def run_scan(
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
        Args:
            - cells0:                       jnp.ndarray[N_init, nb_channels, world_dims...]
            - K:
                - fft:                      jnp.ndarray[1, nb_channels, max_k_per_channel, K_dims...]
                - raw:                      jnp.ndarray[nb_channels * max_k_per_channel, 1, K_dims...]
            - gfn_params:                   jnp.ndarray[nb_kernels, nb_gfn_params]
            - kernels_weight_per_channel:   jnp.ndarray[nb_channels, nb_kernels]
            - T:                            jnp.ndarray[]
    """

    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2

    def fn(carry, x):
        cells, K, gfn_params, kernels_weight_per_channel, T = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel, 1. / T)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        mass_centroid = stat_props['mass_centroid']
        mass_angle = stat_props['mass_angle']
        stats, total_shift_idx, mass_centroid, mass_angle = compute_stats_fn(
            cells, field, potential, total_shift_idx, mass_centroid, mass_angle
        )

        y = {'cells': cells, 'field': field, 'potential': potential, 'stats': stats}

        new_carry = {
            'fn_params': (new_cells, K, gfn_params, kernels_weight_per_channel, T),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
                'mass_centroid': mass_centroid,
                'mass_angle': mass_angle,
            }
        }

        return new_carry, y

    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    mass_centroid = jnp.zeros([nb_world_dims, N])
    mass_angle = jnp.zeros([N])
    init_carry = {
        'fn_params': (cells0, K, gfn_params, kernels_weight_per_channel, T),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
            'mass_centroid': mass_centroid,
            'mass_angle': mass_angle,
        }
    }

    _, ys = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    continue_stat = leniax_stat.check_heuristics(ys['stats'], R, 1. / T)
    ys['stats']['N'] = continue_stat.sum(axis=0)

    return ys['cells'], ys['field'], ys['potential'], ys['stats']


@functools.partial(jit, static_argnums=(5, 6, 7, 8))
@functools.partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None), out_axes=0)
def run_scan_mem_optimized(
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
        Args:
            - cells0:                       jnp.ndarray[N_sols, N_init, nb_channels, world_dims...]
            - K:
                fft:                        jnp.ndarray[N_sols, 1, nb_channels, max_k_per_channel, K_dims...]
                raw:                        jnp.ndarray[N_sols, nb_channels * max_k_per_channel, 1, K_dims...]
            - gfn_params:                   jnp.ndarray[N_sols, nb_kernels, nb_gfn_params]
            - kernels_weight_per_channel:   jnp.ndarray[N_sols, nb_channels, nb_kernels]
            - T:                            jnp.ndarray[N_sols]
    """
    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2

    def fn(carry, x):
        cells, K, gfn_params, kernels_weight_per_channel, T = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel, 1. / T)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        mass_centroid = stat_props['mass_centroid']
        mass_angle = stat_props['mass_angle']
        stats, total_shift_idx, mass_centroid, mass_angle = compute_stats_fn(
            cells, field, potential, total_shift_idx, mass_centroid, mass_angle
        )

        new_carry = {
            'fn_params': (new_cells, K, gfn_params, kernels_weight_per_channel, T),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
                'mass_centroid': mass_centroid,
                'mass_angle': mass_angle,
            }
        }

        return new_carry, stats

    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    mass_centroid = jnp.zeros([nb_world_dims, N])
    mass_angle = jnp.zeros([N])
    init_carry = {
        'fn_params': (cells0, K, gfn_params, kernels_weight_per_channel, T),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
            'mass_centroid': mass_centroid,
            'mass_angle': mass_angle,
        }
    }

    final_carry, stats = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    final_cells = final_carry['fn_params'][0]

    continue_stat = leniax_stat.check_heuristics(stats, R, 1. / T)
    stats['N'] = continue_stat.sum(axis=0)

    return stats, final_cells


@functools.partial(
    pmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None), out_axes=0, static_broadcasted_argnums=(5, 6, 7, 8)
)
@functools.partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None), out_axes=0)
def run_scan_mem_optimized_pmap(
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """
        Args:
            - cells0:                       jnp.ndarray[N_device, N_sols, N_init, nb_channels, world_dims...]
            - K:
                fft:                        jnp.ndarray[N_device, N_sols, 1, nb_channels, max_k_per_channel, K_dims...]
                raw:                        jnp.ndarray[N_device, N_sols, nb_channels * max_k_per_channel, 1, K_dims...]
            - gfn_params:                   jnp.ndarray[N_device, N_sols, nb_kernels, nb_gfn_params]
            - kernels_weight_per_channel:   jnp.ndarray[N_device, N_sols, nb_channels, nb_kernels]
            - T:                            jnp.ndarray[N_device, N_sols]
    """
    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2

    def fn(carry, x):
        cells, K, gfn_params, kernels_weight_per_channel, T = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel, 1. / T)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        mass_centroid = stat_props['mass_centroid']
        mass_angle = stat_props['mass_angle']
        stats, total_shift_idx, mass_centroid, mass_angle = compute_stats_fn(
            cells, field, potential, total_shift_idx, mass_centroid, mass_angle
        )

        new_carry = {
            'fn_params': (new_cells, K, gfn_params, kernels_weight_per_channel, T),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
                'mass_centroid': mass_centroid,
                'mass_angle': mass_angle,
            }
        }

        return new_carry, stats

    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    mass_centroid = jnp.zeros([nb_world_dims, N])
    mass_angle = jnp.zeros([N])
    init_carry = {
        'fn_params': (cells0, K, gfn_params, kernels_weight_per_channel, T),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
            'mass_centroid': mass_centroid,
            'mass_angle': mass_angle,
        }
    }

    final_carry, stats = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    final_cells = final_carry['fn_params'][0]

    continue_stat = leniax_stat.check_heuristics(stats, R, 1. / T)
    stats['N'] = continue_stat.sum(axis=0)

    return stats, final_cells


@functools.partial(jit, static_argnums=(5, 6, 7))
def update(
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    dt: jnp.ndarray,
    get_potential_fn: Callable,
    get_field_fn: Callable,
    update_fn: Callable,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
        Args:
            - cells:                        jnp.ndarray[N, nb_channels, world_dims...]
            - K:                            jnp.ndarray[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]
            - gfn_params:                   jnp.ndarray[nb_kernels, 2]
            - kernels_weight_per_channel:   jnp.ndarray[nb_channels, nb_kernels]
            - T:                            jnp.ndarray[N]

        Outputs:
            - cells:        jnp.ndarray[N, nb_channels, world_dims...]
            - field:        jnp.ndarray[N, nb_channels, world_dims...]
            - potential:    jnp.ndarray[N, nb_channels, world_dims...]
    """

    potential = get_potential_fn(cells, K)
    field = get_field_fn(potential, gfn_params, kernels_weight_per_channel)
    cells = update_fn(cells, field, dt)

    return cells, field, potential


# @functools.partial(jit, static_argnums=(3, 4, 5))
def get_potential_fft(
    cells: jnp.ndarray, K: jnp.ndarray, true_channels: jnp.ndarray, max_k_per_channel: int, C: int, axes: Tuple[int]
) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray[N, nb_channels, world_dims...]
            - K: jnp.ndarray[1, nb_channels, max_k_per_channel, world_dims...]

        The first dimension of cells and K is the vmap dimension
    """
    fft_cells = jnp.fft.fftn(cells, axes=axes)[:, :, jnp.newaxis, ...]  # [N, nb_channels, 1, world_dims...]
    fft_out = fft_cells * K
    conv_out = jnp.real(jnp.fft.ifftn(fft_out, axes=axes))  # [N, nb_channels, max_k_per_channel, world_dims...]

    world_shape = cells.shape[2:]
    final_shape = (-1, max_k_per_channel * C) + world_shape
    conv_out_reshaped = conv_out.reshape(final_shape)

    potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, world_dims...]

    return potential


def get_potential(cells: jnp.ndarray, K: jnp.ndarray, padding: jnp.ndarray, true_channels: jnp.ndarray) -> jnp.ndarray:
    """
            Args:
                - cells: jnp.ndarray[N, nb_channels, world_dims...]
                - K: jnp.ndarray[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]

            The first dimension of cells and K is the vmap dimension
        """
    padded_cells = jnp.pad(cells, padding, mode='wrap')
    nb_channels = cells.shape[1]
    # the beauty of feature_group_count for depthwise convolution
    conv_out_reshaped = lax.conv_general_dilated(padded_cells, K, (1, 1), 'VALID', feature_group_count=nb_channels)

    potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, H, W]

    return potential


@functools.partial(jit, static_argnums=(3, 4))
def get_field(
    potential: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    growth_fn_t: Tuple[Callable],
    weighted_fn: Callable
) -> jnp.ndarray:
    """
        Args:
            - potential: jnp.ndarray[N, nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
            - gfn_params: jnp.ndarray[nb_kernels, 2] shape must be kept constant to avoid recompiling
            - kernels_weight_per_channel: jnp.ndarray[nb_channels, nb_kernels]
            - growth_fn_t: Tuple of growth functions
            - weighted_fn: Apply weight function
        Implicit closure args:
            - nb_kernels must be kept contant to avoid recompiling
            - cout_kernels must be of shape [nb_channels]

        Outputs:
            - fields: jnp.ndarray[N=1, nb_channels, world_dims]
    """
    fields = []
    for i in range(gfn_params.shape[0]):
        sub_potential = potential[:, i]
        current_gfn_params = gfn_params[i]
        growth_fn = growth_fn_t[i]

        sub_field = growth_fn(current_gfn_params, sub_potential)

        fields.append(sub_field)
    fields_jnp = jnp.stack(fields, axis=1)  # [N, nb_kernels, world_dims...]

    fields_jnp = weighted_fn(fields_jnp, kernels_weight_per_channel)  # [N, C, H, W]

    return fields_jnp


def weighted_select(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - fields: jnp.ndarray[N, nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
            - weights: jnp.ndarray[nb_channels, nb_kernels] shape must be kept constant to avoid recompiling
                0. values are used to indicate that a given channels does not receive inputs from this kernel

        Outputs:
            - fields: jnp.ndarray[nb_channels, world_dims...]
    """
    N = fields.shape[0]
    nb_kernels = fields.shape[1]
    world_dims = list(fields.shape[2:])
    nb_channels = weights.shape[0]

    fields_swapped = fields.swapaxes(0, 1)  # [nb_kernels, N, world_dims...]
    fields_reshaped = fields_swapped.reshape(nb_kernels, -1)

    out_tmp = jnp.matmul(weights, fields_reshaped)
    out_tmp = out_tmp.reshape([nb_channels, N] + world_dims)
    fields_out = out_tmp.swapaxes(0, 1)  # [N, nb_channels, world_dims...]

    return fields_out


def weighted_select_average(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - fields: jnp.ndarray[N, nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
            - weights: jnp.ndarray[nb_channels, nb_kernels] shape must be kept constant to avoid recompiling
                0. values are used to indicate that a given channels does not receive inputs from this kernel

        Outputs:
            - fields: jnp.ndarray[nb_channels, world_dims...]
    """
    fields_out = weighted_select(fields, weights)
    fields_normalized = fields_out / weights.sum(axis=1)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

    return fields_normalized


def update_cells(cells: jnp.ndarray, field: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - field: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - dt:     jnp.ndarray[N], shape must be kept constant to avoid recompiling
    """
    cells_new = cells + dt * field
    clipped_cells = jnp.clip(cells_new, 0., 1.)

    # Straight-through estimator
    zero = cells_new - lax.stop_gradient(cells_new)
    out = zero + lax.stop_gradient(clipped_cells)

    return out


def update_cells_v2(cells: jnp.ndarray, field: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - field: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - dt:     jnp.ndarray[N], shape must be kept constant to avoid recompiling
    """
    cells_new = jnp.array(cells * (1 - dt) + dt * field)

    return cells_new
