import os
import pickle
import jax
from jax import vmap, lax, jit
import jax.numpy as jnp
# import numpy as np
import scipy
from typing import Callable, List, Dict, Tuple

from . import utils
from . import statistics as leniax_stat
from .kernels import get_kernels_and_mapping, KernelMapping
from .growth_functions import growth_fns
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
        cells = jnp.array([scipy.ndimage.zoom(cells[i], scale, order=2) for i in range(nb_channels)], dtype=jnp.float32)
        config['world_params']['R'] *= scale

    # assert cells.shape[1] * 2.2 < config['render_params']['world_size'][0]
    # assert cells.shape[2] * 2.2 < config['render_params']['world_size'][1]
    # on = round(max(cells.shape[1] * 1.2, cells.shape[2] * 1.2) / 2.)
    # init_cells = create_init_cells(world_size, nb_channels, [
    #     jnp.rot90(cells, k=2, axes=(1, 2)),
    #     jnp.rot90(cells, k=0, axes=(1, 2)),
    #     jnp.rot90(cells, k=-1, axes=(1, 2)),
    #     jnp.rot90(cells, k=1, axes=(1, 2))
    # ], [
    #     [0, on, on],
    #     [0, -on, -on],
    #     [0, -on, on],
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
        if cells == 'last_frame.p':
            with open(os.path.join(config['main_path'], 'last_frame.p'), 'rb') as f:
                cells = jnp.array(pickle.load(f), dtype=jnp.float32)
        else:
            # Backward compatibility
            cells = utils.decompress_array(cells, nb_dims + 1)  # we add the channel dim
    elif type(cells) is list:
        cells = jnp.array(cells, dtype=jnp.float32)

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
    if other_cells is not None:
        if len(offsets) == len(other_cells):
            for c, offset in zip(other_cells, offsets):
                cells = utils.merge_cells(cells, c, offset)
        else:
            for c in other_cells:
                cells = utils.merge_cells(cells, c)

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


@jax.partial(jit, static_argnums=(5, 6, 7, 8))
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


@jax.partial(jit, static_argnums=(5, 6, 7, 8))
@jax.partial(vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None), out_axes=0)
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
            - cells0: jnp.ndarray[N, nb_channels, world_dims...]
            - K: jnp.ndarray[max_k_per_channel, nb_kernels * nb_channels, K_dims...]
            - gfn_params: jnp.ndarray[N, nb_kernels, 2]
            - kernels_weight_per_channel: jnp.ndarray[N, nb_channels, nb_kernels]
            - T: jnp.ndarray[N]
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
        update_fn = update_cells
    elif update_fn_version == 'v2':
        update_fn = update_cells_v2
    else:
        raise ValueError(f"version {update_fn_version} does not exist")

    @jit
    def update(
        cells: jnp.ndarray,
        K: jnp.ndarray,
        gfn_params: jnp.ndarray,
        kernels_weight_per_channel: jnp.ndarray,
        dt: jnp.ndarray
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

    return update


def build_get_potential_fn(kernel_shape: Tuple[int, ...], true_channels: jnp.ndarray, fft: bool = True) -> Callable:
    # First 2 dimensions are for fake batch dim and nb_channels
    if fft is True:
        max_k_per_channel = kernel_shape[1]
        C = kernel_shape[2]

        def get_potential(cells: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
            """
                Args:
                    - cells: jnp.ndarray[N, nb_channels, world_dims...]
                    - K: jnp.ndarray[1, nb_channels, max_k_per_channel, world_dims...]

                The first dimension of cells and K is the vmap dimension
            """
            fft_cells = jnp.fft.fft2(cells, axes=(-2, -1))[:, :, jnp.newaxis, ...]  # [N, nb_channels, 1, world_dims...]
            fft_out = fft_cells * K
            conv_out = jnp.real(
                jnp.fft.ifft2(fft_out, axes=(-2, -1))
            )  # [N, nb_channels, max_k_per_channel, world_dims...]

            world_shape = cells.shape[2:]
            final_shape = (-1, max_k_per_channel * C) + world_shape
            conv_out_reshaped = conv_out.reshape(final_shape)

            potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, world_dims...]

            return potential
    else:
        padding = jnp.array([[0, 0]] * 2 + [[dim // 2, dim // 2] for dim in kernel_shape[2:]])

        def get_potential(cells: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
            """
                Args:
                    - cells: jnp.ndarray[N, nb_channels, world_dims...]
                    - K: jnp.ndarray[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]

                The first dimension of cells and K is the vmap dimension
            """
            padded_cells = jnp.pad(cells, padding, mode='wrap')
            nb_channels = cells.shape[1]
            # the beauty of feature_group_count for depthwise convolution
            conv_out_reshaped = lax.conv_general_dilated(
                padded_cells, K, (1, 1), 'VALID', feature_group_count=nb_channels
            )

            potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, H, W]

            return potential

    return get_potential


def build_get_field_fn(cin_growth_fns: List[List[int]], average: bool = True) -> Callable:
    growth_fn_l = []
    for growth_fns_per_channel in cin_growth_fns:
        for gf_id in growth_fns_per_channel:
            growth_fn = growth_fns[gf_id]
            growth_fn_l.append(growth_fn)

    if average:
        weighted_fn = weighted_select_average
    else:
        weighted_fn = weighted_select

    def get_field(
        potential: jnp.ndarray, gfn_params: jnp.ndarray, kernels_weight_per_channel: jnp.ndarray
    ) -> jnp.ndarray:
        """
            Args:
                - potential: jnp.ndarray[N, nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
                - gfn_params: jnp.ndarray[nb_kernels, 2] shape must be kept constant to avoid recompiling
                - kernels_weight_per_channel: jnp.ndarray[nb_channels, nb_kernels]
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
            growth_fn = growth_fn_l[i]

            sub_field = growth_fn(sub_potential, current_gfn_params[0], current_gfn_params[1])

            fields.append(sub_field)
        fields_jnp = jnp.stack(fields, axis=1)  # [N, nb_kernels, world_dims...]

        fields_jnp = weighted_fn(fields_jnp, kernels_weight_per_channel)  # [N, C, H, W]

        return fields_jnp

    return get_field


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

    return clipped_cells


def update_cells_v2(cells: jnp.ndarray, field: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - field: jnp.ndarray[N, nb_channels, world_dims...], shape must be kept constant to avoid recompiling
            - dt:     jnp.ndarray[N], shape must be kept constant to avoid recompiling
    """
    cells_new = jnp.array(cells * (1 - dt) + dt * field)

    return cells_new
