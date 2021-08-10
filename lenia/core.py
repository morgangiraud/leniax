import math
import jax
from jax import vmap, lax, jit
import jax.numpy as jnp
# import numpy as np
from typing import Callable, List, Dict, Tuple

from . import utils
from . import statistics as lenia_stat
from .kernels import get_kernels_and_mapping, KernelMapping
from .growth_functions import growth_fns
from .constant import EPSILON, START_CHECK_STOP


def init(config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray, KernelMapping]:
    nb_dims = config['world_params']['nb_dims']
    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    assert len(world_size) == nb_dims
    assert nb_channels > 0

    cells = config['run_params']['cells']
    if type(cells) is str:
        cells = utils.decompress_array(cells, nb_dims + 1)  # we add the channel dim
    cells = init_cells(world_size, nb_channels, [cells])

    kernels_params = config['kernels_params']['k']
    R = config['world_params']['R']
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)

    return cells, K, mapping


def init_cells(world_size: List[int], nb_channels: int, other_cells: List[jnp.ndarray] = None) -> jnp.ndarray:
    """
        Outputs:
            - cells: jnp.ndarray[N=1, C, world_dims...]
    """
    world_shape = [nb_channels] + world_size  # [C, H, W]
    cells = jnp.zeros(world_shape)
    if other_cells is not None:
        for c in other_cells:
            cells = utils.merge_cells(cells, c)

    cells = cells[jnp.newaxis, ...]

    return cells


def run(
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    max_run_iter: int,
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
    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    init_mass = cells.sum()

    previous_mass = init_mass
    previous_sign = 0.
    nb_monotone_step = 0
    nb_slow_mass_step = 0
    nb_too_much_percent_step = 0
    should_continue = True
    for current_iter in range(max_run_iter):
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel)

        stats, total_shift_idx = compute_stats_fn(cells, field, potential, total_shift_idx)

        cells = new_cells
        all_cells.append(cells)
        all_fields.append(field)
        all_potentials.append(potential)
        all_stats.append(stats)

        # Heuristics
        # Looking for non-static species
        mass_speed = stats['mass_speed']
        should_continue_cond, nb_slow_mass_step = lenia_stat.max_mass_speed_heuristic_seq(mass_speed, nb_slow_mass_step)
        should_continue = should_continue and should_continue_cond

        # Checking if it is a kind of cosmic soup
        # growth = stats['growth']
        # should_continue_cond = lenia_stat.max_growth_heuristics_sec(growth)
        # should_continue = should_continue and should_continue_cond

        current_mass = stats['mass']
        percent_activated = stats['percent_activated']

        # Heuristics
        # Stops when there is no more mass in the system
        if current_mass < EPSILON:
            # print("current_mass < should_stop", current_mass)
            should_continue = False

        # Stops when there is too much mass in the system
        if current_mass > 3 * init_mass:  # heuristic to detect explosive behaviour
            # print("current_mass > should_stop", current_mass)
            should_continue = False

        # Stops when mass evolve monotically over a quite long period
        # Signal for long-term explosive/implosivve behaviour
        current_sign = math.copysign(1, previous_mass - current_mass)
        if current_sign == previous_sign:
            nb_monotone_step += 1
            if nb_monotone_step > 80:
                # print("current_sign > should_stop")
                should_continue = False
        else:
            nb_monotone_step = 0
        previous_sign = current_sign
        previous_mass = current_mass

        should_continue_cond, nb_too_much_percent_step = lenia_stat.percent_activated_heuristic_seq(
            percent_activated, nb_too_much_percent_step
        )
        should_continue = should_continue and should_continue_cond

        # We avoid dismissing a simulation during the init period
        if current_iter >= START_CHECK_STOP and not should_continue:
            break

    # To keep the same number of elements per array
    all_cells.pop()

    all_cells_jnp = jnp.array(all_cells)
    all_fields_jnp = jnp.array(all_fields)
    all_potentials_jnp = jnp.array(all_potentials)

    stats_dict = lenia_stat.stats_list_to_dict(all_stats)
    stats_dict['N'] = jnp.array(current_iter)

    return all_cells_jnp, all_fields_jnp, all_potentials_jnp, stats_dict


@jax.partial(jit, static_argnums=(4, 5, 6))
def run_scan(
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    max_run_iter: int,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2

    def fn(carry, x):
        cells, K, gfn_params, kernels_weight_per_channel = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        stats, total_shift_idx = compute_stats_fn(cells, field, potential, total_shift_idx)

        y = {'cells': cells, 'field': field, 'potential': potential, 'stats': stats}

        cells = new_cells
        new_carry = {
            'fn_params': (cells, K, gfn_params, kernels_weight_per_channel),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
            }
        }

        return new_carry, y

    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    init_carry = {
        'fn_params': (cells0, K, gfn_params, kernels_weight_per_channel),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
        }
    }

    _, ys = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    continue_stat = lenia_stat.check_heuristics(ys['stats'])
    ys['stats']['N'] = continue_stat.sum(axis=0)

    return ys['cells'], ys['field'], ys['potential'], ys['stats']


@jax.partial(jit, static_argnums=(4, 5, 6))
@jax.partial(vmap, in_axes=(0, 0, 0, 0, None, None, None), out_axes=0)
def run_scan_mem_optimized(
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    max_run_iter: int,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> int:
    """
        Args:
            - cells0: jnp.ndarray[N, nb_channels, world_dims...]
            - K: jnp.ndarray[max_k_per_channel, nb_kernels * nb_channels, K_dims...]
            - gfn_params: jnp.ndarray[N, nb_kernels, 2]
            - kernels_weight_per_channel: jnp.ndarray[N, nb_channels, nb_kernels]
    """
    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2

    def fn(carry, x):
        cells, K, gfn_params, kernels_weight_per_channel = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        stats, total_shift_idx = compute_stats_fn(cells, field, potential, total_shift_idx)

        cells = new_cells
        new_carry = {
            'fn_params': (cells, K, gfn_params, kernels_weight_per_channel),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
            }
        }

        return new_carry, stats

    total_shift_idx = jnp.zeros([N, nb_world_dims], dtype=jnp.int32)
    init_carry = {
        'fn_params': (cells0, K, gfn_params, kernels_weight_per_channel),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
        }
    }

    _, stats = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    continue_stat = lenia_stat.check_heuristics(stats)
    N = continue_stat.sum(axis=0)

    return N


def build_update_fn(
    world_params: Dict, K_shape: Tuple[int, ...], mapping: KernelMapping, update_fn_version: str = 'v1'
) -> Callable:
    T = world_params['T']
    dt = 1 / T

    get_potential_fn = build_get_potential_fn(K_shape, mapping.true_channels)
    get_field_fn = build_get_field_fn(mapping.cin_growth_fns)
    if update_fn_version == 'v1':
        update_fn = update_cells
    elif update_fn_version == 'v2':
        update_fn = update_cells_v2
    else:
        raise ValueError(f"version {update_fn_version} does not exist")

    @jit
    def update(cells: jnp.ndarray, K: jnp.ndarray, gfn_params: jnp.ndarray,
               kernels_weight_per_channel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
            Args:
                - cells:                        jnp.ndarray[N, nb_channels, world_dims...]
                - K:                            jnp.ndarray[K_o=nb_channels * max_k_per_kernel, K_i=1, kernel_dims...]
                - gfn_params:                   jnp.ndarray[nb_kernels, 2]
                - kernels_weight_per_channel:   jnp.ndarray[nb_channels, nb_kernels]

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


def build_get_potential_fn(kernel_shape: Tuple[int, ...], true_channels: jnp.ndarray) -> Callable:
    # First 2 dimensions are for fake batch dim and nb_channels
    padding = jnp.array([[0, 0]] * 2 + [[dim // 2, dim // 2] for dim in kernel_shape[2:]])

    def get_potential(cells: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        """
            Args:
                - cells: jnp.ndarray[N, nb_channels, world_dims...]
                - K: jnp.ndarray[K_o=nb_channels * max_k_per_kernel, K_i=1, kernel_dims...]

            The first dimension of cells and K is the vmap dimension
        """
        padded_cells = jnp.pad(cells, padding, mode='wrap')
        nb_channels = cells.shape[1]
        # the beauty of feature_group_count for depthwise convolution
        conv_out_reshaped = lax.conv_general_dilated(padded_cells, K, (1, 1), 'VALID', feature_group_count=nb_channels)

        potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, H, W]

        return potential

    return get_potential


def build_get_field_fn(cin_growth_fns: Dict[str, List]) -> Callable:
    growth_fn_l = []
    for i in range(len(cin_growth_fns['gf_id'])):
        gf_id = cin_growth_fns['gf_id'][i]
        growth_fn = growth_fns[gf_id]
        growth_fn_l.append(growth_fn)
    nb_kernels = len(cin_growth_fns['gf_id'])

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
        for i in range(nb_kernels):
            sub_potential = potential[:, i]
            current_gfn_params = gfn_params[i]
            growth_fn = growth_fn_l[i]

            sub_field = growth_fn(sub_potential, current_gfn_params[0], current_gfn_params[1])

            fields.append(sub_field)
        fields_jnp = jnp.stack(fields, axis=1)  # [N, nb_kernels, world_dims...]

        fields_jnp = weighted_select_average(fields_jnp, kernels_weight_per_channel)  # [N, C, H, W]

        return fields_jnp

    return get_field


def weighted_select_average(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
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
    out_tmp = out_tmp.swapaxes(0, 1)  # [N, nb_channels, world_dims...]
    fields_normalized = out_tmp / weights.sum(axis=1)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

    return fields_normalized


def update_cells(cells: jnp.ndarray, field: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray, shape must be kept constant to avoid recompiling
            - field: jnp.ndarray, shape must be kept constant to avoid recompiling
            - float: can change without recompiling
    """
    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0., 1.)

    return cells_new


def update_cells_v2(cells: jnp.ndarray, field: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray, shape must be kept constant to avoid recompiling
            - field: jnp.ndarray, shape must be kept constant to avoid recompiling
            - float: can change without recompiling
    """
    rescaled_field = (field + 1) / 2
    cells_new = jnp.array(cells * (1 - dt) + dt * rescaled_field)

    return cells_new
