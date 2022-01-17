import functools
from jax import vmap, pmap, lax, jit
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, Optional

from . import statistics as leniax_stat
from .constant import EPSILON, START_CHECK_STOP


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

        new_carry = {
            'fn_params': (new_cells, K, gfn_params, kernels_weight_per_channel, T),
            'stats_properties': {
                'total_shift_idx': total_shift_idx,
                'mass_centroid': mass_centroid,
                'mass_angle': mass_angle,
            }
        }

        y = {'cells': cells, 'field': field, 'potential': potential, 'stats': stats}

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


###
# Differentiable functions
###
def run_diff(
    cells: jnp.ndarray,
    K_params: jnp.ndarray,
    gfn_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    target: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]],
    max_run_iter: int,
    K_fn: Callable,
    update_fn: Callable,
    error_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    K = K_fn(K_params)
    dt = jnp.array(1. / T, dtype=jnp.float32)

    for _ in range(max_run_iter):
        new_cells, field, potential = update_fn(cells, K, gfn_params, kernels_weight_per_channel, dt)

    preds = (new_cells, field, potential)

    error = error_fn(preds, target)

    return error
