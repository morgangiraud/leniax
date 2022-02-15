"""
.. highlight:: python
"""
import functools

import jax
from jax import vmap, pmap, lax, jit
from jax.tree_util import tree_map
import jax.numpy as jnp
from typing import Callable, Dict, Tuple, Optional

from . import statistics as leniax_stat
from .constant import EPSILON, START_CHECK_STOP


def run(
    rng_key: jax.random.KeyArray,
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Simulate a single configuration

    It uses a python ``for`` loop under the hood.

    Args:
        rng_key: JAX PRNG key.
        cells: Initial cells state ``[N_init=1, nb_channels, world_dims...]``
        K: Stacked Kernels (shape depends on convolution implementation)
        gf_params: Growth function parameters ``[nb_kernels, params_shape...]``
        kernels_weight_per_channel: Kernels weight used in the average function ``[nb_channels, nb_kernels]``
        dt: Update rate ``[1]``
        max_run_iter: Maximum number of simulation iterations
        R: Main kernel Resolution
        update_fn: Function used to compute the new cell state
        compute_stats_fn: Function used to compute the statistics

    Returns:
        A 5-tuple of arrays representing a jax PRNG key, the updated cells state,
        the used potential and used field and statistics
    """

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
        rng_key, new_cells, field, potential = update_fn(rng_key, cells, K, gf_params, kernels_weight_per_channel, 1. / T)

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
        cond, counters['nb_max_volume_step'] = leniax_stat.mass_volume_heuristic(mass_volume, mass_volume_counter)
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

    return rng_key, all_cells_jnp, all_fields_jnp, all_potentials_jnp, stats_dict


@functools.partial(jit, static_argnums=(6, 7, 8, 9))
def run_scan(
    rng_key: jax.random.KeyArray,
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Simulate a single configuration

    This function is jitted, it uses jax.lax.scan function under the hood.
    It can be used to simulate a single configuration with multiple initialization.

    Args:
        rng_key: JAX PRNG key.
        cells0: Initial cells state ``[N_init, nb_channels, world_dims...]``
        K: Stacked Kernels ``[kernel_shape...]``
        gf_params: Growth function parameters ``[nb_kernels, params_shape...]``
        kernels_weight_per_channel: Kernels weight used in the average function ``[nb_channels, nb_kernels]``
        dt: Update rate ``[1]``
        max_run_iter: Maximum number of simulation iterations
        R: Main kernel Resolution
        update_fn: Function used to compute the new cell state
        compute_stats_fn: Function used to compute the statistics

    Returns:
        A 4-tuple of arrays representing the updated cells state, the used potential
        and used field and simulations statistics
    """
    init_carry = _get_init_carry(rng_key, cells0, K, gf_params, kernels_weight_per_channel, T)
    fn: Callable = functools.partial(
        _scan_fn, update_fn=update_fn, compute_stats_fn=compute_stats_fn, keep_intermediary_data=True
    )

    final_carry, ys = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    rng_key = final_carry['fn_params'][0]

    continue_stat = leniax_stat.check_heuristics(ys['stats'])
    ys['stats']['N'] = continue_stat.sum(axis=0)

    return rng_key, ys['cells'], ys['field'], ys['potential'], ys['stats']


@functools.partial(jit, static_argnums=(6, 7, 8, 9))
@functools.partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None), out_axes=0)
def run_scan_mem_optimized(
    rng_key: jax.random.KeyArray,
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jax.random.KeyArray, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Simulate multiple configurations

    This function is jitted, it uses jax.lax.scan function under the hood.
    It can be used to simulate multiple configurations with multiple initialization.

    Args:
        rng_key: JAX PRNG key.
        cells0: Initial cells state ``[N_sols, N_init, nb_channels, world_dims...]``
        K: Stacked Kernels ``[N_sols, kernel_shape...]``
        gf_params: Growth function parameters ``[N_sols, nb_kernels, params_shape...]``
        kernels_weight_per_channel: Kernels weight used in the average function ``[N_sols, nb_channels, nb_kernels]``
        T: Update rate ``[N_sols]``
        max_run_iter: Maximum number of simulation iterations
        R: Main kernel Resolution
        update_fn: Function used to compute the new cell state
        compute_stats_fn: Function used to compute the statistics

    Returns:
        A 3-tuple representing a jax PRNG key, the simulations statistics and final cells states
    """
    init_carry = _get_init_carry(rng_key, cells0, K, gf_params, kernels_weight_per_channel, T)
    fn: Callable = functools.partial(
        _scan_fn, update_fn=update_fn, compute_stats_fn=compute_stats_fn, keep_intermediary_data=False
    )

    final_carry, stats = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    rng_key = final_carry['fn_params'][0]
    final_cells = final_carry['fn_params'][1]

    continue_stat = leniax_stat.check_heuristics(stats)
    stats['N'] = continue_stat.sum(axis=0)

    return rng_key, stats, final_cells


@functools.partial(
    pmap, in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None), out_axes=0, static_broadcasted_argnums=(6, 7, 8, 9)
)
@functools.partial(vmap, in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None), out_axes=0)
def run_scan_mem_optimized_pmap(
    rng_key: jax.random.KeyArray,
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    max_run_iter: int,
    R: float,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jax.random.KeyArray, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Simulate multiple configurations on multiple devices

    This function is jitted, it uses jax.lax.scan function under the hood.
    It can be used to simulate multiple configurations with multiple initialization
    on multiple devices.

    Args:
        rng_key: JAX PRNG key.
        cells0: Initial cells state ``[N_device, N_sols, N_init, nb_channels, world_dims...]``
        K: Stacked Kernels ``[N_device, N_sols, kernel_shape...]``
        gf_params: Growth function parameters ``[N_device, N_sols, nb_kernels, params_shape...]``
        kernels_weight_per_channel: Kernels weight used in the average function ``[N_device, N_sols, nb_channels, nb_kernels]``
        T: Update rate ``[N_device, N_sols]``
        max_run_iter: Maximum number of simulation iterations
        R: Main kernel Resolution
        update_fn: Function used to compute the new cell state
        compute_stats_fn: Function used to compute the statistics

    Returns:
        A 3-tuple representing a jax PRNG key, the simulations statistics and final cells states
    """
    init_carry = _get_init_carry(rng_key, cells0, K, gf_params, kernels_weight_per_channel, T)
    fn: Callable = functools.partial(
        _scan_fn, update_fn=update_fn, compute_stats_fn=compute_stats_fn, keep_intermediary_data=False
    )

    final_carry, stats = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    rng_key = final_carry['fn_params'][0]
    final_cells = final_carry['fn_params'][1]

    continue_stat = leniax_stat.check_heuristics(stats)
    stats['N'] = continue_stat.sum(axis=0)

    return rng_key, stats, final_cells


def _get_init_carry(
    rng_key: jax.random.KeyArray,
    cells0: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    T: jnp.ndarray,
    with_stat: bool = True,
) -> Dict:
    N = cells0.shape[0]
    nb_world_dims = cells0.ndim - 2
    init_carry: Dict = {
        'fn_params': (rng_key, cells0, K, gf_params, kernels_weight_per_channel, T),
    }

    if with_stat is True:
        init_carry['stats_properties'] = {
            'total_shift_idx': jnp.zeros([N, nb_world_dims], dtype=jnp.int32),
            'mass_centroid': jnp.zeros([nb_world_dims, N]),
            'mass_angle': jnp.zeros([N]),
        }

    return init_carry


@functools.partial(jit, static_argnums=(2, 3, 4))
def _scan_fn(
    carry: Dict,
    x: Optional[jnp.ndarray],
    update_fn: Callable,
    compute_stats_fn: Callable,
    keep_intermediary_data: bool = False
) -> Tuple[Dict, Dict]:
    """Update function used in the scan implementation"""

    rng_key, cells, K, gf_params, kernels_weight_per_channel, T = carry['fn_params']
    rng_key, new_cells, field, potential = update_fn(rng_key, cells, K, gf_params, kernels_weight_per_channel, 1. / T)

    stat_props = carry['stats_properties']
    total_shift_idx = stat_props['total_shift_idx']
    mass_centroid = stat_props['mass_centroid']
    mass_angle = stat_props['mass_angle']
    stats, total_shift_idx, mass_centroid, mass_angle = compute_stats_fn(
        cells, field, potential, total_shift_idx, mass_centroid, mass_angle
    )

    new_carry = {
        'fn_params': (rng_key, new_cells, K, gf_params, kernels_weight_per_channel, T),
        'stats_properties': {
            'total_shift_idx': total_shift_idx,
            'mass_centroid': mass_centroid,
            'mass_angle': mass_angle,
        }
    }

    if keep_intermediary_data is True:
        y = {'cells': cells, 'field': field, 'potential': potential, 'stats': stats}
    else:
        y = stats

    return new_carry, y


###
# Differentiable functions
###
def make_pipeline_fn(
    max_iter: int,
    dt: float,
    apply_fn: Callable,
    loss_fn: Callable[[jax.random.KeyArray, Tuple, Tuple], Tuple[jax.random.KeyArray, jnp.ndarray]],
    keep_intermediary_data: bool = False,
    keep_all_timesteps: bool = False,
):
    @jax.jit
    def fn(rng_key, params, variables, state0, targets):
        """Simulate a single configuration

        It uses a python for loop under the hood.
        Compute the error using the provided `loss_fn`::

            def loss_fn(...):
                ...

            run_fn = functools.partial(
                leniax_runner.run_diff,
                nb_steps=nb_steps,
                K_fn=K_fn,
                update_fn=update_fn,
                loss_fn=loss_fn
            )
            run_fn_value_and_grads = jax.value_and_grad(run_fn, argnums=(1))

        Args:
            cells: Initial cells state ``[N_init=1, nb_channels, world_dims...]``
            K_params: Kernel parameters used to generate the final kernels
            gf_params: Growth function parameters ``[nb_kernels, params_shape...]``
            kernels_weight_per_channel: Kernels weight used in the average function ``[nb_channels, nb_kernels]``
            dt: Update rate ``[N]``
            target: 3-tuple of optionnal true cells, potential and field values
            nb_steps: Maximum number of simulation iterations
            K_fn: Function used to compute the kernels
            update_fn: Function used to compute the new cell state
            loss_fn: Function used to compute the error

        Returns:
            The error value.
        """
        update_fn = functools.partial(apply_fn, {'params': params, **variables}, dt=dt)
        fn: Callable = functools.partial(
            _scan_fn_without_stat,
            update_fn=update_fn,
            keep_intermediary_data=keep_intermediary_data,
            keep_all_timesteps=keep_all_timesteps,
        )

        # The number of iterations is controlled by the number of PRNG keys
        rng_key, *subkeys = jax.random.split(rng_key, max_iter + 1)
        final_state_carry, preds = lax.scan(fn, state0, jnp.array(subkeys))

        if keep_all_timesteps is True:
            loss, rng_key, *rest = loss_fn(rng_key, preds, targets)
        else:
            loss, rng_key, *rest = loss_fn(rng_key, (final_state_carry, ), targets)

        return loss, (rng_key, rest)

    return fn


@functools.partial(jit, static_argnums=(2, 3, 4))
def _scan_fn_without_stat(
    state_carry: jnp.ndarray,
    rng_key: jnp.ndarray,
    update_fn: Callable,
    keep_intermediary_data: bool = False,
    keep_all_timesteps: bool = False,
) -> Tuple[Dict, Optional[Tuple]]:
    """Update function used in the scan implementation"""

    _, new_state_carry, field, potential = update_fn(rng_key, state_carry)

    if keep_all_timesteps is True:
        if keep_intermediary_data is True:
            y = (new_state_carry, field, potential)
        else:
            y = (new_state_carry, None, None)
    else:
        y = None

    return new_state_carry, y


def make_gradient_fn(pipeline_fn: Callable, normalize: bool = True):
    @jax.jit
    def fn(rng_key, params, variables, state0, targets):
        grads_fn = jax.value_and_grad(pipeline_fn, argnums=(1), has_aux=True)
        (loss, aux), grads = grads_fn(rng_key, params, variables, state0, targets)
        rng_key = aux[0]
        rest = aux[1]

        if normalize is True:

            def normalize_fn(g):
                return g / (jnp.linalg.norm(g) + 1e-8)

            grads = tree_map(normalize_fn, grads)

        return (rng_key, loss, rest), grads

    return fn
