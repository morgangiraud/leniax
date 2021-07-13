import math
from functools import partial
import jax
from jax import vmap, lax, jit
import jax.numpy as jnp
# import numpy as np
from typing import Callable, List, Dict, Tuple, Optional

from . import utils
from . import initializations
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
        cells = utils.decompress_array(cells, nb_dims + 1)
    cells = init_cells(world_size, nb_channels, [cells])

    kernels_params = config['kernels_params']['k']
    R = config['world_params']['R']
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)

    return cells, K, mapping


def init_cells(
    world_size: List[int], nb_channels: int, other_cells: List[jnp.ndarray] = None, nb_init: int = 1
) -> jnp.ndarray:
    if nb_init == 1:
        world_shape = [nb_channels] + world_size  # [C, H, W]
    else:
        world_shape = [nb_init, nb_channels] + world_size  # [C, H, W]
    cells = jnp.zeros(world_shape)
    if other_cells is not None:
        for c in other_cells:
            cells = utils.merge_cells(cells, c)

    # 2D convolutions needs an input with shape [N, C, H, W]
    # And we are going to map over first dimension with vmap so we need th following shape
    # [C, N=1, c=1, H, W]
    if nb_init == 1:
        cells = cells[:, jnp.newaxis, jnp.newaxis]
    else:
        # In this case we also vmap on the number on init: [N_init, C, N=1, c=1, H, W]
        cells = cells[:, :, jnp.newaxis, jnp.newaxis]

    return cells


def run(cells: jnp.ndarray,
        max_run_iter: int,
        update_fn: Callable,
        compute_stats_fn: Optional[Callable] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List]:
    assert max_run_iter > 0, f"max_run_iter must be positive, value given: {max_run_iter}"

    # cells shape: [C, N=1, c=1, dims...]
    nb_dims = len(cells.shape) - 3
    axes = tuple(range(-nb_dims, 0, 1))

    all_cells = [cells]
    all_fields = []
    all_potentials = []
    all_stats = []
    total_shift_idx = 0
    init_mass = cells.sum()

    previous_mass = init_mass
    previous_sign = 0.
    nb_monotone_step = 0
    nb_slow_mass_step = 0
    nb_too_much_percent_step = 0
    should_continue = True
    for current_iter in range(max_run_iter):
        new_cells, field, potential = update_fn(cells)

        if compute_stats_fn:
            # To compute stats, the world (cells, field, potential) has to be centered and taken from the same timestep
            stats, shift_idx = compute_stats_fn(cells, field, potential)
            all_stats.append(stats)

            # centering
            total_shift_idx += shift_idx
            cells, field, potential = utils.center_world(new_cells, field, potential, shift_idx, axes)

            # To visualize, it's nicer to keep the world uncentered
            all_cells.append(jnp.roll(cells, total_shift_idx, axes))
            all_fields.append(jnp.roll(field, total_shift_idx, axes))
            all_potentials.append(jnp.roll(potential, total_shift_idx, axes))

            # Heuristics
            # Looking for non-static species
            mass_speed = stats['mass_speed']
            should_continue_cond, nb_slow_mass_step = lenia_stat.max_mass_speed_heuristic_seq(
                mass_speed, nb_slow_mass_step
            )
            should_continue = should_continue and should_continue_cond

            # Checking if it is a kind of cosmic soup
            # growth = stats['growth']
            # should_continue_cond = lenia_stat.max_growth_heuristics_sec(growth)
            # should_continue = should_continue and should_continue_cond

            current_mass = stats['mass']
            percent_activated = stats['percent_activated']
        else:
            cells = new_cells
            all_cells.append(cells)
            all_fields.append(field)
            all_potentials.append(potential)

            current_mass = cells.sum()
            percent_activated = (cells > EPSILON).sum()

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

    return all_cells_jnp, all_fields_jnp, all_potentials_jnp, all_stats


# @jax.partial(jit, static_argnums=(1, 2, 3))
# @jax.partial(vmap, in_axes=(0, None, None, None), out_axes=(0, 0, 0, None))
def run_parralel(cells: jnp.ndarray, max_run_iter: int, update_fn: Callable,
                 compute_stats_fn: Callable) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List]:

    # This function should be vmapped
    # cells shape for the vmapped input: [N_init, C, N=1, c=1, dims...]
    # actual cells shape: [C, N=1, c=1, dims...]
    nb_dims = len(cells.shape) - 3
    axes = tuple(range(-nb_dims, 0, 1))

    all_cells = [cells]
    all_fields = []
    all_potentials = []
    all_stats = []
    total_shift_idx = 0

    for _ in range(max_run_iter):
        new_cells, field, potential = update_fn(cells)
        # To compute stats, the world (cells, field, potential) has to be centered and taken from the same timestep
        stats, shift_idx = compute_stats_fn(cells, field, potential)
        all_stats.append(stats)

        # centering
        total_shift_idx += shift_idx
        cells, field, potential = utils.center_world(new_cells, field, potential, shift_idx, axes)

        # To have a nice viz, it's much nicer to NOT have the world centered
        all_cells.append(jnp.roll(cells, total_shift_idx, axes))
        all_fields.append(jnp.roll(field, total_shift_idx, axes))
        all_potentials.append(jnp.roll(potential, total_shift_idx, axes))

    # To keep the same number of elements per array
    all_cells.pop()

    all_cells_jnp = jnp.array(all_cells)
    all_fields_jnp = jnp.array(all_fields)
    all_potentials_jnp = jnp.array(all_potentials)

    return all_cells_jnp, all_fields_jnp, all_potentials_jnp, all_stats


def run_init_search(rng_key: jnp.ndarray, config: Dict, with_stats: bool = True) -> Tuple[jnp.ndarray, List]:
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    update_fn = build_update_fn(world_params, K, mapping)
    compute_stats_fn: Optional[Callable]
    if with_stats:
        compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])
    else:
        compute_stats_fn = None

    rng_key, noises = initializations.random_uniform(rng_key, nb_init_search, nb_channels)

    runs = []
    for i in range(nb_init_search):
        cells_0 = init_cells(world_size, nb_channels, [noises[i]])

        all_cells, _, _, all_stats = run(cells_0, max_run_iter, update_fn, compute_stats_fn)
        nb_iter_done = len(all_cells)

        runs.append({"N": nb_iter_done, "all_cells": all_cells, "all_stats": all_stats})
        if nb_iter_done >= max_run_iter:
            break

    return rng_key, runs


def run_init_search_parralel(rng_key: jnp.ndarray, config: Dict) -> Tuple[jnp.ndarray, List]:
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    update_fn = build_update_fn(world_params, K, mapping)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    rng_key, noises = initializations.random_uniform(rng_key, nb_init_search, nb_channels)
    cells_0 = init_cells(world_size, nb_channels, [noises], nb_init_search)

    # prevent any gradient-related operations from downstream operations, including tracing / graph building
    cells_0 = lax.stop_gradient(cells_0)
    run_parralel_partial = partial(
        run_parralel, max_run_iter=max_run_iter, update_fn=update_fn, compute_stats_fn=compute_stats_fn
    )
    vrun = vmap(run_parralel_partial, 0, 0)

    # v_all_cells: list of N_iter ndarray with shape [N_init, C, N=1, c=1, H, W]
    # v_all_stats: list of N_iter stat object with N_init values
    v_all_cells, _, _, v_all_stats = vrun(cells_0)

    all_should_continue = []
    should_continue = jnp.ones(v_all_cells.shape[0])
    init_mass = v_all_cells[:, 0].sum(axis=tuple(range(1, len(v_all_cells.shape) - 1)))
    mass = init_mass.copy()
    sign = jnp.ones_like(init_mass)
    counters = {
        'nb_slow_mass_step': jnp.zeros_like(init_mass),
        'nb_monotone_step': jnp.zeros_like(init_mass),
        'nb_too_much_percent_step': jnp.zeros_like(init_mass),
    }
    for i in range(len(v_all_stats)):
        stats = v_all_stats[i]
        should_continue, mass, sign = lenia_stat.check_heuristics(
            stats, should_continue, init_mass, mass, sign, counters
        )
        all_should_continue.append(should_continue)
    all_should_continue_jnp = jnp.array(all_should_continue)
    idx = all_should_continue_jnp.sum(axis=0).argmax()
    best = v_all_cells[idx, :, :, 0, 0, ...]

    return rng_key, best


def build_update_fn(world_params: Dict, K: jnp.ndarray, mapping: KernelMapping) -> Callable:
    T = world_params['T']

    get_potential_fn = build_get_potential_fn(K.shape, mapping.true_channels)
    get_field_fn = build_get_field_fn(mapping)

    @jit
    def update(cells: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        potential = get_potential_fn(cells, K)
        field = get_field_fn(potential)
        cells = update_cells(cells, field, T)

        return cells, field, potential

    return update


def build_get_potential_fn(kernel_shape: Tuple[int, ...], true_channels: List[bool]) -> Callable:
    true_channels_jnp = jnp.array(true_channels)
    vconv = vmap(lax.conv, (0, 0, None, None), 0)
    nb_channels = kernel_shape[0]
    depth = kernel_shape[1]
    pad_w = kernel_shape[-1] // 2
    pad_h = kernel_shape[-2] // 2

    @jit
    def get_potential(cells: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        padded_cells = jnp.pad(cells, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='wrap')
        conv_out = vconv(padded_cells, K, (1, 1), 'VALID')  # [C, N=1, c=O, H, W]

        H, W = cells.shape[-2:]
        conv_out_reshaped = conv_out.reshape(nb_channels * depth, H, W)
        potential = conv_out_reshaped[true_channels_jnp]  # [nb_kernels, H, W]

        return potential

    return get_potential


def build_get_field_fn(mapping: KernelMapping) -> Callable:
    cout_kernels = jnp.array(mapping.cout_kernels)
    cout_kernels_h = jnp.array(mapping.cout_kernels_h)

    cin_growth_fns = mapping.cin_growth_fns
    growth_fn_l = []
    for i in range(len(cin_growth_fns['gf_id'])):
        gf_id = cin_growth_fns['gf_id'][i]
        m = cin_growth_fns['m'][i]
        s = cin_growth_fns['s'][i]

        growth_fn = growth_fns[gf_id]
        growth_fn_l.append(partial(growth_fn, m=m, s=s))

    @jit
    def get_field(potential: jnp.ndarray) -> jnp.ndarray:
        fields = []
        for i in range(len(cin_growth_fns['gf_id'])):
            sub_potential = potential[i]
            growth_fn = growth_fn_l[i]

            sub_field = growth_fn(sub_potential)

            fields.append(sub_field)
        fields_jnp = jnp.stack(fields)  # Add a dimension

        fields_jnp = weighted_select_average(fields_jnp, cout_kernels, cout_kernels_h)  # [C, H, W]
        fields_jnp = fields_jnp[:, jnp.newaxis, jnp.newaxis]

        return fields_jnp

    return get_field


@jit
@jax.partial(vmap, in_axes=(None, 0, 0), out_axes=0)
def weighted_select_average(field: jnp.ndarray, channel_list: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    out_fields = field[channel_list]
    out_field = jnp.average(out_fields, axis=0, weights=weights)

    return out_field


@jax.partial(jit, static_argnums=2)
def update_cells(cells: jnp.ndarray, field: jnp.ndarray, T: float) -> jnp.ndarray:
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)
    # smoothstep(x)
    # cells_new = 3 * cells_new ** 2 - 2 * cells_new ** 3

    return cells_new
