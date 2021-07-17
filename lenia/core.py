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


def run(
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    max_run_iter: int,
    update_fn: Callable,
    compute_stats_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List]:
    assert max_run_iter > 0, f"max_run_iter must be positive, value given: {max_run_iter}"

    # cells shape: [C, N=1, c=1, dims...]
    world_shape = jnp.array(cells.shape[3:])
    nb_dims = len(world_shape)
    axes = tuple(range(-nb_dims, 0, 1))

    all_cells = [cells]
    all_fields = []
    all_potentials = []
    all_stats = []
    total_shift_idx = jnp.zeros([nb_dims], dtype=jnp.int32)
    init_mass = cells.sum()

    previous_mass = init_mass
    previous_sign = 0.
    nb_monotone_step = 0
    nb_slow_mass_step = 0
    nb_too_much_percent_step = 0
    should_continue = True
    for current_iter in range(max_run_iter):
        new_cells, field, potential = update_fn(cells, K, gfn_params)

        # To compute stats, the world (cells, field, potential) has to be centered and taken from the same timestep
        centered_cells, centered_field, centered_potential = utils.center_world(
            new_cells, field, potential, total_shift_idx, axes
        )
        stats, shift_idx = compute_stats_fn(centered_cells, centered_field, centered_potential)
        total_shift_idx = (total_shift_idx + shift_idx) % world_shape
        all_stats.append(stats)

        cells = new_cells
        all_cells.append(cells)
        all_fields.append(field)
        all_potentials.append(potential)

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

    return all_cells_jnp, all_fields_jnp, all_potentials_jnp, all_stats


@jax.partial(jit, static_argnums=(3, 4, 5))
def run_scan(
    cells: jnp.ndarray,
    K: jnp.ndarray,
    gfn_params: jnp.ndarray,
    max_run_iter: int,
    update_fn: Callable,
    compute_stats_fn: Callable
):
    world_shape = jnp.array(cells.shape[3:])
    nb_dims = len(world_shape)
    axes = tuple(range(-nb_dims, 0, 1))

    def fn(carry, x):
        cells, K, gfn_params = carry['fn_params']
        new_cells, field, potential = update_fn(cells, K, gfn_params)

        stat_props = carry['stats_properties']
        total_shift_idx = stat_props['total_shift_idx']
        # To compute stats, the world (cells, field, potential) has to be centered and taken from the same timestep
        centered_cells, centered_field, centered_potential = utils.center_world(
            new_cells, field, potential, total_shift_idx, axes
        )
        stats, shift_idx = compute_stats_fn(centered_cells, centered_field, centered_potential)
        total_shift_idx = (total_shift_idx + shift_idx) % world_shape

        y = {'cells': cells, 'field': field, 'potential': potential, 'stats': stats}

        cells = new_cells
        new_carry = {
            'fn_params': (cells, K, gfn_params), 'stats_properties': {
                'total_shift_idx': total_shift_idx,
            }
        }

        return new_carry, y

    total_shift_idx = jnp.zeros([nb_dims], dtype=jnp.int32)
    init_carry = {
        'fn_params': (cells, K, gfn_params), 'stats_properties': {
            'total_shift_idx': total_shift_idx,
        }
    }

    final_carry, ys = lax.scan(fn, init_carry, None, length=max_run_iter, unroll=1)

    return ys['cells'], ys['field'], ys['potential'], ys['stats']


def build_update_fn(world_params: Dict, K_shape: Tuple[int, ...], mapping: KernelMapping) -> Callable:
    T = world_params['T']
    dt = 1 / T

    get_potential_fn = build_get_potential_fn(K_shape, mapping.true_channels)
    get_field_fn = build_get_field_fn(mapping)

    @jit
    def update(cells: jnp.ndarray, K: jnp.ndarray,
               gfn_params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        potential = get_potential_fn(cells, K)
        field = get_field_fn(potential, gfn_params)
        cells = update_cells(cells, field, dt)

        return cells, field, potential

    return update


def build_get_potential_fn(kernel_shape: Tuple[int, ...], true_channels: List[bool]) -> Callable:
    true_channels_jnp = jnp.array(true_channels)
    nb_channels = kernel_shape[0]
    depth = kernel_shape[1]
    # First 3 dimensions are for nb_channels, fake batch dim and fake channel dim
    padding = jnp.array([[0, 0]] * 3 + [[dim // 2, dim // 2] for dim in kernel_shape[3:]])

    vconv = vmap(lax.conv, (0, 0, None, None), 0)

    @jit
    def get_potential(cells: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        """
            Args:
                - cells: jnp.ndarray[C, N=1, c=1, world_dims...]
                - K: jnp.ndarray[C, K_o, K_i=1, kernel_dims...]

            The first dimension of cells and K is the vmap dimension
        """

        padded_cells = jnp.pad(cells, padding, mode='wrap')
        conv_out = vconv(padded_cells, K, (1, 1), 'VALID')  # [C, N=1, c=O, H, W]

        conv_out_reshaped = conv_out.reshape(nb_channels * depth, *cells.shape[3:])
        potential = conv_out_reshaped[true_channels_jnp]  # [nb_kernels, H, W]

        return potential

    return get_potential


def build_get_field_fn(mapping: KernelMapping) -> Callable:
    kernels_weight_per_channel = mapping.kernels_weight_per_channel
    cin_growth_fns = mapping.cin_growth_fns

    growth_fn_l = []
    for i in range(len(cin_growth_fns['gf_id'])):
        gf_id = cin_growth_fns['gf_id'][i]
        growth_fn = growth_fns[gf_id]
        growth_fn_l.append(growth_fn)

    @jit
    def get_field(potential: jnp.ndarray, gfn_params: jnp.ndarray) -> jnp.ndarray:
        """
            Args:
                - potential: jnp.ndarray[nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
                - fn_params: jnp.ndarray[nb_kernels, 2] shape must be kept constant to avoid recompiling

            Implicit closure args:
                - nb_kernels must be kept contant to avoid recompiling
                - cout_kernels must be of shape [nb_channels]
        """
        fields = []
        nb_kernels = len(cin_growth_fns['gf_id'])
        for i in range(nb_kernels):
            sub_potential = potential[i]
            current_gfn_params = gfn_params[i]
            growth_fn = growth_fn_l[i]

            sub_field = growth_fn(sub_potential, current_gfn_params[0], current_gfn_params[1])

            fields.append(sub_field)
        fields_jnp = jnp.stack(fields)  # [nb_kernels, world_dims...]

        fields_jnp = weighted_select_average(fields_jnp, kernels_weight_per_channel)  # [C, H, W]
        fields_jnp = fields_jnp[:, jnp.newaxis, jnp.newaxis]  # [C, N=1, c=1, H, W]

        return fields_jnp

    return get_field


@jit
@jax.partial(vmap, in_axes=(None, 0), out_axes=0)
def weighted_select_average(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
        Args:
            - fields: jnp.ndarray[nb_kernels, world_dims...] shape must be kept constant to avoid recompiling
            - weights: jnp.ndarray[nb_channels, nb_kernels] shape must be kept constant to avoid recompiling
                0. values are used to indicate that a given channels does not receive inputs from this kernel

        Outputs:
            - fields: jnp.ndarray[nb_channels, world_dims...]

        ! This function must be vmap-ed to get the dimensions right !
    """
    out_field = jnp.average(fields, axis=0, weights=weights)

    return out_field


@jit
def update_cells(cells: jnp.ndarray, field: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
        Args:
            - cells: jnp.ndarray, shape must be kept constant to avoid recompiling
            - field: jnp.ndarray, shape must be kept constant to avoid recompiling
            - float: can change without recompiling
    """
    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0., 1.)
    # smoothstep(x)
    # cells_new = 3 * cells_new ** 2 - 2 * cells_new ** 3

    return cells_new
