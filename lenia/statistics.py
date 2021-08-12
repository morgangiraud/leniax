import jax
import jax.numpy as jnp
from jax import jit
from typing import List, Dict, Callable

from .constant import EPSILON
from .utils import center_world


def build_compute_stats_fn(world_params: Dict, render_params: Dict) -> Callable:
    world_size = render_params['world_size']
    R = world_params['R']
    dt = 1. / world_params['T']

    midpoint = jnp.asarray([size // 2 for size in world_size])[:, jnp.newaxis, jnp.newaxis]
    coords = jnp.indices(world_size)
    centered_coords = coords - midpoint  # [nb_dims, H, W]

    @jit
    def compute_stats(cells, field, potential, previous_total_shift_idx):
        # cells: # [N, C, H, W]
        # field: # [N, C, H, W]
        # potential: # [N, C, H, W]
        non_batch_dims = tuple(range(1, cells.ndim, 1))
        axes = tuple(range(-(cells.ndim - 2), 0, 1))

        # https://en.wikipedia.org/wiki/Image_moment
        # Those statistic works on a centered world
        centered_cells, centered_field, _ = center_world(cells, field, potential, previous_total_shift_idx, axes)
        positive_field = jnp.maximum(centered_field, 0)

        m_00 = centered_cells.sum(axis=non_batch_dims)
        g_00 = positive_field.sum(axis=non_batch_dims)

        mass = m_00 / R**2
        mass_volume = (centered_cells > EPSILON).sum(axis=non_batch_dims) / R**2
        mass_density = mass / mass_volume

        growth = g_00 / R**2
        growth_volume = (positive_field > EPSILON).sum(axis=non_batch_dims) / R**2
        growth_density = growth / growth_volume

        AX = [centered_cells * x for x in centered_coords]  # [nb_world_dims, N, C, world_dims...]
        MX = [ax.sum(axis=non_batch_dims) for ax in AX]  # [nb_world_dims, N]
        mass_centroid = jnp.array(MX) / (m_00 + EPSILON)
        # This function returns "mass centered" cells so the previous mass_centroid is at origin
        dm = jnp.linalg.norm(mass_centroid, axis=0)
        mass_speed = dm / R / dt  # [N]
        mass_angle_speed = jnp.degrees(jnp.arctan2(mass_centroid[1], mass_centroid[0])) / dt  # [N]

        GX = [(positive_field * x).sum(axis=non_batch_dims) for x in centered_coords]
        growth_centroid = jnp.array(GX) / (g_00 + EPSILON)
        mass_growth_dist = jnp.linalg.norm(growth_centroid - mass_centroid, axis=0) / R

        MX2 = [(ax * x).sum(axis=non_batch_dims) for ax, x in zip(AX, centered_coords)]
        MuX2 = [mx2 - mc * mx for mc, mx, mx2 in zip(mass_centroid, MX, MX2)]
        inertia = jnp.array(MuX2).sum(axis=0) / R**2

        shift_idx = mass_centroid.astype(int).T
        world_shape = jnp.array(cells.shape[2:])
        total_shift_idx = (previous_total_shift_idx + shift_idx) % world_shape

        stats = {
            'mass': mass,
            'mass_volume': mass_volume,
            'mass_density': mass_density,
            'growth': growth,
            'growth_volume': growth_volume,
            'growth_density': growth_density,
            'mass_speed': mass_speed,
            'mass_angle_speed': mass_angle_speed,
            'mass_growth_dist': mass_growth_dist,
            'inertia': inertia,
        }

        return (stats, total_shift_idx)

    return compute_stats


###
# Heuristics
###
@jax.partial(jit, static_argnums=(1, 2))
def check_heuristics(stats: Dict[str, jnp.ndarray], R: float, dt: float):
    def fn(carry, stat_t):
        should_continue = carry['should_continue']
        init_mass = carry['init_mass']
        previous_mass = carry['previous_mass']
        previous_sign = carry['previous_sign']
        counters = carry['counters']

        mass = stat_t['mass']
        cond = min_mass_heuristic(EPSILON, mass)
        should_continue_cond = cond
        cond = max_mass_heuristic(init_mass, mass)
        should_continue_cond *= cond

        sign = jnp.sign(mass - previous_mass)
        monotone_counter = counters['nb_monotone_step']
        cond, counters['nb_monotone_step'] = monotonic_heuristic(sign, previous_sign, monotone_counter)
        should_continue_cond *= cond

        # growth = stat_t['growth']
        # cond = max_growth_heuristic(growth, R)
        # should_continue_cond *= cond

        mass_speed = stat_t['mass_speed']
        mass_speed_counter = counters['nb_slow_mass_step']
        cond, counters['nb_slow_mass_step'] = min_mass_speed_heuristic(mass_speed, mass_speed_counter, R, dt)
        should_continue_cond *= cond

        mass_volume = stat_t['mass_volume']
        mass_volume_counter = counters['nb_max_volume_step']
        cond, counters['nb_max_volume_step'] = mass_volume_heuristic(mass_volume, mass_volume_counter, R)
        should_continue_cond *= cond

        should_continue *= should_continue_cond
        new_carry = {
            'should_continue': should_continue,
            'init_mass': init_mass,
            'previous_mass': mass,
            'previous_sign': sign,
            'counters': counters,
        }

        return new_carry, should_continue

    N = stats['mass'].shape[1]
    init_carry = {
        'should_continue': jnp.ones(N),
        'init_mass': stats['mass'][0],
        'previous_mass': stats['mass'][0],
        'previous_sign': jnp.zeros(N),
        'counters': init_counters(N),
    }
    _, continue_stat = jax.lax.scan(fn, init_carry, stats, unroll=1)

    return continue_stat


def init_counters(N: int) -> Dict[str, jnp.ndarray]:
    return {
        'nb_monotone_step': jnp.zeros(N),
        'nb_slow_mass_step': jnp.zeros(N),
        'nb_max_volume_step': jnp.zeros(N),
    }


def min_mass_heuristic(epsilon: float, mass: jnp.ndarray):
    should_continue_cond = mass >= epsilon

    return should_continue_cond


def max_mass_heuristic(init_mass: jnp.ndarray, mass: jnp.ndarray):
    should_continue_cond = mass <= 3 * init_mass

    return should_continue_cond


MONOTONIC_STOP_STEP = 80


def monotonic_heuristic(sign: jnp.ndarray, previous_sign: jnp.ndarray, monotone_counter: jnp.ndarray):
    sign_cond = (sign == previous_sign)
    monotone_counter = monotone_counter * sign_cond + 1
    should_continue_cond = monotone_counter <= MONOTONIC_STOP_STEP

    return should_continue_cond, monotone_counter


MASS_SPEED_THRESHOLD = 0.01
MASS_SPEED_STOP_STEP = 16


def min_mass_speed_heuristic(mass_speed: jnp.ndarray, mass_speed_counter: jnp.ndarray, R: float, dt: float):
    mass_speed_cond = jnp.array(mass_speed < MASS_SPEED_THRESHOLD / R / dt)
    mass_speed_counter = mass_speed_counter * mass_speed_cond + 1
    should_continue_cond = mass_speed_counter <= MASS_SPEED_STOP_STEP

    return should_continue_cond, mass_speed_counter


GROWTH_THRESHOLD = 500


def max_growth_heuristic(growth: jnp.ndarray, R: float):
    should_continue_cond = growth <= GROWTH_THRESHOLD / R**2

    return should_continue_cond


# Those are 3200 pixels activated
# This number comes from the early days when I set the threshold
# at 20% of a 128*128 ~= 3277 pixels
MASS_VOLUME_THRESHOLD = 3200
MASS_VOLUME_STOP_STEP = 50


def mass_volume_heuristic(mass_volume: jnp.ndarray, mass_volume_counter: jnp.ndarray, R: float):
    volume_cond = jnp.array(mass_volume > MASS_VOLUME_THRESHOLD / R**2)
    mass_volume_counter = mass_volume_counter * volume_cond + 1
    should_continue_cond = mass_volume_counter <= MASS_VOLUME_STOP_STEP

    return should_continue_cond, mass_volume_counter


###
# Utils
###
def stats_list_to_dict(all_stats: List[Dict]) -> Dict[str, jnp.ndarray]:
    if len(all_stats) == 0:
        return {}

    stats_dict_list: Dict[str, List[float]] = {}
    all_keys = list(all_stats[0].keys())
    for k in all_keys:
        stats_dict_list[k] = []
    for stat in all_stats:
        for k, v in stat.items():
            stats_dict_list[k].append(v)
    stats_dict = {k: jnp.array(stats_dict_list[k]) for k in all_keys}

    return stats_dict
