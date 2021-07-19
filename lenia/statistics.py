import jax
import jax.numpy as jnp
from jax import jit
from typing import List, Dict, Callable, Tuple
from .constant import EPSILON


def build_compute_stats_fn(world_params: Dict, render_params: Dict) -> Callable:
    world_size = render_params['world_size']
    R = world_params['R']

    midpoint = jnp.asarray([size // 2 for size in world_size])[:, jnp.newaxis, jnp.newaxis]
    coords = jnp.indices(world_size)
    whitened_coords = (coords - midpoint) / R  # [nb_dims, H, W]

    @jit
    def compute_stats(cells, field, potential):
        # cells: # [C, N=1, c=1, H, W]
        # field: # [C, N=1, c=1, H, W]
        # potential: # [C, H, W]
        mass = cells.sum()
        pos_field = jnp.maximum(field, 0)
        growth = pos_field.sum()
        percent_activated = (cells > EPSILON).mean()

        # Those statistic works on a centered world
        AX = [cells * x for x in whitened_coords]
        MX1 = [ax.sum() for ax in AX]
        mass_center = jnp.asarray(MX1) / (mass + EPSILON)
        # This function returns "mass centered" cells so the previous mass_center is at origin
        dm = mass_center
        mass_speed = jnp.linalg.norm(dm)
        mass_angle = jnp.degrees(jnp.arctan2(dm[1], dm[0]))

        MX2 = [(ax * x).sum() for ax, x in zip(AX, whitened_coords)]
        MuX2 = [mx2 - mx * mx1 for mx, mx1, mx2 in zip(mass_center, MX1, MX2)]
        inertia = sum(MuX2)

        GX1 = [(pos_field * x).sum() for x in whitened_coords]
        growth_center = jnp.asarray(GX1) / (growth + EPSILON)
        mass_growth_dist = jnp.linalg.norm(mass_center - growth_center)

        shift_idx = (mass_center * R).astype(int)
        # stats = jnp.array([
        #     mass,
        #     growth,
        #     inertia,
        #     mass_growth_dist,
        #     mass_speed,
        #     mass_angle,
        #     percent_activated
        # ])
        stats = {
            'mass': jnp.array(mass),
            'growth': jnp.array(growth),
            'inertia': jnp.array(inertia),
            'mass_growth_dist': jnp.array(mass_growth_dist),
            'mass_speed': jnp.array(mass_speed),
            'mass_angle': jnp.array(mass_angle),
            'percent_activated': jnp.array(percent_activated)
        }

        return (stats, shift_idx)

    return compute_stats


###
# Heuristics
###
@jit
def check_heuristics(stats: Dict[str, jnp.ndarray]):
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
        cond, counters['nb_monotone_step'] = monotonic_heuristic_seq(sign, previous_sign, monotone_counter)
        should_continue_cond *= cond

        # growth = stat_t['growth']
        # cond = max_growth_heuristic(growth)
        # should_continue_cond *= cond

        mass_speed = stat_t['mass_speed']
        mass_speed_counter = counters['nb_slow_mass_step']
        cond, counters['nb_slow_mass_step'] = max_mass_speed_heuristic_seq(mass_speed, mass_speed_counter)
        should_continue_cond *= cond

        percent_activated = stat_t['percent_activated']
        percent_activated_counter = counters['nb_too_much_percent_step']
        cond, counters['nb_too_much_percent_step'] = percent_activated_heuristic_seq(
            percent_activated, percent_activated_counter
        )
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

    init_carry = {
        'should_continue': True,
        'init_mass': stats['mass'][0],
        'previous_mass': stats['mass'][0],
        'previous_sign': 0.,
        'counters': init_counters(),
    }
    final_carry, continue_stat = jax.lax.scan(fn, init_carry, stats, unroll=1)

    return continue_stat


def init_counters() -> Dict[str, int]:
    return {
        'nb_monotone_step': 0,
        'nb_slow_mass_step': 0,
        'nb_too_much_percent_step': 0,
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
    monotone_counter = jax.ops.index_add(monotone_counter, sign_cond, 1.)
    monotone_counter = jax.ops.index_update(monotone_counter, ~sign_cond, 0.)
    should_continue_cond = monotone_counter <= MONOTONIC_STOP_STEP

    return should_continue_cond, monotone_counter


def monotonic_heuristic_seq(sign: float, previous_sign: float, monotone_counter: int) -> Tuple[bool, int]:
    sign_cond = (sign == previous_sign)
    monotone_counter = monotone_counter * sign_cond + 1
    should_continue_cond = monotone_counter <= MONOTONIC_STOP_STEP

    return should_continue_cond, monotone_counter


MASS_SPEED_THRESHOLD = 0.01
MASS_SPEED_STOP_STEP = 16


def max_mass_speed_heuristic(mass_speed: jnp.ndarray, mass_speed_counter: jnp.ndarray):
    mass_speed_cond = (mass_speed < MASS_SPEED_THRESHOLD)
    mass_speed_counter = jax.ops.index_add(mass_speed_counter, mass_speed_cond, 1.)
    mass_speed_counter = jax.ops.index_update(mass_speed_counter, ~mass_speed_cond, 0.)
    should_continue_cond = mass_speed_counter <= MASS_SPEED_STOP_STEP

    return should_continue_cond, mass_speed_counter


def max_mass_speed_heuristic_seq(mass_speed: float, mass_speed_counter: int) -> Tuple[bool, int]:
    mass_speed_cond = (mass_speed < MASS_SPEED_THRESHOLD)
    mass_speed_counter = mass_speed_counter * mass_speed_cond + 1
    should_continue_cond = mass_speed_counter <= MASS_SPEED_STOP_STEP

    return should_continue_cond, mass_speed_counter


GROWTH_THRESHOLD = 500


def max_growth_heuristic(growth: jnp.ndarray):
    should_continue_cond = growth <= GROWTH_THRESHOLD

    return should_continue_cond


def max_growth_heuristic_sec(growth: float):
    should_continue_cond = growth <= GROWTH_THRESHOLD

    return should_continue_cond


PERCENT_ACTIVATED_THRESHOLD = 0.2
PERCENT_ACTIVATED_STOP_STEP = 50


def percent_activated_heuristic(percent_activated: jnp.ndarray, percent_activated_counter: jnp.ndarray):
    percent_activated_cond = (percent_activated > PERCENT_ACTIVATED_THRESHOLD)
    percent_activated_counter = jax.ops.index_add(percent_activated_counter, percent_activated_cond, 1.)
    percent_activated_counter = jax.ops.index_update(percent_activated_counter, ~percent_activated_cond, 0.)
    should_continue_cond = percent_activated_counter <= PERCENT_ACTIVATED_STOP_STEP

    return should_continue_cond, percent_activated_counter


def percent_activated_heuristic_seq(percent_activated: float, percent_activated_counter: int) -> Tuple[bool, int]:
    # Stops when the world is N% covered
    # Hopefully, no species takes so much space
    percent_cond = (percent_activated > PERCENT_ACTIVATED_THRESHOLD)
    percent_activated_counter = percent_activated_counter * percent_cond + 1
    should_continue_cond = percent_activated_counter <= PERCENT_ACTIVATED_STOP_STEP

    return should_continue_cond, percent_activated_counter


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
