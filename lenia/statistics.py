import jax
import jax.numpy as jnp
from jax import jit
from typing import Dict, Callable, Tuple
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
# @jit
def check_heuristics(
    stats: Dict[str, jnp.ndarray],
    prev_should_continue: jnp.ndarray,
    init_mass: jnp.ndarray,
    previous_mass: jnp.ndarray,
    previous_sign: jnp.ndarray,
    counters: Dict[str, jnp.ndarray]
):
    mass = stats['mass']
    cond = min_mass_heuristic(EPSILON, mass)
    should_continue_cond = cond
    cond = max_mass_heuristic(init_mass, mass)
    should_continue_cond *= cond

    sign = jnp.sign(mass - previous_mass)
    monotone_counter = counters['nb_monotone_step']
    cond, counters['nb_monotone_step'] = monotonic_heuristic(sign, previous_sign, monotone_counter)
    should_continue_cond *= cond

    # growth = stats['growth']
    # cond = max_growth_heuristic(growth)
    # should_continue_cond *= cond

    mass_speed = stats['mass_speed']
    mass_speed_counter = counters['nb_slow_mass_step']
    cond, counters['nb_slow_mass_step'] = max_mass_speed_heuristic(mass_speed, mass_speed_counter)
    should_continue_cond *= cond

    percent_activated = stats['percent_activated']
    percent_activated_counter = counters['nb_too_much_percent_step']
    cond, counters['nb_too_much_percent_step'] = percent_activated_heuristic(
        percent_activated, percent_activated_counter
    )
    should_continue_cond *= cond

    should_continue = jax.ops.index_update(prev_should_continue, ~should_continue_cond, 0.)

    return should_continue, mass, sign


def min_mass_heuristic(epsilon: float, mass: jnp.ndarray):
    should_continue_cond = mass >= epsilon

    return should_continue_cond


def max_mass_heuristic(init_mass: jnp.ndarray, mass: jnp.ndarray):
    should_continue_cond = mass <= 3 * init_mass

    return should_continue_cond


def monotonic_heuristic(sign: jnp.ndarray, previous_sign: jnp.ndarray, monotone_counter: jnp.ndarray):
    sign_cond = (sign == previous_sign)
    monotone_counter = jax.ops.index_add(monotone_counter, sign_cond, 1.)
    monotone_counter = jax.ops.index_update(monotone_counter, ~sign_cond, 0.)
    should_continue_cond = monotone_counter <= 80

    return should_continue_cond, monotone_counter


MASS_SPEED_THRESHOLD = 0.01
MASS_SPEED_STOP_STEP = 16


def max_mass_speed_heuristic(mass_speed: jnp.ndarray, mass_speed_counter: jnp.ndarray):
    mass_speed_cond = (mass_speed < MASS_SPEED_THRESHOLD)
    mass_speed_counter = jax.ops.index_add(mass_speed_counter, mass_speed_cond, 1.)
    mass_speed_counter = jax.ops.index_update(mass_speed_counter, ~mass_speed_cond, 0.)
    should_continue_cond = mass_speed_counter <= MASS_SPEED_STOP_STEP

    return should_continue_cond, mass_speed_counter


def max_mass_speed_heuristic_seq(mass_speed: float, nb_slow_mass_step: int) -> Tuple[bool, int]:
    should_continue_cond = True
    if mass_speed < MASS_SPEED_THRESHOLD:
        nb_slow_mass_step += 1
        if nb_slow_mass_step > MASS_SPEED_STOP_STEP:
            should_continue_cond = False
    else:
        nb_slow_mass_step = 0

    return should_continue_cond, nb_slow_mass_step


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


def percent_activated_heuristic_seq(percent_activated: float, nb_too_much_percent_step: int) -> Tuple[bool, int]:
    # Stops when the world is N% covered
    # Hopefully, no species takes so much space
    should_continue_cond = True
    if percent_activated > PERCENT_ACTIVATED_THRESHOLD:
        nb_too_much_percent_step += 1
        if nb_too_much_percent_step > PERCENT_ACTIVATED_STOP_STEP:
            # print("nb_too_much_percent_step > should_stop")
            should_continue_cond = False
    else:
        nb_too_much_percent_step = 0

    return should_continue_cond, nb_too_much_percent_step
