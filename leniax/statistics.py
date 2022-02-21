import functools
import jax
import jax.numpy as jnp
from jax import jit
from typing import List, Dict, Callable, Tuple

from .constant import EPSILON
from .utils import center_world


def build_compute_stats_fn(world_params: Dict, render_params: Dict) -> Callable:
    """Construct the conpute_statistics function

    Args:
        world_params: World parameters dictrionnary.
        render_params: Render parameters dictrionnary.

    Returns:
        The compute statistics function
    """

    world_size = render_params['world_size']
    R = world_params['R']
    dt = 1. / world_params['T']

    world_dims_axes = tuple(range(-len(world_size), 0, 1))
    non_batch_dims_axes = tuple(range(-(1 + len(world_size)), 0, 1))
    midpoint = jnp.expand_dims(
        jnp.asarray([size // 2 for size in world_size]), axis=world_dims_axes
    )  # [nb_dims, 1, 1, ...]
    coords = jnp.indices(world_size)  # [nb_dims, H, W, ...]
    # We expand our coordinate to control and take advantage of automatic broadcasting
    centered_coords = jnp.expand_dims(coords - midpoint, axis=(1, 2))  # [nb_dims, 1, 1, H, W, ...]

    @jit
    def compute_stats(
        cells: jnp.ndarray,
        field: jnp.ndarray,
        potential: jnp.ndarray,
        previous_total_shift_idx: jnp.ndarray,
        previous_mass_centroid: jnp.ndarray,
        previous_mass_angle: jnp.ndarray,
    ) -> Tuple[Dict, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute statistics of a Leniax simulation

        Args:
            cells: state of shape ``[N, C, world_dims...]``
            field: state of shape ``[N, C, world_dims...]``
            potential: state of shape ``[N, C, world_dims...]``
            previous_total_shift_idx: state of shape ``[N, 2]``
            previous_mass_centroid: state of shape ``[2, N]``
            previous_mass_angle: state of shape ``[2, N]``

        Returns:
            A tuple containing the statistics dictionnary and other carry informations
        """
        # cells: # [N, C, H, W]
        # field: # [N, C, H, W]
        # potential: # [N, C, H, W]

        # https://en.wikipedia.org/wiki/Image_moment
        # To avoid weird behaviours when species are crossing the frontiers, we need to compute stats
        # from a centered world
        centered_cells, centered_field, _ = center_world(cells, field, potential, previous_total_shift_idx, world_dims_axes)
        positive_field = jnp.maximum(centered_field, 0)

        m_00 = centered_cells.sum(axis=non_batch_dims_axes)  # [N]
        g_00 = positive_field.sum(axis=non_batch_dims_axes)  # [N]

        potential_volume = (potential > EPSILON).sum(axis=non_batch_dims_axes) / R**2  # [N]

        channel_mass = centered_cells.sum(axis=world_dims_axes) / R**2  # [N, C]
        mass = m_00 / R**2
        mass_volume = (centered_cells > EPSILON).sum(axis=non_batch_dims_axes) / R**2  # [N]
        mass_density = mass / (mass_volume + EPSILON)

        growth = g_00 / R**2
        growth_volume = (positive_field > EPSILON).sum(axis=non_batch_dims_axes) / R**2  # [N]
        growth_density = growth / (growth_volume + EPSILON)

        AX = centered_cells * centered_coords  # [nb_world_dims, N, C, world_dims...]
        MX = AX.sum(axis=non_batch_dims_axes)  # [nb_world_dims, N]
        mass_centroid = MX / (m_00 + EPSILON)

        delta_mass = mass_centroid - previous_mass_centroid  # [nb_world_dims, N]
        dist_m = jnp.linalg.norm(delta_mass, axis=0)  # [N]

        mass_speed = dist_m / R / dt  # [N]
        mass_angle = jnp.degrees(jnp.arctan2(delta_mass[1], delta_mass[0])) * (dist_m / R > 0.001)

        mass_angle_speed = ((mass_angle - previous_mass_angle + 540) % 360 - 180) / dt  # [N]

        GX = (positive_field * centered_coords).sum(axis=non_batch_dims_axes)  # [nb_world_dims, N]
        growth_centroid = GX / (g_00 + EPSILON)
        mass_growth_dist = jnp.linalg.norm(growth_centroid - mass_centroid, axis=0) / R

        MX2 = (AX * centered_coords).sum(axis=non_batch_dims_axes)  # [nb_world_dims, N]
        MuX2 = MX2 - mass_centroid * MX
        mass2_centroid = MuX2 / (m_00**2 + EPSILON)
        inertia = mass2_centroid.sum(axis=0)

        stats = {
            'channel_mass': channel_mass,
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
            'potential_volume': potential_volume,
        }

        shift_idx = mass_centroid.astype(jnp.int32).T
        world_shape = jnp.array(cells.shape[2:], dtype=jnp.int32)
        total_shift_idx = (previous_total_shift_idx + shift_idx) % world_shape

        # Since we will center the world before computing stats
        # The mass_centroid will also be shifted, so here we make sure we avoid the
        # shifting errors
        mass_centroid = mass_centroid - mass_centroid.astype(jnp.int32)

        return (stats, total_shift_idx, mass_centroid, mass_angle)

    return compute_stats


###
# Heuristics
###
@functools.partial(jit, static_argnums=(1, ))
def check_heuristics(stats: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Check heuristics on statistic data

    Args:
        stats: Simulation statistics dictionnary

    Returns:
        An array of boolean value indicating if the heuristics are valid for each timsteps
    """
    def fn(carry: Dict, stat_t: Dict[str, jnp.ndarray]):
        should_continue = carry['should_continue']
        init_channel_mass = carry['init_channel_mass']
        init_mass = carry['init_mass']
        previous_mass = carry['previous_mass']
        previous_sign = carry['previous_sign']
        counters = carry['counters']

        mass = stat_t['mass']
        channel_mass = stat_t['channel_mass']
        # cond = min_mass_heuristic(EPSILON, mass)
        # should_continue_cond = cond
        # cond = max_mass_heuristic(init_mass, mass)
        # should_continue_cond *= cond
        cond = min_channel_mass_heuristic(EPSILON, channel_mass)
        should_continue_cond = cond
        cond = max_channel_mass_heuristic(init_channel_mass, channel_mass)
        should_continue_cond *= cond

        sign = jnp.sign(mass - previous_mass)
        monotone_counter = counters['nb_monotone_step']
        cond, counters['nb_monotone_step'] = monotonic_heuristic(sign, previous_sign, monotone_counter)
        should_continue_cond *= cond

        mass_volume = stat_t['mass_volume']
        mass_volume_counter = counters['nb_max_volume_step']
        cond, counters['nb_max_volume_step'] = mass_volume_heuristic(mass_volume, mass_volume_counter)
        should_continue_cond *= cond

        should_continue *= should_continue_cond

        new_carry = {
            'should_continue': should_continue,
            'init_channel_mass': init_channel_mass,
            'init_mass': init_mass,
            'previous_mass': mass,
            'previous_sign': sign,
            'counters': counters,
        }

        return new_carry, should_continue

    N = stats['mass'].shape[1]
    init_carry = {
        'should_continue': jnp.ones(N),
        'init_channel_mass': stats['channel_mass'][0],
        'init_mass': stats['mass'][0],
        'previous_mass': stats['mass'][0],
        'previous_sign': jnp.zeros(N),
        'counters': init_counters(N),
    }
    _, continue_stat = jax.lax.scan(fn, init_carry, stats, unroll=1)

    # Used for debugging (lax.scan is a jitted function)
    # continue_stat = []
    # all_keys = stats.keys()
    # for i in range(stats['mass'].shape[0]):
    #     stat_t = {key: stats[key][i] for key in all_keys}
    #     init_carry, should_continue = fn(init_carry, stat_t)
    #     continue_stat.append(should_continue)

    return continue_stat


def init_counters(N: int) -> Dict[str, jnp.ndarray]:
    """Initialize different counters used in heuristics decisions

    Args:
        N: Number of simulated timesteps

    Returns:
        Adictionnary of counters
    """
    return {
        'nb_monotone_step': jnp.zeros(N, dtype=jnp.int32),
        'nb_slow_mass_step': jnp.zeros(N, dtype=jnp.int32),
        'nb_max_volume_step': jnp.zeros(N, dtype=jnp.int32),
    }


def min_channel_mass_heuristic(epsilon: float, channel_mass: jnp.ndarray) -> jnp.ndarray:
    """Check if a total mass per channel is below the threshold

    Args:
        epsilon: A very small value to avoid division by zero
        channel_mass: Total mass per channel of shape ``[N, C]``

    Returns:
        A boolean array of shape ``[N]``
    """
    should_continue_cond = (channel_mass >= epsilon).all(axis=1)

    return should_continue_cond


def max_channel_mass_heuristic(init_channel_mass: jnp.ndarray, channel_mass: jnp.ndarray) -> jnp.ndarray:
    """Check if a total mass per channel is above the threshold

    Args:
        init_channel_mass: Initial mass per channel of shape ``[N, C]``
        channel_mass: Total mass per channel of shape ``[N, C]``

    Returns:
        A boolean array of shape ``[N]``
    """
    should_continue_cond = (channel_mass <= 3 * init_channel_mass).all(axis=1)

    return should_continue_cond


def min_mass_heuristic(epsilon: float, mass: jnp.ndarray) -> jnp.ndarray:
    """Check if the total mass of the system is below the threshold

    Args:
        epsilon: A very small value to avoid division by zero
        mass: Total mass of shape ``[N]``

    Returns:
        A boolean array of shape ``[N]``
    """
    should_continue_cond = mass >= epsilon

    return should_continue_cond


def max_mass_heuristic(init_mass: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Check if a total mass per channel is above the threshold

    Args:
        init_mass: Initial mass per channel of shape ``[N]``
        mass: Total mass per channel of shape ``[N]``

    Returns:
        A boolean array of shape ``[N]``
    """
    should_continue_cond = mass <= 3 * init_mass

    return should_continue_cond


MONOTONIC_STOP_STEP = 128


def monotonic_heuristic(
    sign: jnp.ndarray,
    previous_sign: jnp.ndarray,
    monotone_counter: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Check if the mass variation is being monotonic for too many timesteps

    Args:
        sign: Current sign of mass variation of shape ``[N]``
        previous_sign: Previous sign of mass variation of shape ``[N]``
        monotone_counter: Counter used to count number of timesteps with monotonic variations of shape ``[N]``

    Returns:
        A tuple representing a boolean array of shape ``[N]`` and the counter
    """
    sign_cond = (sign == previous_sign)
    monotone_counter = monotone_counter * sign_cond + 1
    should_continue_cond = monotone_counter <= MONOTONIC_STOP_STEP

    return should_continue_cond, monotone_counter


# Those are 3200 pixels activated
# This number comes from the early days when I set the threshold
# at 10% of a 128*128 ~= 1600 pixels with a kernel radius of 13
# which gives a volume threshold of 1600 / 13^2 ~= 9.5
MASS_VOLUME_THRESHOLD = 10.
MASS_VOLUME_STOP_STEP = 128


def mass_volume_heuristic(mass_volume: jnp.ndarray,
                          mass_volume_counter: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Check if the mass volume is above the threshold for too manye timesteps

    Args:
        mass_volume: Mass volume of shape ``[N]``
        mass_volume_counter: Counter of shape ``[N]`` used to count number of timesteps with a volume above the threshold

    Returns:
        A tuple representing a boolean array of shape ``[N]`` and the counter
    """

    volume_cond = jnp.array(mass_volume > MASS_VOLUME_THRESHOLD)
    mass_volume_counter = mass_volume_counter * volume_cond + 1
    should_continue_cond = mass_volume_counter <= MASS_VOLUME_STOP_STEP

    return should_continue_cond, mass_volume_counter


###
# Utils
###
def stats_list_to_dict(all_stats: List[Dict]) -> Dict[str, jnp.ndarray]:
    """Change a list of dictionnary in a dictionnary of array

    Args:
        all_stats: List of 1-timestep statistics dictionary

    Returns:
        A dictionnary of N-timestep array.
    """
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
