import jax.numpy as jnp
from typing import Dict, Callable, List
from .constant import EPSILON


def build_compute_stats_fn(world_params: Dict, render_params: Dict) -> Callable:
    world_size = render_params['world_size']
    nb_dims = world_params['nb_dims']
    R = world_params['R']

    midpoint = jnp.asarray([size // 2 for size in world_size])[:, jnp.newaxis, jnp.newaxis]
    coords = jnp.indices(world_size)
    whitened_coords = (coords - midpoint) / R  # [nb_dims, H, W]

    def compute_stats(cells, field, potential):
        # cells: # [C, N=1, c=1, H, W]
        # field: # [C, N=1, c=1, H, W]
        # potential: # [C, H, W]

        mass = cells.sum()
        pos_field = jnp.maximum(field, 0)
        growth = pos_field.sum()
        percent_activated = (cells > EPSILON).mean()

        # Those statistics works on a centered world
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

        centered_cells, centered_field, centered_potential, shift_idx = center_world(
            cells, field, potential, mass_center, R, nb_dims
        )

        stats = {
            'mass': mass,
            'growth': growth,
            'inertia': inertia,
            'mass_growth_dist': mass_growth_dist,
            'mass_speed': mass_speed,
            'mass_angle': mass_angle,
            'percent_activated': percent_activated
        }

        return (centered_cells, centered_field, centered_potential, shift_idx, stats)

    return compute_stats


def center_world(cells, field, potential, mass_center, R, nb_dims):
    axes = tuple(reversed(range(-1, -nb_dims - 1, -1)))
    shift_idx = (mass_center * R).astype(int)
    cells = jnp.roll(cells, -shift_idx, axes)
    field = jnp.roll(field, -shift_idx, axes)
    potential = jnp.roll(potential, -shift_idx, axes)

    return cells, field, potential, shift_idx


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
