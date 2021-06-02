import jax.numpy as jnp
from typing import Dict, Callable


def build_compute_stats_fn(world_params: Dict, render_params: Dict) -> Callable:
    world_size = render_params['world_size']
    nb_dims = world_params['nb_dims']
    R = world_params['R']

    midpoint = jnp.asarray([size // 2 for size in world_size])
    coords = jnp.indices(world_size)
    whitened_coords = [(coords[i] - midpoint[i]) / R for i in range(coords.shape[0])]

    def compute_stats(cells, field, potential):
        mass = cells.sum()
        pos_field = jnp.maximum(field, 0)
        growth = pos_field.sum()

        # Those statistics works on a centered world
        AX = [cells * x for x in whitened_coords]
        MX1 = [ax.sum() for ax in AX]
        center_of_mass = jnp.asarray(MX1) / mass
        cells, field, potential, shift_idx = center_world(cells, field, potential, center_of_mass, R, nb_dims)

        MX2 = [(ax * x).sum() for ax, x in zip(AX, whitened_coords)]
        MuX2 = [mx2 - mx * mx1 for mx, mx1, mx2 in zip(center_of_mass, MX1, MX2)]
        inertia = sum(MuX2)

        GX1 = [(pos_field * x).sum() for x in whitened_coords]
        center_of_growth = jnp.asarray(GX1) / growth
        mg_dist = jnp.linalg.norm(center_of_mass - center_of_growth)

        stats = {'mass': mass, 'growth': growth, 'inertia': inertia, 'mg_dist': mg_dist}

        return (cells, field, potential, shift_idx, stats)

    return compute_stats


def center_world(cells, field, potential, center_of_mass, R, nb_dims):
    axes = tuple(reversed(range(-1, -nb_dims - 1, -1)))
    shift_idx = (center_of_mass * R).astype(int)
    cells = jnp.roll(cells, -shift_idx, axes)
    field = jnp.roll(field, -shift_idx, axes)
    potential = jnp.roll(potential, -shift_idx, axes)

    return cells, field, potential, shift_idx
