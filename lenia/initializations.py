import math
import jax
import jax.numpy as jnp
from typing import List, Dict

from .perlin import generate_perlin_noise_2d


def random_uniform(rng_key, nb_noise: int, world_size: List[int], nb_channels: int):
    rng_key, subkey = jax.random.split(rng_key)
    maxvals = jnp.linspace(0.4, 1., nb_noise)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    rand_shape = [nb_noise] + [nb_channels] + world_size
    noises = jax.random.uniform(subkey, rand_shape, minval=0, maxval=maxvals)

    return rng_key, noises


def random_uniform_1k(rng_key, nb_noise: int, world_size, nb_channels: int, kernel_params):
    """
    Random uniform initialization which target the mean of the growth function
    """
    rng_key, subkey = jax.random.split(rng_key)
    rand_shape = [nb_noise] + [nb_channels] + world_size

    cells = jax.random.uniform(subkey, rand_shape, minval=0, maxval=kernel_params['m'] * 2)

    return rng_key, cells


def sinusoide_1c1k(rng_key, world_size, R, kernel_params):
    rng_key, subkey = jax.random.split(rng_key)

    midpoint = jnp.asarray([size // 2 for size in world_size])  # [nb_dims]
    midpoint = midpoint.reshape([-1] + [1] * len(world_size))  # [nb_dims, 1, 1, ...]
    coords = jnp.indices(world_size)  # [nb_dims, dim_0, dim_1, ...]
    centered_coords = (coords - midpoint) / (kernel_params['r'] * R)  # [nb_dims, dim_0, dim_1, ...]

    phases = jnp.array([0, 0]).reshape([-1] + [1] * len(world_size))
    freqs = jnp.array([1, 3]).reshape([-1] + [1] * len(world_size))
    scaled_coords = freqs * centered_coords + phases
    # Distances from the center of the grid
    distances = jnp.sqrt(jnp.sum(scaled_coords**2, axis=0))  # [dim_0, dim_1, ...]
    cells = ((distances % (6 * math.pi)) < 2 * math.pi) * (jnp.sin(distances) + 1) / 2.

    return rng_key, cells


def perlin(rng_key, nb_noise: int, world_size: List[int], R: float, kernel_params: Dict):
    """
    Perlin noise initialization which target the mean of the growth function
    """
    kernel_radius = int(R * kernel_params['r'])
    res = [world_size[0] // (kernel_radius * 3), world_size[1] // (kernel_radius * 2)]
    min_mean_bound = kernel_params['m']
    max_mean_bound = min(1, 2 * kernel_params['m'])
    scaling = jnp.array([min_mean_bound + i / nb_noise * (max_mean_bound - min_mean_bound) for i in range(nb_noise)])
    scaling = scaling[:, jnp.newaxis, jnp.newaxis]

    rng_key, subkey = jax.random.split(rng_key)
    angles_shape = [nb_noise] + res
    angles = 2 * jnp.pi * jax.random.uniform(subkey, shape=angles_shape)  # [nb_noise, H_res, W_res]

    # Why tuple here? -> List are non-hashable which breaks jit function
    cells = generate_perlin_noise_2d(angles, tuple(world_size), tuple(res), nb_noise)

    cells -= cells.min(axis=(1, 2), keepdims=True)
    cells /= cells.max(axis=(1, 2), keepdims=True)
    cells *= scaling
    init_cells = cells[:, jnp.newaxis, ...]

    return rng_key, init_cells
