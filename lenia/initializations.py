import math
import jax
import jax.numpy as jnp
from .perlin import generate_perlin_noise_2d


def random_uniform(rng_key, nb_noise: int, nb_channels: int):
    key, subkey = jax.random.split(rng_key)
    maxvals = jnp.linspace(0.4, 1., nb_noise)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    rand_shape = [nb_noise] + [nb_channels] + [128, 128]
    noises = jax.random.uniform(subkey, rand_shape, minval=0, maxval=maxvals)

    return key, noises


def random_uniform_1c1k(rng_key, world_size, kernel_params):
    """
    Random uniform initialization which target the mean of the growth function
    """
    key, subkey = jax.random.split(rng_key)
    shape = jnp.array([1] + world_size)

    cells = jax.random.uniform(subkey, shape, minval=0, maxval=kernel_params['m'] * 2)

    return key, cells


def sinusoide_1c1k(rng_key, world_size, R, kernel_params):
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

    return cells


def perlin(rng_key, world_size, kernel_params):
    """
    Perlin noise initialization which target the mean of the growth function
    """
    res = [world_size[0] // 2 ^ 5, world_size[1] // 2 ^ 5]
    rng_key, cells = generate_perlin_noise_2d(rng_key, world_size, res, tileable=(True, True))

    cells -= cells.min()
    cells /= cells.max()
    cells *= kernel_params['m'] / cells.mean()

    return cells
