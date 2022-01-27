import math
import jax
import jax.numpy as jnp
from typing import List, Dict, Callable

from .loader import make_array_compressible
from .perlin import generate_perlin_noise_2d


def random_uniform(rng_key, nb_noise: int, world_size: List[int], nb_channels: int):
    rng_key, subkey = jax.random.split(rng_key)
    maxvals = jnp.linspace(0.4, 1., nb_noise)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    rand_shape = [nb_noise] + [nb_channels] + world_size
    cells = jax.random.uniform(subkey, rand_shape, minval=0, maxval=maxvals)

    cells = make_array_compressible(cells)

    return rng_key, cells


def random_uniform_1k(rng_key, nb_noise: int, world_size, nb_channels: int, kernel_params):
    """
    Random uniform initialization which target the mean of the growth function
    """
    rng_key, subkey = jax.random.split(rng_key)
    rand_shape = [nb_noise] + [nb_channels] + world_size

    cells = jax.random.uniform(subkey, rand_shape, minval=0, maxval=kernel_params['m'] * 2)

    cells = make_array_compressible(cells)

    return rng_key, cells


def perlin(rng_key, nb_noise: int, world_size: List[int], R: float, k_params: Dict, gf_params: list):
    """
    Perlin noise initialization which target the mean of the growth function
    """
    r = k_params[1]

    kernel_radius = math.ceil(R * r)
    res = [world_size[0] // (kernel_radius * 3), world_size[1] // (kernel_radius * 2)]
    min_mean_bound = gf_params[0]
    max_mean_bound = min(1, 3 * min_mean_bound)
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

    init_cells = make_array_compressible(init_cells)

    return rng_key, init_cells


def perlin_local(rng_key, nb_noise: int, world_size: List[int], R: float, kernel_params: Dict):
    """
    Perlin noise initialization which target the mean of the growth function
    """
    kernel_radius = int(R * kernel_params['r'])
    res = [world_size[0] // (kernel_radius * 3), world_size[1] // (kernel_radius * 2)]
    min_mean_bound = kernel_params['m']
    max_mean_bound = min(1, 3 * kernel_params['m'])
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

    size = kernel_radius * 2
    pad_left = (128 - size) // 2
    if pad_left * 2 + size != 128:
        pad_right = pad_left + 1
    else:
        pad_right = pad_left
    init_cells = jnp.pad(
        init_cells[:, :, 24:24 + size, 24:24 + size], ((0, 0), (0, 0), (pad_left, pad_right), (pad_left, pad_right)),
        mode='constant',
        constant_values=(0, 0)
    )

    init_cells = make_array_compressible(init_cells)

    return rng_key, init_cells


register: Dict[str, Callable] = {
    'random_uniform': random_uniform,
    'random_uniform_1k': random_uniform_1k,
    'perlin': perlin,
    'perlin_local': perlin_local
}
