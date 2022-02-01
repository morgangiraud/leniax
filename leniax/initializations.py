import math
import jax
import jax.numpy as jnp
from typing import List, Dict, Callable, Tuple

from .loader import make_array_compressible
from .perlin import generate_perlin_noise_2d


def random_uniform(rng_key: jax.random.KeyArray, nb_init: int, world_size: List[int], R: float,
                   gf_params: List) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Random uniform initial state

    Args:
        rng_key: JAX PRNG key.
        nb_init: Number of initializations
        world_size: World size in pixels along each dimensions
        R: Kernels world radius
        gf_params: Growth_function parameters

    Returns:
        The new rng_key and initial states
    """
    rng_key, subkey = jax.random.split(rng_key)

    rand_shape = [nb_init] + world_size
    maxvals = jnp.linspace(0.4, 1., nb_init)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    cells = jax.random.uniform(subkey, rand_shape, minval=0, maxval=maxvals)

    cells = make_array_compressible(cells)

    return rng_key, cells


def perlin(
    rng_key: jax.random.KeyArray,
    nb_init: int,
    world_size: List[int],
    R: float,
    gf_params: List,
) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Perlin noise initial state

    Args:
        rng_key: JAX PRNG key.
        nb_init: Number of initializations
        world_size: World size in pixels along each dimensions
        R: Kernels world radius
        K_params: Kernel parameters
        gf_params: Growth_function parameters

    Returns:
        The new rng_key and initial states
    """
    kernel_radius = math.ceil(R)
    res = [world_size[0] // (kernel_radius * 3), world_size[1] // (kernel_radius * 2)]
    min_mean_bound = gf_params[0]
    max_mean_bound = min(1, 3 * min_mean_bound)
    scaling = jnp.array([min_mean_bound + i / nb_init * (max_mean_bound - min_mean_bound) for i in range(nb_init)])
    scaling = scaling[:, jnp.newaxis, jnp.newaxis]

    rng_key, subkey = jax.random.split(rng_key)
    angles_shape = [nb_init] + res
    angles = 2 * jnp.pi * jax.random.uniform(subkey, shape=angles_shape)  # [nb_init, H_res, W_res]

    # Why tuple here? -> List are non-hashable which breaks jit function
    cells = generate_perlin_noise_2d(angles, tuple(world_size), tuple(res), nb_init)

    cells -= cells.min(axis=(1, 2), keepdims=True)
    cells /= cells.max(axis=(1, 2), keepdims=True)
    cells *= scaling
    init_cells = cells[:, jnp.newaxis, ...]

    init_cells = make_array_compressible(init_cells)

    return rng_key, init_cells


def cropped_perlin(
    rng_key: jax.random.KeyArray,
    nb_init: int,
    world_size: List[int],
    R: float,
    gf_params: List,
) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Cropped Perlin noise initial state

    Args:
        rng_key: JAX PRNG key.
        nb_init: Number of initializations
        world_size: World size in pixels along each dimensions
        R: Kernels world radius
        gf_params: Growth_function parameters

    Returns:
        The new rng_key and initial states
    """
    rng_key, init_cells = perlin(rng_key, nb_init, world_size, R, gf_params)

    kernel_radius = math.ceil(R)
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
    'perlin': perlin,
    'cropped_perlin': cropped_perlin,
}
