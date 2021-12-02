"""
Source code taken directly from: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
and adapted to jax by Morgan Giraud
"""

import functools
import jax
import jax.numpy as jnp
from typing import Tuple


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_perlin_noise_2d(
    angles: jnp.ndarray,
    shape: Tuple[int, int],
    res: Tuple[int, int],
    nb_noise: int = 1,
    interpolant=interpolant
) -> jnp.ndarray:
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """

    # Gradients
    gradients = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # [nb_noise, H_res, W_res, 2]
    gradients = jnp.pad(gradients, [(0, 0), (0, 1), (0, 1), (0, 0)], mode='wrap')  # [nb_noise, H_res + 1, W_res + 1, 2]
    d = (shape[0] // res[0], shape[1] // res[1])
    gradients = gradients.repeat(d[0], 1).repeat(d[1], 2)  # [nb_noise, (H_res + 1) * d, (W_res + 1) * d, 2]

    diff = [gradients.shape[1] - shape[0], gradients.shape[2] - shape[1]]
    g00 = gradients[:, :-diff[0], :-diff[1]]  # [nb_noise, shape, 2]
    g10 = gradients[:, diff[0]:, :-diff[1]]  # [nb_noise, shape, 2]
    g01 = gradients[:, :-diff[0], diff[1]:]  # [nb_noise, shape, 2]
    g11 = gradients[:, diff[0]:, diff[1]:]  # [nb_noise, shape, 2]

    # Ramps
    delta = (res[0] / shape[0], res[1] / shape[1])
    grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1  # type: ignore
    grid = grid[jnp.newaxis, ...]  # [1, shape, 2]
    grid = grid.repeat(nb_noise, 0)  # [nb_noise, shape, 2]
    n00 = jnp.sum(jnp.stack([grid[:, :, :, 0], grid[:, :, :, 1]], axis=-1) * g00, 3)  # [nb_noise, shape]
    n10 = jnp.sum(jnp.stack([grid[:, :, :, 0] - 1, grid[:, :, :, 1]], axis=-1) * g10, 3)  # [nb_noise, shape]
    n01 = jnp.sum(jnp.stack([grid[:, :, :, 0], grid[:, :, :, 1] - 1], axis=-1) * g01, 3)  # [nb_noise, shape]
    n11 = jnp.sum(jnp.stack([grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1], axis=-1) * g11, 3)  # [nb_noise, shape]

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n10  # [nb_noise, shape]
    n1 = n01 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n11  # [nb_noise, shape]

    return jnp.sqrt(2) * ((1 - t[:, :, :, 1]) * n0 + t[:, :, :, 1] * n1)
