"""
Source code taken directly from: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
and adapted to jax by Morgan Giraud
"""

import jax
import jax.numpy as jnp


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(rng_key, shape, res, interpolant=interpolant):
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
    rng_key, subkey = jax.random.split(rng_key)

    # Gradients
    d = (shape[0] // res[0], shape[1] // res[1])
    angles = 2 * jnp.pi * jax.random.uniform(subkey, shape=res)  # [H_res, W_res]
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))  # [H_res, W_res, 2]
    gradients = jnp.pad(gradients, [(0, 1), (0, 1), (0, 0)], mode='wrap')  # [H_res + 1, W_res + 1, 2]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)  # []

    diff = [gradients.shape[0] - shape[0], gradients.shape[1] - shape[1]]
    g00 = gradients[:-diff[0], :-diff[1]]
    g10 = gradients[diff[0]:, :-diff[1]]
    g01 = gradients[:-diff[0], diff[1]:]
    g11 = gradients[diff[0]:, diff[1]:]

    # Ramps
    delta = (res[0] / shape[0], res[1] / shape[1])
    grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    n00 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return rng_key, jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
