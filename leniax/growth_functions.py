import jax.numpy as jnp


def poly_quad4(X: jnp.ndarray, m: float, s: float):
    out = 1 - (X - m)**2 / (9 * s**2)
    out = jnp.maximum(0, out)
    out = out**4 * 2 - 1

    return out


def gaussian(X: jnp.ndarray, m: float, s: float):
    out = ((X - m) / s)**2
    out = jnp.exp(-out / 2) * 2 - 1

    return out


def gaussian_target(X: jnp.ndarray, m: float, s: float):
    out = ((X - m) / s)**2
    out = jnp.exp(-out / 2)

    return out


def step(X: jnp.ndarray, m: float, s: float):
    out = jnp.abs(X - m)
    out = (out <= s) * 2 - 1

    return out


growth_fns = {0: poly_quad4, 1: gaussian, 2: gaussian_target, 3: step}
