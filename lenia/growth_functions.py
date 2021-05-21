import jax.numpy as jnp


def poly_quad4(X: jnp.array, m: float, s: float):
    X = 1 - (X - m)**2 / (9 * s**2)
    X = jnp.maximum(0, X)
    X = X**4 * 2 - 1

    return X


def gaussian(X: jnp.array, m: float, s: float):
    X = -(X - m)**2 / (2 * s**2)
    X = jnp.exp(X) * 2 - 1

    return X


def step(X: jnp.array, m: float, s: float):
    X = jnp.abs(X - m)
    X = (X <= s) * 2 - 1

    return X


growth_func = {0: poly_quad4, 1: gaussian, 2: step}
