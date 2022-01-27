from typing import Dict, Callable
import jax.numpy as jnp


def poly_quad4(params: jnp.ndarray, X: jnp.ndarray):
    r"""Quadratic polynomial growth function

    .. math::
        y = 2 * \max{
            \left[
                (4 * X * (1 - X))^q
            \right]
        }^4 - 1

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import poly_quad4
        x = np.linspace(0, 1, 1000)
        q = 4
        params = np.array([q])
        y = poly_quad4(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Poly_quad4: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: parameters of the kernel function
        X: raw kernel
    """

    q = params[0]

    out = 4 * X * (1 - X)
    out = out**q

    return out


def gauss_bump(params: jnp.ndarray, X: jnp.ndarray):
    q = params[0]

    out = q - 1 / (X * (1 - X))
    out = jnp.exp(q * out)

    return out


def step4(params: jnp.ndarray, X: jnp.ndarray):
    q = params[0]

    return (X >= q) * (X <= 1 - q)


def staircase(params: jnp.ndarray, X: jnp.ndarray):
    q = params[0]

    return (X >= q) * (X <= 1 - q) + (X < q) * 0.5


def gauss(params: jnp.ndarray, X: jnp.ndarray):
    q = params[0]

    out = ((X - q) / (0.3 * q))**2
    out = jnp.exp(-out / 2)

    return out


register: Dict[str, Callable] = {
    'poly_quad4': poly_quad4, 'gauss_bump': gauss_bump, 'step4': step4, 'staircase': staircase, 'gauss': gauss
}
