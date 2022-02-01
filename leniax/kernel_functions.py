from typing import Dict, Callable
import jax.numpy as jnp

from .constant import EPSILON


def poly_quad4(params: jnp.ndarray, X: jnp.ndarray):
    r"""Quadratic polynomial

    .. math::
        y = (4 * X * (1 - X))^q

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import poly_quad4
        x = np.linspace(0., 1., 100)
        q = 4
        params = np.array([q])
        y = poly_quad4(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Poly_quad4: q=%.2f'%(q))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """

    q = params[0]

    out = 4 * X * (1 - X)
    out = out**q

    return out


def gauss_bump(params: jnp.ndarray, X: jnp.ndarray):
    r"""Gaussian bump function

    .. math::
        y = e^{q * [q - \frac{1}{X * (1 - X)}]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import gauss_bump
        x = np.linspace(0., 1., 100)
        q = 0.15
        params = np.array([q])
        y = gauss_bump(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Gauss bump: q=%.2f'%(q))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    q = params[0]

    out = q - 1 / (X * (1 - X) + EPSILON)
    out = jnp.exp(q * out)

    return out


def step(params: jnp.ndarray, X: jnp.ndarray):
    r"""Step function

    .. math::
        y =
        \begin{cases}
            1 ,& \text{if } X \geq q \text{ and } X \leq 1 - q\\
            0 ,& \text{otherwise}
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import step
        x = np.linspace(0., 1., 100)
        q = 0.3
        params = np.array([q])
        y = step(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Step: q=%.2f'%(q))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    q = params[0]

    return (X >= q) * (X <= 1 - q)


def staircase(params: jnp.ndarray, X: jnp.ndarray):
    r"""Staircase function

    .. math::
        y =
        \begin{cases}
            0.5 ,& \text{if } X \le q\\
            1 ,& \text{if } X \geq q \text{ and } X \leq 1 - q\\
            0 ,& \text{otherwise}
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import staircase
        x = np.linspace(0., 1., 100)
        q = 0.3
        params = np.array([q])
        y = staircase(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Staircase: q=%.2f'%(q))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    q = params[0]

    return (X >= q) * (X <= 1 - q) + (X >= 0) * (X < q) * 0.5


def gauss(params: jnp.ndarray, X: jnp.ndarray):
    r"""Gauss function

    .. math::
        y = e^{-\frac{1}{2} \left(\frac{X - q}{0.3 * q}\right)^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import gauss
        x = np.linspace(0., 1., 100)
        q = 0.15
        params = np.array([q])
        y = gauss(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Gauss: q=%.2f'%(q))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    q = params[0]

    out = ((X - q) / (0.3 * q))**2
    out = jnp.exp(-out / 2)

    return out


register: Dict[str, Callable] = {
    'poly_quad4': poly_quad4, 'gauss_bump': gauss_bump, 'step': step, 'staircase': staircase, 'gauss': gauss
}
