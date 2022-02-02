"""Leniax growth functions"""

import jax.numpy as jnp


def poly_quad4(params: jnp.ndarray, X: jnp.ndarray):
    r"""Quadratic polynomial growth function

    .. math::
        y = 2 * \max{
            \left[
                1 - \left(\frac{X - m}{3s}\right)^2, 0
            \right]
        }^4 - 1

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.growth_functions import poly_quad4
        x = np.linspace(0, 1, 1000)
        m = 0.3
        s = 0.05
        params = np.array([m, s])
        y = poly_quad4(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Poly_quad4: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: parameters of the growth function
        X: potential

    Returns:
        A field
    """
    m = params[0]
    s = params[1]

    out = 1 - (X - m)**2 / (9 * s**2)
    out = jnp.maximum(0, out)
    out = out**4

    return 2 * out - 1


def gaussian(params: jnp.ndarray, X: jnp.ndarray):
    r"""Gaussian growth function

    .. math::
        y = 2 * e^{-\frac{1}{2} \left(\frac{X - m}{s}\right)^2} - 1

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.growth_functions import gaussian
        x = np.linspace(0, 1, 1000)
        m = 0.3
        s = 0.05
        params = np.array([m, s])
        y = gaussian(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Gaussian: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: parameters of the growth function
        X: potential

    Returns:
        A field
    """
    m = params[0]
    s = params[1]

    out = ((X - m) / s)**2
    out = jnp.exp(-out / 2)

    return 2 * out - 1


def gaussian_target(params: jnp.ndarray, X: jnp.ndarray):
    r"""Gaussian growth function for aymptotic Lenia

    .. math::
        y = e^{-\frac{1}{2} \left(\frac{X - m}{s}\right)^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.growth_functions import gaussian_target
        x = np.linspace(0, 1, 1000)
        m = 0.3
        s = 0.05
        params = np.array([m, s])
        y = gaussian_target(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Gaussian target: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: parameters of the growth function
        X: potential

    Returns:
        A field
    """
    m = params[0]
    s = params[1]

    out = ((X - m) / s)**2
    out = jnp.exp(-out / 2)

    return out


def step(params: jnp.ndarray, X: jnp.ndarray):
    r"""Step growth function

    .. math::
        y =
        \begin{cases}
            1 ,& \text{if } |X - m| \leq s\\
            -1,& \text{otherwise}
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.growth_functions import step
        x = np.linspace(0, 1, 1000)
        m = 0.3
        s = 0.05
        params = np.array([m, s])
        y = step(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Step: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: parameters of the growth function
        X: potential

    Returns:
        A field
    """
    m = params[0]
    s = params[1]

    out = jnp.abs(X - m)
    out = (out <= s)

    return 2 * out - 1


def staircase(params: jnp.ndarray, X: jnp.ndarray):
    r"""Staircase function

    .. math::
        y =
        \begin{cases}
            0.5 ,& \text{if } X \geq m - s \text{ and } X < m - \frac{s}{2}\\
            1 ,& \text{if } X \geq m - \frac{s}{2} \text{ and } X \leq m + \frac{s}{2}\\
            0.5 ,& \text{if } X > m + \frac{s}{2} \text{ and } X \leq m + s\\
            0 ,& \text{otherwise}
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import staircase
        x = np.linspace(0., 1., 100)
        m = 0.5
        s = 0.1
        params = np.array([m, s])
        y = staircase(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Staircase: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    m = params[0]
    s = params[1]

    out = 0.5 * (X >= m - s) * (X < m - s / 2)
    out += 1 * (X >= m - s / 2) * (X <= m + s / 2)
    out += 0.5 * (X > m + s / 2) * (X <= m + s)

    return out


def triangle(params: jnp.ndarray, X: jnp.ndarray):
    r"""Gauss function

    .. math::
        y =
        \begin{cases}
            \frac{X - (m - s)}{m - (m - s)} ,& \text{if } X \geq m - s \text{ and } X < m\\
            \frac{X - (m + s)}{m - (m + s)} ,& \text{if } X \geq m \text{ and } X \leq m + s\\
            0 ,& \text{otherwise}
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from leniax.kernel_functions import triangle
        x = np.linspace(0., 1., 100)
        m = 0.5
        s = 0.1
        params = np.array([m, s])
        y = triangle(params, x)
        plt.plot(x, y)
        plt.axhline(y=0, color='grey', linestyle='dotted')
        plt.title(r'Triangle: m=%.2f, s=%.2f'%(m, s))
        plt.show()

    Args:
        params: Kernel function parameters
        X: Raw kernel
    """
    m = params[0]
    s = params[1]

    left = m - s
    right = m + s

    out = (X >= left) * (X < m) * (X - left) / (m - left)
    out += (X >= m) * (X <= right) * (X - right) / (m - right)

    return out


register = {
    'poly_quad4': poly_quad4,
    'gaussian': gaussian,
    'gaussian_target': gaussian_target,
    'step': step,
    'staircase': staircase,
    'triangle': triangle,
}
