import functools
import jax.numpy as jnp
from matplotlib.colors import ListedColormap
from typing import List, Dict

# An interesting idea from https://iquilezles.org/www/articles/palettes/palettes.htm


class LeniaTemporalColormap():
    def __init__(self, name) -> None:
        self.name = name
        self.framerate = 32
        self.cmap_generator = get_cmap_generator(name)(self.framerate)

    def __call__(self, data):
        cmap = next(self.cmap_generator)
        return cmap(data)


linspace_255 = jnp.linspace(0, 1, 255)[:, jnp.newaxis]

colormaps_coefs: Dict[str, List[jnp.ndarray]] = {
    'earth': [
        jnp.array([0.8, 0.5, 0.4])[jnp.newaxis],
        jnp.array([0.2, 0.4, 0.2])[jnp.newaxis],
        jnp.array([2.0, 1.0, 1.0])[jnp.newaxis],
        jnp.array([0.00, 0.25, 0.25])[jnp.newaxis],
    ],
}


def color(t: float, coefs: List[jnp.ndarray]):
    rgb = coefs[0] + coefs[1] * jnp.cos(2 * jnp.pi * (t * coefs[2] + coefs[3]))
    t0 = jnp.array([[0.]])
    rgb_zero = coefs[0] + coefs[1] * jnp.cos(2 * jnp.pi * (t0 * coefs[2] + coefs[3]))
    full_rgb = jnp.vstack([rgb_zero, rgb])

    alpha_channel = jnp.ones((full_rgb.shape[0], 1))
    colors = jnp.hstack([full_rgb, alpha_channel])

    return colors


def get_cmap_generator(name: str):
    if name not in colormaps_coefs:
        raise ValueError(f"Colormap {name} does not exist")

    coefs = colormaps_coefs[name]
    color_palette = functools.partial(color, coefs=coefs)

    def cmap_generator(framerate: int = 32):
        i = 0
        while True:
            current_t = (linspace_255 + i / framerate) % 1
            yield ListedColormap(color_palette(current_t))
            i += 1

    return cmap_generator
