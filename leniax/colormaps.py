import jax.numpy as jnp
from matplotlib.colors import ListedColormap
from typing import List, Callable

# An interesting idea from https://iquilezles.org/www/articles/palettes/palettes.htm


class LeniaColormap():
    def __init__(self, name, hex_bg_color, hex_colors) -> None:
        self.name = name
        self.cmap = ListedColormap(hex_color(hex_bg_color, hex_colors))

    def __call__(self, data):
        return self.cmap(data)


#######


def all_channels(func: Callable) -> Callable:
    def wrapper(channel, *args, **kwargs):
        try:
            return func(channel, *args, **kwargs)
        except TypeError:
            return list(func(c, *args, **kwargs) for c in channel)

    return wrapper


def all_channels2(func: Callable) -> Callable:
    def wrapper(channel1, channel2, *args, **kwargs):
        try:
            return func(channel1, channel2, *args, **kwargs)
        except TypeError:
            return list(func(c1, c2, *args, **kwargs) for c1, c2 in zip(channel1, channel2))

    return wrapper


@all_channels
def to_sRGB_f(x: float) -> float:
    ''' Returns a sRGB value in the range [0,1]
        for linear input in [0,1].
    '''
    return 12.92 * x if x <= 0.0031308 else (1.055 * (x**(1 / 2.4))) - 0.055


@all_channels
def to_sRGB(x: float) -> int:
    ''' Returns a sRGB value in the range [0,255]
        for linear input in [0,1]
    '''
    return int(255.9999 * to_sRGB_f(x))


@all_channels
def from_sRGB(x: int) -> float:
    ''' Returns a linear value in the range [0,1]
        for sRGB input in [0,255].
    '''
    x_f = x / 255.0
    if x_f <= 0.04045:
        y = x_f / 12.92
    else:
        y = ((x_f + 0.055) / 1.055)**2.4
    return y


@all_channels2
def lerp(color1: float, color2: float, frac: float) -> float:
    return color1 * (1 - frac) + color2 * frac


def perceptual_steps(color1: List[int], color2: List[int], steps: int) -> List[List[int]]:
    gamma = .43
    color1_lin = from_sRGB(color1)
    bright1 = sum(color1_lin)**gamma
    color2_lin = from_sRGB(color2)
    bright2 = sum(color2_lin)**gamma

    colors = []
    for step in range(steps):
        intensity = lerp(bright1, bright2, step / steps)**(1 / gamma)
        color = lerp(color1_lin, color2_lin, step / steps)
        if sum(color) != 0:
            color = [c * intensity / sum(color) for c in color]
        color = to_sRGB(color)
        colors.append(color)

    return colors


#####


def hex_to_rgba(hex: str) -> List[float]:
    hex = hex.replace('#', '')
    return [int(hex[i:i + 2], 16) / 255. for i in (0, 2, 4)] + [1.]


def hex_color(hex_bg_color: str, hex_colors: List[str]) -> jnp.ndarray:
    steps = 254 // (len(hex_colors) - 1)
    rgb_colors = []
    for i in range(0, len(hex_colors) - 1):
        rgb1 = [int(c * 255) for c in hex_to_rgba(hex_colors[i])[:3]]
        rgb2 = [int(c * 255) for c in hex_to_rgba(hex_colors[i + 1])[:3]]
        rgb_colors += perceptual_steps(rgb1, rgb2, steps)
    rgba_colors = [rgb_color + [255] for rgb_color in rgb_colors]
    rgba_bg_color = hex_to_rgba(hex_bg_color)

    bg_colors = jnp.array([rgba_bg_color])
    colors = jnp.array(rgba_colors) / 255.
    all_colors = jnp.vstack([bg_colors, colors])

    return all_colors


colormaps = {
    'blackwhite': LeniaColormap(
        'blackwhite',  # Name
        '#000000',  # Background color
        ['#000000', '#ffffff'],  # Color palette
    ),
    'grenadine': LeniaColormap(
        'grenadine',  # Name
        '#051230',  # Background color
        ['#051230', '#f48067'],  # Color palette
    ),
    'olive': LeniaColormap(
        'olive',  # Name
        '#112f2c',  # Background color
        ['#112f2c', '#f37420', '#d6b43e'],  # Color palette
    ),
    'carmine-blue': LeniaColormap(
        'carmine-blue',
        '#006eb8',
        ['#006eb8', '#fff200', '#cc1236'],
    ),
    'carmine-green': LeniaColormap(
        'carmine-green',
        '#1a7444',
        ['#1a7444', '#fff200', '#cc1236'],
    ),
    'red': LeniaColormap(
        'red',
        '#1a7444',
        ['#1a7444', '#fff200', '#a72144'],
    ),
    'vistoris-violet': LeniaColormap(
        'vistoris-violet',
        '#4f4086',
        ['#4f4086', '#fdbf68', '#6d4145'],
    ),
    'vistoris-green': LeniaColormap(
        'vistoris-green',
        '#555832',
        ['#555832', '#ffefae', '#6d4145'],
    ),
    'cinnamon': LeniaColormap(
        'cinnamon',
        '#a7d4e4',
        ['#a7d4e4', '#71502f', '#fdc57e'],
    ),
    'golden': LeniaColormap(
        'golden',
        '#b6bfc1',
        ['#b6bfc1', '#253122', '#f3a257'],
    ),
    'ochraceous': LeniaColormap(
        'ochraceous',
        '#a1a39a',
        ['#a1a39a', '#501345', '#d8a37b'],
    ),
    'lemon-turquoise': LeniaColormap(
        'lemon-turquoise',
        '#b5decc',
        ['#b5decc', '#762c19', '#f8ed43'],
    ),
    'lemon-venice': LeniaColormap(
        'lemon-venice',
        '#62c6bf',
        ['#62c6bf', '#253122', '#f8ed43'],
    ),
    'salvia': LeniaColormap(
        'salvia',
        '#b6bfc1',
        ['#b6bfc1', '#051230', '#97acc8'],
    ),
    'rainbow': LeniaColormap(
        'rainbow',
        '#000000',
        ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#2E2B5F', '#8B00FF'],
    ),
}
