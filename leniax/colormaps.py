import json
import numpy as np
import jax.numpy as jnp
from matplotlib.colors import ListedColormap
from typing import List, Callable
from hilbertcurve.hilbertcurve import HilbertCurve

# This might be cool? https://bottosson.github.io/misc/colorpicker/#ce7d96


class LeniaColormap():
    def __init__(self, name: str, hex_bg_color: str, hex_colors: List[str]) -> None:
        self.name = name
        self.hex_bg_color = hex_bg_color
        self.hex_colors = hex_colors
        self.cmap = ListedColormap(hex_to_palette_rgba(hex_bg_color, hex_colors))

    def __call__(self, data):
        return self.cmap(data)

    def save(self) -> str:
        return json.dumps({
            'name': self.name,
            'hex_bg_color': self.hex_bg_color,
            'hex_colors': self.hex_colors,
        })

    @staticmethod
    def load(self, json_string: str):
        raw_obj = json.loads(json_string)

        return LeniaColormap(raw_obj['name'], raw_obj['hex_bg_color'], raw_obj['hex_colors'])

    def print_uint8_rgb_colors(self):
        print(jnp.array(jnp.array(self.cmap.colors) * 255, dtype=jnp.int32)[:, :3].tolist())


class Hilbert2d3dColormap():
    def __init__(self, name: str, p_pixels: int = 7, p_colors: int = 12):
        self.name = name

        H = 2**p_pixels
        W = 2**p_pixels
        self.nb_pixels = H * W
        self.mapping = np.zeros([H, W, 4])

        self.nb_colors = (2**p_colors)**2

        self.pixel_colors_step_size = (2**(p_colors - p_pixels))**2

        self.hilbert2d = HilbertCurve(p_pixels, 2)
        self.hilbert3d = HilbertCurve(p_colors, 3)

        self.populate()

    def populate(self):
        for i in range(self.nb_pixels):
            rgb = self.hilbert3d.point_from_distance(self.pixel_colors_step_size * i)
            px_pos = self.hilbert2d.point_from_distance(i)

            self.mapping[px_pos[0], px_pos[1]] = np.array(list(rgb) + [255]) / 255

    def __call__(self, data):
        assert data.shape[-1] == 2
        ori_shape = data.shape
        data = data.reshape(-1, 2)

        max_h = self.mapping.shape[0] - 1
        max_w = self.mapping.shape[1] - 1

        data[:, 0] = (data[:, 0] * max_h).round()
        data[:, 1] = (data[:, 1] * max_w).round()
        data = data.astype(np.uint8)

        data = self.mapping[data[:, 0], data[:, 1]]

        return data.reshape(list(ori_shape[:-1]) + [1, 4])


class ExtendedColormap():
    def __init__(self, name: str):
        self.name = name

    def __call__(self, data):
        c = data.shape[-1]
        if c == 1:
            g_layer = np.zeros(list(data.shape[:-1]) + [1])
            b_layer = np.zeros(list(data.shape[:-1]) + [1])
            a_layer = np.ones(list(data.shape[:-1]) + [1])

            data = np.concatenate([data, g_layer, b_layer, a_layer], axis=-1)
        elif c == 2:
            b_layer = np.zeros(list(data.shape[:-1]) + [1])
            a_layer = np.ones(list(data.shape[:-1]) + [1])

            data = np.concatenate([data, b_layer, a_layer], axis=-1)
        elif c == 3:
            a_layer = np.ones(list(data.shape[:-1]) + [1])

            data = np.concatenate([data, a_layer], axis=-1)
        else:
            raise ValueError(f"This colormap can't handle more than 3 channels. Current value {c}")

        return np.expand_dims(data, axis=-2)


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


def calculate_luminance(color_code: int) -> float:
    index = float(color_code) / 255

    if index < 0.03928:
        return index / 12.92
    else:
        return ((index + 0.055) / 1.055)**2.4


def calculate_relative_luminance(rgb: List[int]) -> float:
    return 0.2126 * calculate_luminance(rgb[0]) + 0.7152 * calculate_luminance(rgb[1]
                                                                               ) + 0.0722 * calculate_luminance(rgb[2])


def check_ratio(rgb1: List[int], rgb2: List[int]) -> int:
    light = rgb1 if sum(rgb1) > sum(rgb2) else rgb2
    dark = rgb1 if sum(rgb1) < sum(rgb2) else rgb2

    contrast_ratio = (calculate_relative_luminance(light) + 0.05) / (calculate_relative_luminance(dark) + 0.05)

    if contrast_ratio < 4.5:
        return 0
    if contrast_ratio >= 4.5 and contrast_ratio < 7:
        return 1
    else:  # contrast_ratio >= 7
        return 2


#####


def hex_to_rgba_uint8(hex: str) -> List[int]:
    hex = hex.replace('#', '')
    return [int(hex[i:i + 2], 16) for i in (0, 2, 4)] + [255]


def hex_to_palette_rgba(hex_bg_color: str, hex_colors: List[str]) -> jnp.ndarray:
    steps = 254 // (len(hex_colors) - 1)
    palette_rgb_uint8 = []
    for i in range(0, len(hex_colors) - 1):
        rgb1_uint8 = hex_to_rgba_uint8(hex_colors[i])[:3]
        rgb2_uint8 = hex_to_rgba_uint8(hex_colors[i + 1])[:3]
        palette_rgb_uint8 += perceptual_steps(rgb1_uint8, rgb2_uint8, steps)
    fg_palette_rgba_uint8 = [rgb + [255] for rgb in palette_rgb_uint8]

    # Transparent background
    if hex_bg_color == '':
        bg_rgba_uint8 = [0, 0, 0, 0]
    else:
        bg_rgba_uint8 = hex_to_rgba_uint8(hex_bg_color)

    bg_rgba = jnp.array([bg_rgba_uint8]) / 255.
    fg_palette_rgba = jnp.array(fg_palette_rgba_uint8) / 255.
    palette_rgba = jnp.vstack([bg_rgba, fg_palette_rgba])

    return palette_rgba


colormaps = {
    'alizarin':
    LeniaColormap(
        'alizarin', "d6c3c9", ['f9c784', 'e7e7e7', '485696', '19180a', '3f220f', '772014', 'af4319', 'e71d36']
    ),
    'black-white':
    LeniaColormap(
        'black-white', '000000', ['ffffff', 'd9dbe1', 'b6b9c1', '9497a1', '737780', '555860', '393b41', '1f2123'][::-1]
    ),
    'carmine-blue':
    LeniaColormap('carmine-blue', '#006eb8', ['#006eb8', '#fff200', '#cc1236']),
    'cinnamon':
    LeniaColormap('cinnamon', '#a7d4e4', ['#a7d4e4', '#71502f', '#fdc57e']),
    'city':
    LeniaColormap(
        'city', 'F93943', ['ffa600', 'fff6e6', 'ffca66', '004b63', 'e6f9ff', '66daff', '3a0099', '23005c'][::-1]
    ),
    'golden':
    LeniaColormap('golden', '#b6bfc1', ['#b6bfc1', '#253122', '#f3a257']),
    'laurel':
    LeniaColormap(
        'laurel', '381d2a', ['ffbfd7', 'ffe6ef', 'ff80b0', '71bf60', 'eaffe6', '96ff80', 'bffbff', '60b9bf'][::-1]
    ),
    'msdos':
    LeniaColormap('msdos', '#0c0786', ['#0c0786', '#7500a8', '#c03b80', '#f79241', '#fcfea4']),
    'pink-beach':
    LeniaColormap(
        'pink-beach', 'f4777f', ['00429d', '4771b2', '73a2c6', 'a5d5d8', 'ffffe0', 'ffbcaf', 'cf3759', '93003a'][::-1]
    ),
    'rainbow':
    LeniaColormap('rainbow', '#000000', ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#2E2B5F', '#8B00FF']),
    'rainbow_transparent':
    LeniaColormap(
        'rainbow_transparent', '', ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#2E2B5F', '#8B00FF']
    ),
    'river-Leaf':
    LeniaColormap(
        'river-Leaf', "80ab82", ["4c5b5c", "ff715b", "f9cb40", "bced09", "2f52e0", "99f7ab", "c5d6d8", "7dcd85"][::-1]
    ),
    'salvia':
    LeniaColormap('salvia', '#b6bfc1', ['#b6bfc1', '#051230', '#97acc8']),
    'summer':
    LeniaColormap(
        'summer', 'ffe000', ['003dc7', '002577', 'e6edff', '6695ff', 'ff9400', '995900', 'fff4e6', 'ffbf66'][::-1]
    ),
    'white-black':
    LeniaColormap('white-black', '#ffffff', ['#ffffff', '#000000'])
}
