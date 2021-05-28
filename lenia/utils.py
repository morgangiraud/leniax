import os
import json
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from fractions import Fraction
from PIL import Image
from typing import List, Callable, Dict

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')
image_ext = "png"

DIM_DELIM = {0: '', 1: '$', 2: '%', 3: '#', 4: '@A', 5: '@B', 6: '@C', 7: '@D', 8: '@E', 9: '@F'}


###
# Loader
###
def st2fracs2float(st: str) -> List[float]:
    return [float(Fraction(st)) for st in st.split(',')]


def append_stack(list1: List, list2: List, count, is_repeat=False):
    list1.append(list2)
    if count != '':
        repeated = list2 if is_repeat else []
        list1.extend([repeated] * (int(count) - 1))


def recur_get_max_lens(dim, list1, max_lens, nb_dims: int):
    max_lens[dim] = max(max_lens[dim], len(list1))
    if dim < nb_dims - 1:
        for list2 in list1:
            recur_get_max_lens(dim + 1, list2, max_lens, nb_dims)


def recur_cubify(dim, list1, max_lens, nb_dims: int):
    more = max_lens[dim] - len(list1)
    if dim < nb_dims - 1:
        list1.extend([[]] * more)
        for list2 in list1:
            recur_cubify(dim + 1, list2, max_lens, nb_dims)
    else:
        list1.extend([0] * more)


def ch2val(c: str) -> int:
    if c in '.b':
        return 0
    elif c == 'o':
        return 255
    elif len(c) == 1:
        return ord(c) - ord('A') + 1
    else:
        return (ord(c[0]) - ord('p')) * 24 + (ord(c[1]) - ord('A') + 25)


def rle2arr(init_cells_code: List[str], nb_dims: int, nb_channels: int) -> jnp.array:
    assert nb_channels == len(init_cells_code)

    all_cells = []
    all_max_lens = []
    for st in init_cells_code:
        stacks = [[] for dim in range(nb_dims)]
        last, count = '', ''
        delims = list(DIM_DELIM.values())
        st = st.rstrip('!') + DIM_DELIM[nb_dims - 1]
        for ch in st:
            if ch.isdigit():
                count += ch
            elif ch in 'pqrstuvwxy@':
                last = ch
            else:
                if last + ch not in delims:
                    append_stack(stacks[0], ch2val(last + ch) / 255, count, is_repeat=True)
                else:
                    dim = delims.index(last + ch)
                    for d in range(dim):
                        append_stack(stacks[d + 1], stacks[d], count, is_repeat=False)
                        stacks[d] = []
                    # print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
                last, count = '', ''

        cells = stacks[nb_dims - 1]
        max_lens = [0 for dim in range(nb_dims)]
        recur_get_max_lens(0, cells, max_lens, nb_dims)
        recur_cubify(0, cells, max_lens, nb_dims)

        if len(all_max_lens) == 0:
            all_max_lens = max_lens
        else:
            all_max_lens = np.max([all_max_lens, max_lens], axis=0)

        cells = cells
        all_cells.append(cells)

    for i, cells in enumerate(all_cells):
        recur_cubify(0, cells, all_max_lens, nb_dims)
        all_cells[i] = jnp.array(cells)
    animal_cells = jnp.asarray(all_cells)

    return animal_cells


###
# Animals
###
def merge_cells(cells: jnp.array, other_cells: jnp.array, offset: List[int] = None) -> jnp.array:
    # We ensure the animal and the world are compatible:
    # - same number of dims
    # - same number of channels
    assert len(cells.shape) == len(other_cells.shape)
    assert cells.shape[0] == other_cells.shape[0]

    if offset:
        assert len(cells.shape) == len(offset)
    else:
        offset = [0] * len(cells.shape)

    pads = []
    for i in range(len(cells.shape)):
        pad_start = (cells.shape[i] - other_cells.shape[i]) // 2 - offset[i]
        pad_end = cells.shape[i] - other_cells.shape[i] - pad_start

        pads.append((pad_start, pad_end))

    padded_other_cells = jnp.pad(other_cells, pads, mode='constant', constant_values=(0, 0))

    cells += padded_other_cells

    return cells


###
# IMAGE
###
MARKER_COLORS_W = [0x5F, 0x5F, 0x5F, 0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF]
MARKER_COLORS_B = [0x9F, 0x9F, 0x9F, 0x7F, 0x7F, 0x7F, 0x0F, 0x0F, 0x0F]


def save_image(
    save_dir: str,
    cells: jnp.array,
    vmin: float,
    vmax: float,
    pixel_size: int,
    pixel_border_size: int,
    colormap,
    idx: int,
    suffix: str = ""
):
    assert len(cells.shape) == 3, 'we only handle images under the format [C, H, W]'
    assert cells.shape[0] < 4, 'we only handle images with less than 3 channels'

    norm_cells = jnp.clip(normalize(cells, vmin, vmax), 0, 1)
    img = get_image(norm_cells, pixel_size, pixel_border_size, colormap)

    filename = f"{str(idx).zfill(5)}{suffix}.{image_ext}"
    fullpath = os.path.join(save_dir, filename)

    img.save(fullpath)


def get_image(cells_buffer: jnp.array, pixel_size: int, pixel_border_size: int, colormap):
    _, y, x = cells_buffer.shape
    cells_buffer = jnp.repeat(cells_buffer, pixel_size, axis=1)
    cells_buffer = jnp.repeat(cells_buffer, pixel_size, axis=2)
    # zero = np.uint8(np.clip(normalize(0, vmin, vmax), 0, 1) * 252)
    zero = 0
    for i in range(pixel_border_size):
        cells_buffer[:, i::pixel_size, :] = zero
        cells_buffer[:, :, i::pixel_size] = zero

    img = colormap(cells_buffer)  # the colormap is applied to each channel, we just merge them
    blank_img = np.zeros_like(img[0, :, :, :3]).astype(jnp.uint8)
    blank_img = Image.fromarray(blank_img)
    for i in range(img.shape[0]):
        other_img = Image.fromarray((img[i, :, :, :3] * 255).astype(jnp.uint8))
        if i == 0:
            blank_img = Image.blend(blank_img, other_img, alpha=1.)
        else:
            blank_img = Image.blend(blank_img, other_img, alpha=0.5)

    return blank_img


def normalize(
    v: jnp.array, vmin: float, vmax: float, is_square: bool = False, vmin2: float = 0, vmax2: float = 0
) -> jnp.array:
    if not is_square:
        return (v - vmin) / (vmax - vmin)
    else:
        return (v - vmin) / max(vmax - vmin, vmax2 - vmin2)


def plot_function(save_di: str, id: int, fn: Callable, m: float, s: float):
    x = jnp.linspace(-0.5, 1.5, 500)

    y = fn(x, m, s)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')

    plt.plot(x, y, 'r', label='Growth function')

    plt.legend(loc='upper left')

    fullpath = os.path.join(save_dir, f"growth_function{str(id).zfill(2)}.{image_ext}")
    plt.savefig(fullpath)


###
# Others
###
def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif not os.path.isdir(dir):
        raise Exception('The path provided ({}) exist and is not a dir, aborting'.format(dir))


def save_config(save_dir: str, config: Dict):
    check_dir(save_dir)

    if type(config['run_params']['cells']) is not list:
        config['run_params']['cells'] = config['run_params']['cells'].tolist()
    with open(os.path.join(save_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)


###
# Random
###
def seed_everything(seed: int) -> int:
    rng_key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    return rng_key


def generate_noise_using_numpy(nb_noise: int, nb_channels: int, rng_key):
    sizes_np = np.hstack([
        np.array([nb_channels] * nb_noise)[:, np.newaxis], np.random.randint(2**3, 2**6, [nb_noise, 2])
    ])
    max_h_np = np.max(sizes_np[:, 1])
    max_w_np = np.max(sizes_np[:, 2])

    all_masks = []
    for i in range(nb_noise):
        shape = sizes_np[i]
        mask = generate_mask(shape, max_h_np, max_w_np)
        all_masks.append(mask)
    masks = jnp.array(np.vstack(mask))

    key, subkey = jax.random.split(rng_key)
    noise = jax.random.uniform(subkey, (nb_noise, 1, max_h_np, max_w_np), minval=0, maxval=1.)

    noises = noise * masks

    return key, noises


def generate_mask(shape, max_h, max_w):
    mask = np.ones(shape)

    pad_h_before = (max_h - mask.shape[1]) // 2
    pad_h_after = max_h - mask.shape[1] - pad_h_before
    pad_w_before = (max_w - mask.shape[2]) // 2
    pad_w_after = max_w - mask.shape[2] - pad_w_before
    padding = [(0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    mask = np.pad(mask, padding, mode='constant')

    return mask
