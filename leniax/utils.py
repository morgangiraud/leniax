# import time
import functools
import io
import binascii
import copy
import itertools
import random
import os
import codecs
import pickle
import yaml
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any

from .constant import NB_STATS_STEPS, EPSILON, NB_CHARS

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')
image_ext = "png"


###
# Loader
###
def append_stack(list1: List, elem: Any, count, is_repeat=False):
    list1.append(elem)
    if count != '':
        repeated = elem if is_repeat else []
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


def _recur_drill_list(dim, lists, row_func, nb_dims: int):
    if dim < nb_dims - 1:
        return [_recur_drill_list(dim + 1, e, row_func, nb_dims) for e in lists]
    else:
        return row_func(lists)


def _recur_join_st(dim, lists, row_func, nb_dims: int):
    if dim < nb_dims - 1:
        return DIM_DELIM[nb_dims - 1 - dim].join(_recur_join_st(dim + 1, e, row_func, nb_dims) for e in lists)
    else:
        return DIM_DELIM[nb_dims - 1 - dim].join(row_func(lists))


def make_array_compressible(cells: jnp.ndarray) -> jnp.ndarray:
    max_val = NB_CHARS**2 - 1

    cells_int32 = jnp.array(jnp.round(cells * max_val), dtype=jnp.int32)
    compressible_cells = jnp.array(cells_int32 / max_val, dtype=jnp.float32)

    return compressible_cells


def compress_array(cells: jnp.ndarray) -> str:
    max_val = NB_CHARS**2 - 1
    cells_int32 = cells_int32 = jnp.array(jnp.round(cells * max_val), dtype=jnp.int32)
    cells_shape = cells_int32.shape

    cells_char_l = [val2ch(item) for item in cells_int32.flatten()]
    string_cells = "".join(cells_char_l)
    string_cells += '::' + ';'.join([str(c) for c in cells_shape])

    return string_cells


def decompress_array(string_cells: str, nb_dims: int) -> jnp.ndarray:
    try:
        string_array = string_cells.split('::')
        if len(string_array) != 2 and len(string_array[0]) % 2 == 0:
            raise Exception()
        max_val = NB_CHARS**2 - 1
        cells_shape = [int(c) for c in string_array[-1].split(";")]
        cells_val_l = [ch2val(string_array[0][i:i + 2]) for i in range(0, len(string_array[0]), 2)]
        cells_int32 = jnp.array(cells_val_l, dtype=jnp.int32).reshape(cells_shape)
        cells = jnp.array(cells_int32 / max_val, dtype=jnp.float32)
    except Exception:
        try:
            string_bytes = io.BytesIO(string_cells.encode('latin1'))
            cells_uint8 = np.load(string_bytes)['x']  # type: ignore
            cells = cells_uint8 / 255.
        except Exception:
            try:
                cells = decompress_array_base64(string_cells)
            except binascii.Error:
                cells = deprecated_decompress_array(string_cells, nb_dims)

    return jnp.array(cells, dtype=jnp.float32)


def decompress_array_base64(string_cells: str) -> jnp.ndarray:
    serialized_cells = codecs.decode(string_cells.encode(), "base64")
    cells = pickle.loads(serialized_cells)

    return jnp.array(cells, dtype=jnp.float32)


def ch2val(c: str) -> int:
    assert len(c) == 2
    first_char = c[0]
    second_char = c[1]

    if ord(first_char) >= ord('À'):
        first_char_idx = ord(first_char) - ord('À') + (ord('Z') - ord('A')) + (ord('z') - ord('a'))
    elif ord(first_char) >= ord('a'):
        first_char_idx = ord(first_char) - ord('a') + (ord('Z') - ord('A'))
    else:
        first_char_idx = ord(first_char) - ord('A')

    if ord(second_char) >= ord('À'):
        second_char_idx = ord(second_char) - ord('À') + (ord('Z') - ord('A')) + (ord('z') - ord('a'))
    elif ord(second_char) >= ord('a'):
        second_char_idx = ord(second_char) - ord('a') + (ord('Z') - ord('A'))
    else:
        second_char_idx = ord(second_char) - ord('A')

    return first_char_idx * NB_CHARS + second_char_idx


def val2ch(v: int) -> str:
    first_char_idx = v // NB_CHARS
    second_char_idx = v % NB_CHARS
    # We do this trick to avoid the spaecial characters between
    if ord('A') + first_char_idx <= ord('Z'):
        first_char = chr(ord('A') + first_char_idx)
    elif ord('a') + first_char_idx - (ord('Z') - ord('A')) <= ord('z'):
        first_char = chr(ord('a') + first_char_idx - (ord('Z') - ord('A')))
    else:
        first_char = chr(ord('À') + first_char_idx - (ord('Z') - ord('A')) - (ord('z') - ord('a')))

    if ord('A') + second_char_idx <= ord('Z'):
        second_char = chr(ord('A') + second_char_idx)
    elif ord('a') + second_char_idx - (ord('Z') - ord('A')) <= ord('z'):
        second_char = chr(ord('a') + second_char_idx - (ord('Z') - ord('A')))
    else:
        second_char = chr(ord('À') + second_char_idx - (ord('Z') - ord('A')) - (ord('z') - ord('a')))

    return first_char + second_char


###
# Deprecated, kept for backward compatibilities
###
DIM_DELIM = {0: '', 1: '$', 2: '%', 3: '#', 4: '@A', 5: '@B', 6: '@C', 7: '@D', 8: '@E', 9: '@F'}


def char2val_deprecated(ch: str) -> int:
    if ch in '.b':
        return 0
    elif ch == 'o':
        return 255
    elif len(ch) == 1:
        return ord(ch) - ord('A') + 1
    else:
        return (ord(ch[0]) - ord('p')) * 24 + (ord(ch[1]) - ord('A') + 25)


def val2char_deprecated(v: int) -> str:
    if v == 0:
        return '.'
    elif v < 25:
        return chr(ord('A') + v - 1)
    else:
        return chr(ord('p') + (v - 25) // 24) + chr(ord('A') + (v - 25) % 24)


def deprecated_compress_array(cells):
    def drill_row(row):
        return [(len(list(g)), val2char_deprecated(v).strip()) for v, g in itertools.groupby(row)]

    def join_row_shorten(row):
        return [(str(n) if n > 1 else '') + c for n, c in row]

    nb_dims = len(cells.shape)
    cells_int_l = np.rint(cells * 255).astype(int).tolist()  # [[255 255] [255 0]]

    rle_groups = _recur_drill_list(0, cells_int_l, drill_row, nb_dims)
    st = _recur_join_st(0, rle_groups, join_row_shorten, nb_dims)  # "2 yO $ 1 yO"

    return st + '!'


def deprecated_decompress_array(cells_code: str, nb_dims: int) -> jnp.ndarray:
    stacks: List[List] = [[] for dim in range(nb_dims)]
    last, count = '', ''
    delims = list(DIM_DELIM.values())
    st = cells_code.rstrip('!') + DIM_DELIM[nb_dims - 1]
    for ch in st:
        if ch.isdigit():
            count += ch
        elif ch in 'pqrstuvwxy@':
            last = ch
        else:
            if last + ch not in delims:
                append_stack(stacks[0], char2val_deprecated(last + ch) / 255, count, is_repeat=True)
            else:
                dim = delims.index(last + ch)
                for d in range(dim):
                    append_stack(stacks[d + 1], stacks[d], count, is_repeat=False)
                    stacks[d] = []
                # print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
            last, count = '', ''

    cells_l = stacks[nb_dims - 1]
    max_lens = [0 for dim in range(nb_dims)]
    recur_get_max_lens(0, cells_l, max_lens, nb_dims)
    recur_cubify(0, cells_l, max_lens, nb_dims)

    cells = jnp.array(cells_l, dtype=jnp.float32)

    return cells


###
# Config
###
def get_param(dic: Dict, key_string: str) -> Any:
    keys = key_string.split('.')
    for key in keys:
        if key.isdigit():
            dic = dic[int(key)]
        else:
            dic = dic[key]

    if isinstance(dic, jnp.ndarray):
        val = float(jnp.mean(dic))
    else:
        val = dic

    return val


def set_param(dic: Dict, key_string: str, value: Any):
    keys = key_string.split('.')
    for i, key in enumerate(keys[:-1]):
        if key.isdigit():
            if isinstance(dic, list):
                idx = int(key)
                if len(dic) < idx + 1:
                    for _ in range(idx + 1 - len(dic)):
                        dic.append({})
                dic = dic[idx]
            else:
                raise Exception('This key should be an array')
        else:
            if keys[i + 1].isdigit():
                dic = dic.setdefault(key, [{}])
            else:
                dic = dic.setdefault(key, {})

    dic[keys[-1]] = value


def update_config_to_hd(ori_config: Dict) -> Dict:
    config = copy.deepcopy(ori_config)

    config['render_params']['pixel_size_power2'] = 0
    config['render_params']['pixel_size'] = 1
    config['render_params']['size_power2'] = 10
    config['render_params']['world_size'] = [1024, 1024]

    if 'cells' in config['run_params']:
        nb_dims = config['world_params']['nb_dims']
        cells = config['run_params']['cells']
        if type(cells) is str:
            if cells == 'last_frame.p':
                with open(os.path.join(config['main_path'], 'last_frame.p'), 'rb') as f:
                    cells = jnp.array(pickle.load(f))
            else:
                # Backward compatibility
                cells = decompress_array(cells, nb_dims + 1)  # we add the channel dim

        max_scaling = min(512 / cells.shape[1], 512 / cells.shape[2])

    config['world_params']['scale'] = min(10, round(max_scaling))

    return config


###
# Animals
###
def merge_cells(cells: jnp.ndarray, other_cells: jnp.ndarray, offset: List[int] = None) -> jnp.ndarray:
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
        pad_start = int(max((cells.shape[i] - other_cells.shape[i]) // 2 + offset[i], 0))
        pad_end = int(cells.shape[i] - other_cells.shape[i] - pad_start)

        pads.append((pad_start, pad_end))
    padded_other_cells = jnp.pad(other_cells, pads, mode='constant', constant_values=(0, 0))

    cells += padded_other_cells

    return cells


###
# Math
###
def get_unit_distances(world_size: List[int]) -> jnp.ndarray:
    midpoint = jnp.asarray([size // 2 for size in world_size])  # [nb_dims]
    midpoint = midpoint.reshape([-1] + [1] * len(world_size))  # [nb_dims, 1, 1, ...]
    coords = jnp.indices(world_size)  # [nb_dims, dim_0, dim_1, ...]
    centered_coords = coords - midpoint
    unit_coords = centered_coords / centered_coords.max()

    unit_distances = jnp.sqrt(jnp.sum(unit_coords**2, axis=0))

    return unit_distances


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
def center_world(cells: jnp.ndarray, field: jnp.ndarray, potential: jnp.ndarray, shift_idx: jnp.ndarray,
                 axes) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    cells = jnp.roll(cells, -shift_idx, axes)
    field = jnp.roll(field, -shift_idx, axes)
    potential = jnp.roll(potential, -shift_idx, axes)

    return cells, field, potential


def crop_zero(kernels: jnp.ndarray) -> jnp.ndarray:
    if len(kernels.shape) == 3:
        where_zero = kernels == 0
        kernels = kernels[:, ~jnp.all(where_zero, axis=(0, 2))]  # remove 0 columns
        cropped_kernels = kernels[:, :, ~jnp.all(where_zero, axis=(0, 1))]  # remove 0 lines
    elif len(kernels.shape) == 4:
        where_zero = kernels == 0
        kernels = kernels[:, ~jnp.all(where_zero, axis=(0, 2, 3))]  # remove 0 columns
        kernels = kernels[:, :, ~jnp.all(where_zero, axis=(0, 1, 3))]  # remove 0 lines
        cropped_kernels = kernels[:, :, :, ~jnp.all(where_zero, axis=(0, 1, 2))]  # remove 0 lines
    else:
        raise ValueError("Can't handle more than 3 dimensions")

    return cropped_kernels


def auto_center_cells(cells):
    world_size = cells.shape[1:]
    axes = tuple(range(-len(world_size), 0, 1))

    midpoint = jnp.expand_dims(jnp.asarray([size // 2 for size in world_size]), axis=axes)
    coords = jnp.indices(world_size)
    centered_coords = coords - midpoint

    m_00 = cells.sum()
    MX = [(cells * coord).sum() for coord in centered_coords]
    mass_centroid = jnp.array(MX) / (m_00 + EPSILON)

    shift_idx = jnp.round(mass_centroid).astype(int).T
    centered_cells, _, _ = center_world(
        cells[jnp.newaxis],
        cells[jnp.newaxis],
        cells[jnp.newaxis],
        shift_idx[jnp.newaxis],
        axes,
    )

    return centered_cells[0]


def center_and_crop_cells(cells):
    world_size = cells.shape[1:]
    axes = tuple(range(-len(world_size), 0, 1))
    midpoint = jnp.expand_dims(jnp.asarray([size // 2 for size in world_size]), axis=axes)

    pre_shift_idx = [0] * len(world_size)
    for i in range(0, len(world_size)):
        axes_to_colapse = list(range(len(world_size) + 1))
        axes_to_colapse.pop(i + 1)
        check_zero = ~jnp.all(cells == 0, axis=axes_to_colapse)
        if check_zero[0] and check_zero[-1] and not check_zero.prod():
            pre_shift_idx[i] = int(midpoint[0])

    cells, _, _ = center_world(
        cells[jnp.newaxis],  # Adding the batch axis
        cells[jnp.newaxis],
        cells[jnp.newaxis],
        jnp.array(pre_shift_idx)[jnp.newaxis],
        axes,
    )
    cells = cells[0]

    centered_cells = auto_center_cells(cells)
    cells = crop_zero(centered_cells)

    if 0 in cells.shape:
        cells = jnp.zeros([cells.shape[0]] + [1] * (len(cells.shape) - 1))

    return cells


###
# VIZ
###
def get_image(cells_buffer: np.ndarray, pixel_size: int, colormap) -> Image:
    nb_dims = len(cells_buffer.shape) - 1

    if pixel_size != 1:
        for i in range(nb_dims):
            cells_buffer = np.repeat(cells_buffer, pixel_size, axis=i + 1)

    if nb_dims == 2:
        cells_buffer = np.transpose(cells_buffer, (1, 2, 0))
        img = colormap(cells_buffer)[:, :, 0]  # the colormap is applied to each channel, we just merge them
        rgba_uint8 = np.uint8(img * 255)
        final_img = Image.fromarray(rgba_uint8, 'RGBA')
    elif nb_dims == 3:
        fig = plt.figure()
        ax = fig.gca()
        ax.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)
        colors = plt.get_cmap('viridis')(cells_buffer)
        colors[:, :, :, 3] = 0.5
        ax.voxels(cells_buffer, facecolors=colors)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        final_img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    else:
        raise ValueError("nb_dims > 3 not supported")

    return final_img


def normalize(v: jnp.ndarray, vmin: float, vmax: float, is_square: bool = False, vmin2: float = 0, vmax2: float = 0):
    if not is_square:
        return (v - vmin) / (vmax - vmin)
    else:
        return (v - vmin) / max(vmax - vmin, vmax2 - vmin2)


def plot_stats(save_dir: str, stats_dict: Dict):
    nb_steps = int(stats_dict['N'])

    all_keys = list(stats_dict.keys())
    del all_keys[all_keys.index('N')]

    ticks = max(nb_steps // 20, 1)
    fig, axs = plt.subplots(len(all_keys), sharex=True)
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        axs[i].set_title(k.capitalize())
        axs[i].plot(stats_dict[k][:nb_steps])
    plt.tight_layout()
    plt.xticks(np.arange(0, nb_steps, ticks))
    fig.savefig(os.path.join(save_dir, 'stats.png'))
    plt.close(fig)

    # Running means
    conv_len = min(32, nb_steps - 1)
    fig, axs = plt.subplots(len(all_keys), sharex=True)
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        axs[i].set_title(k.capitalize())
        axs[i].plot(jnp.convolve(stats_dict[k][conv_len:nb_steps], jnp.ones(conv_len) / conv_len, mode='valid'))
    plt.tight_layout()
    plt.xticks(np.arange(0, nb_steps - 2 * conv_len, ticks))
    fig.savefig(os.path.join(save_dir, f"stats_trunc_running_means_{conv_len}.png"))
    plt.close(fig)

    # Remove init transition
    fig, axs = plt.subplots(len(all_keys), sharex=True)
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        axs[i].set_title(k.capitalize())
        tmp_stat = stats_dict[k][:nb_steps]
        axs[i].plot(tmp_stat[-NB_STATS_STEPS:])
    plt.tight_layout()
    truncated_nb_steps = min(nb_steps, NB_STATS_STEPS)
    plt.xticks(np.arange(0, truncated_nb_steps, 8))
    fig.savefig(os.path.join(save_dir, f"stats_last_{truncated_nb_steps}.png"))
    plt.close(fig)


def generate_beta_faces(nb_betas: int, denominator: int) -> List[List[List[float]]]:
    """
        Generate a grid of all the valid beta values given a maximum number
        of beta values.
        This function makes sense only if we normalize our kernels.
    """
    all_values: List[List[float]] = []
    nb_val_per_column = denominator + 1
    for i in range(nb_val_per_column**nb_betas):
        all_values.append([((i // nb_val_per_column**j) % nb_val_per_column) / denominator
                           for j in range(nb_betas - 1, -1, -1)])

    face_list: List[List[List[float]]] = [[] for _ in range(nb_betas)]
    # We make sure all values contain at least one 1.
    for value in all_values:
        for i in range(nb_betas):
            if value[i] == 1.:
                # We remove trailing zeros
                while value[-1] == 0:
                    value.pop()
                face_list[i].append(value)
                # We break to avoid adding value with multiple ones to different faces
                break

    return face_list


###
# OS
###
def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif not os.path.isdir(dir):
        raise Exception('The path provided ({}) exist and is not a dir, aborting'.format(dir))


def save_config(save_dir: str, config: Dict):
    check_dir(save_dir)

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile)


def print_config(config: Dict):
    printable_config = copy.deepcopy(config)

    if 'cells' in printable_config['run_params'].keys():
        del printable_config['run_params']['cells']
    if 'init_cells' in printable_config['run_params'].keys():
        del printable_config['run_params']['init_cells']

    print(printable_config)


###
# Random
###
def seed_everything(seed: int):
    random.seed(seed)

    npseed = random.randint(1, 1_000_000)
    np.random.seed(npseed)

    ospyseed = random.randint(1, 1_000_000)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)

    jseed = random.randint(1, 1_000_000)
    rng_key = jax.random.PRNGKey(jseed)

    return rng_key


def generate_mask(shape, max_h, max_w):
    mask = np.ones(shape)

    pad_h_before = (max_h - mask.shape[1]) // 2
    pad_h_after = max_h - mask.shape[1] - pad_h_before
    pad_w_before = (max_w - mask.shape[2]) // 2
    pad_w_after = max_w - mask.shape[2] - pad_w_before
    padding = [(0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after)]
    mask = np.pad(mask, padding, mode='constant')

    return mask


###
# Memory
###
def get_needed_memory_per_sequence(config):
    nb_bytes_float_32 = 4
    nb_cells = config['world_params']['nb_channels'] * np.prod(config['render_params']['world_size'])
    bytes_per_world = nb_cells * nb_bytes_float_32
    bytes_per_sequence = bytes_per_world * config['run_params']['max_run_iter']

    mb_per_sequence = bytes_per_sequence * 8 / 1024 / 1024

    return mb_per_sequence
