# import time
import functools
import copy
import random
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Union
import uuid
from omegaconf import DictConfig, OmegaConf

from .constant import NB_STATS_STEPS, EPSILON
from .loader import decompress_array

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')
image_ext = "png"


###
# Config
###
def get_container(omegaConf: DictConfig, main_path: str) -> Dict:
    # TODO: Need to check if missing first
    omegaConf.run_params.code = str(uuid.uuid4())

    world_params = omegaConf.world_params
    render_params = omegaConf.render_params
    if omegaConf.render_params.pixel_size == 'MISSING':
        pixel_size = 2**render_params.pixel_size_power2
        omegaConf.render_params.pixel_size = pixel_size
    if omegaConf.render_params.world_size == 'MISSING':
        world_size = [2**render_params['size_power2']] * world_params['nb_dims']
        omegaConf.render_params.world_size = world_size

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    config['main_path'] = main_path

    if 'scale' not in config['world_params']:
        config['world_params']['scale'] = 1.

    return config


def get_param(dic: Dict, key_string: str) -> Union[float, int]:
    keys = key_string.split('.')
    for key in keys:
        if key.isdigit():
            dic = dic[int(key)]
        else:
            dic = dic[key]

    if isinstance(dic, jnp.ndarray):
        return float(jnp.mean(dic))
    elif type(dic) == int or type(dic) == float:
        return float(dic)  # type: ignore
    elif type(dic) == str:
        return dic  # type: ignore
    else:
        raise ValueError(f"dic type {type(dic)} not supported")


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
        if k == 'channel_mass' and len(stats_dict['channel_mass'].shape) > 1:
            for j in range(stats_dict['channel_mass'].shape[1]):
                axs[i].plot(stats_dict['channel_mass'][:nb_steps, j])
        else:
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
        if k == 'channel_mass' and len(stats_dict['channel_mass'].shape) > 1:
            for j in range(stats_dict['channel_mass'].shape[1]):
                conv_stat = jnp.convolve(
                    stats_dict['channel_mass'][conv_len:nb_steps, j], jnp.ones(conv_len) / conv_len, mode='valid'
                )
                axs[i].plot(conv_stat)
        else:
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
        if k == 'channel_mass' and len(stats_dict['channel_mass'].shape) > 1:
            for j in range(stats_dict['channel_mass'].shape[1]):
                axs[i].plot(stats_dict['channel_mass'][:nb_steps][-NB_STATS_STEPS:, j])
        else:
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
