# import time
import functools
import copy
import random
import logging
from pprint import pprint
import os
import yaml
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from fractions import Fraction
from PIL import Image
from typing import Tuple, List, Dict, Any
from omegaconf import DictConfig, OmegaConf

from .constant import NB_STATS_STEPS, EPSILON

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')


###
# Config
###
def get_container(omegaConf: DictConfig, main_path: str) -> Dict:
    """Retrieve the populated container from an omegaConf

    This functions is used to:
    - Populate missing mandatory value in Leniax configuration
    - Handle versioning update if necessary

    .. code-block:: python

        # Use with Hydra as follow
        @hydra.main(config_path=config_path, config_name=config_name)
        def run(omegaConf: DictConfig) -> None:
            config = leniax_utils.get_container(omegaConf, config_path)

    Args:
        omegaConf: Hydra's Omega configuration
        main_path: Absolute path of the configuration directory

    Returns:
        A populated python dictionnary
    """
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

    if "algo" not in config:
        config["algo"] = {}

    if "other" not in config:
        config["other"] = {}
    if "log_level" not in config["other"]:
        config["other"]["log_level"] = 20

    if 'colormaps' not in config['render_params']:
        config['render_params']['colormaps'] = ['extended']
    if type(config['render_params']['colormaps']) == str:
        config['render_params']['colormaps'] = [config['render_params']['colormaps']]

    if 'update_fn_version' in config['world_params']:
        config['world_params']['get_state_fn_slug'] = config['world_params']['update_fn_version']
        del config['world_params']['update_fn_version']

    if 'version' not in config or config['version'] == 1:
        # we are dealing with a V1 config
        config = update_config_v1_v2(config)

    return config


def update_config_v1_v2(config: Dict) -> Dict:
    """Update a configuration from version 1 to version 2

    Args:
        config: A v1 configuration

    Returns:
        A v2 configuration
    """
    config['version'] = 2

    old_gf_register = {0: 'poly_quad4', 1: 'gaussian', 2: 'gaussian_target', 3: 'step'}
    old_k_register = {0: 'poly_quad', 1: 'gauss_bump', 2: 'step', 3: 'staircase', 4: 'gauss'}
    new_kernels_params = []
    for kernel_params in config['kernels_params']['k']:
        if type(kernel_params['b']) == str:
            bs = st2fracs2float(kernel_params['b'])
        else:
            bs = kernel_params['b']

        new_kernel_params = {
            'k_slug': 'circle_2d',
            'k_params': [
                kernel_params['r'] if 'r' in kernel_params else 1.,
                bs,
            ],
            'kf_slug': old_k_register[kernel_params['k_id']],
            'kf_params': [kernel_params['q']],
            'gf_slug': old_gf_register[kernel_params['gf_id']],
            'gf_params': [kernel_params['m'], kernel_params['s']],
            'h': kernel_params['h'],
            'c_in': kernel_params['c_in'],
            'c_out': kernel_params['c_out'],
        }
        new_kernels_params.append(new_kernel_params)
    config['kernels_params'] = new_kernels_params

    if 'genotype' in config:
        for gen_conf in config['genotype']:
            if 'kernels_params.k' in gen_conf['key']:
                gen_conf['key'] = gen_conf['key'].replace('kernels_params.k', 'kernels_params')
                gen_conf['key'] = gen_conf['key'].replace('.m', '.gf_params.0')
                gen_conf['key'] = gen_conf['key'].replace('.s', '.gf_params.1')

    if 'phenotype' in config:
        for idx, phen_str in enumerate(config['phenotype']):
            if 'kernels_params.k' in phen_str:
                phen_str = phen_str.replace('kernels_params.k', 'kernels_params')
                phen_str = phen_str.replace('.m', '.gf_params.0')
                config['phenotype'][idx] = phen_str.replace('.s', '.gf_params.1')

    get_state_fn_slug = config['world_params']['get_state_fn_slug'] if 'get_state_fn_slug' in config['world_params'
                                                                                                     ] else 'v1'
    if get_state_fn_slug == 'v1':
        config['algo']['init_slug'] = 'perlin'
    elif get_state_fn_slug == 'v2':
        config['algo']['init_slug'] = 'perlin_local'
    config['algo']['init_param'] = []

    return config


def get_param(dic: Dict, key_string: str) -> Any:
    """Retrieve a parameter in a dictionnary using a string

    Args:
        dic: Dictionnary
        key_string: String representing the parameter path in the dicitonnary

    Returns:
        The parameter
    """
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
    """Set a parameter in a dictionnary using a string

    Args:
        dic: Dictionnary
        key_string: String representing the parameter path in the dicitonnary
        value: Value to be set
    """
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
                dic = dic.setdefault(key, [])
            else:
                dic = dic.setdefault(key, {})
    if keys[-1].isdigit():
        idx = int(keys[-1])
        if len(dic) < idx + 1:
            for _ in range(idx + 1 - len(dic)):
                dic.append(None)  # type: ignore
        dic[idx] = value
    else:
        dic[keys[-1]] = value


def st2fracs2float(st: str) -> List[float]:
    """Convert a string of fraction into an list of floats

    String example: ``st = '1,2/3,6.7'``

    Args:
        st: String of fractions.

    Returns:
        An list of float
    """
    return [float(Fraction(st)) for st in st.split(',')]


###
# Animals
###
def merge_cells(cells: jnp.ndarray, other_cells: jnp.ndarray, offset: List[int] = None) -> jnp.ndarray:
    """Merge cells together using addition

    Args:
        cells: Base cells
        other_cells: Cells to be merged into the base cells
        offset: Offsets in each dimensions

    Returns:
        Resulting cells
    """
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
@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, None), out_axes=(0, 0, 0))
def center_world(
    cells: jnp.ndarray,
    field: jnp.ndarray,
    potential: jnp.ndarray,
    shift_idx: jnp.ndarray,
    axes: Tuple[int, ...],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rool cells, field and potential

    Args:
        cells: Lenia state of shape ``[bs, other_axes...]``
        field: Lenia field ``[bs, other_axes...]``
        potential: Lenia potential ``[bs, other_axes...]``
        shift_idx: Amount to roll of shape ``[bs, nb_axes]```
        axes: Axes to roll

    Returns:
        Updated cells, field and potential
    """
    cells = jnp.roll(cells, -shift_idx, axes)
    field = jnp.roll(field, -shift_idx, axes)
    potential = jnp.roll(potential, -shift_idx, axes)

    return cells, field, potential


def crop_zero(kernels: jnp.ndarray) -> jnp.ndarray:
    """Crop zero values out for 3 and 4 dimensions array

    Args:
        kernels: A 3 or 4 dimension kernels

    Returns:
        The croppped kernels
    """
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


def auto_center(cells: jnp.ndarray) -> jnp.ndarray:
    """Automatically center cells on its total mass centroid

    Args:
        cells: Lenia state

    Returns:
        The mass centered Lenia state
    """
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


def center_and_crop(cells: jnp.ndarray) -> jnp.ndarray:
    """Automatically center cells on its total mass centroid and crop zeros values

    Args:
        cells: Lenia state

    Returns:
        The mass centered cropped Lenia state
    """
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

    centered_cells = auto_center(cells)
    cells = crop_zero(centered_cells)

    if 0 in cells.shape:
        cells = jnp.zeros([cells.shape[0]] + [1] * (len(cells.shape) - 1))

    return cells


###
# VIZ
###
def get_image(cells_buffer: np.ndarray, pixel_size: int, colormap) -> Image:
    """Convert a numpy array into a PIL image

    Args:
        cells_buffer: A Lenia state of shape ``[C, world_dims...]``
        pixel_size: Size of each state pixels in the image
        colormap: A matplotlib compatible colormap

    Returns:
        A PIL image
    """
    nb_world_dims = len(cells_buffer.shape) - 1

    if pixel_size != 1:
        for i in range(nb_world_dims):
            cells_buffer = np.repeat(cells_buffer, pixel_size, axis=i + 1)

    if nb_world_dims == 2:
        cells_buffer = np.transpose(cells_buffer, (1, 2, 0))
        img = colormap(cells_buffer)[:, :, 0]
        rgba_uint8 = np.uint8(img * 255)
        final_img = Image.fromarray(rgba_uint8, 'RGBA')
    elif nb_world_dims == 3:
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
        raise ValueError("nb_world_dims > 3 not supported")

    return final_img


def plot_stats(save_dir: str, stats_dict: Dict):
    """Plot Leniax statistics

    Args:
        save_dir: directory used to save assets.
        stats_dict: Statistics dictionnary
    """
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
    """Generate a grid of all the valid beta values given a maximum number
        of beta values.

    This function makes sense only if we normalize our kernels.

    Args:
        nb_betas: Maximum number of betas values
        denominator: Denominator to construct the betas grid

    Returns:
        The list of possible list of beta values
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
    """Ensure a directory exist and is not a file

    Args:
        dir: Checked directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif not os.path.isdir(dir):
        raise Exception('The path provided ({}) exist and is not a dir, aborting'.format(dir))


def save_config(save_dir: str, config: Dict):
    """Save a configuration file

    Args:
        save_dir: directory used to save assets.
        config: Leniax configuration
    """
    check_dir(save_dir)

    config = copy.deepcopy(config)
    if 'main_path' in config:
        del config['main_path']

    with open(os.path.join(save_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile)


def print_config(config: Dict):
    """Pretty print a Leniax configuration

    Args:
        config: Leniax configuration
    """
    printable_config = copy.deepcopy(config)

    if 'cells' in printable_config['run_params'].keys():
        del printable_config['run_params']['cells']
    if 'init_cells' in printable_config['run_params'].keys():
        del printable_config['run_params']['init_cells']

    if logging.getLogger().getEffectiveLevel() <= 20:
        pprint(printable_config)


def load_img(fullpath: str, resize: Tuple[int, int]) -> jnp.ndarray:
    """Load an image as a np.array

    Args:
        fullpath: Absolute image path
        resize: Resize factors for the image dimensions

    Returns:
        The image as a np.array
    """
    img = Image.open(fullpath).resize(resize)
    img_np = np.rollaxis(np.array(img), -1)[np.newaxis] / 255.
    return jnp.array(img_np, dtype=jnp.float32)


def set_log_level(config: Dict):
    """Set the python logging root level

    Args:
        config: Leniax configuration
    """
    numeric_level = config['other']['log_level'] if 'log_level' in config['other'] else 1
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {numeric_level}')
    logging.getLogger().setLevel(numeric_level)


###
# Random
###
def seed_everything(seed: int) -> jax.random.KeyArray:
    """Seed all the depedencies used by Leniax

    Args:
        seed: A seed integer

    Returns:
        A JAX PRNG key
    """
    random.seed(seed)

    npseed = random.randint(1, 1_000_000)
    np.random.seed(npseed)

    ospyseed = random.randint(1, 1_000_000)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)

    jseed = random.randint(1, 1_000_000)
    rng_key = jax.random.PRNGKey(jseed)

    return rng_key


###
# Memory
###
def get_needed_memory(config: Dict, nb_sols: int = 1):
    """Compute an approximate of the needed memory by different kind of simulations

    Args:
        config: Leniax configuration
        nb_sols: How many solutions will be simulated at the same time

    Returns:
        A dictionnary witth different memory requirements
    """
    nb_init = config['run_params']['nb_init_search']

    kb_per_float = 4 / 1024
    nb_cells = config['world_params']['nb_channels'] * np.prod(config['render_params']['world_size'])
    nb_kernels = len(config['kernels_params'])
    R = config['world_params']['R']

    kernel_size_conv = (2 * R)**2
    kernel_size_fft = np.prod(config['render_params']['world_size'])
    kb_kernels_conv = nb_kernels * kernel_size_conv * kb_per_float * nb_sols
    kb_kernels_fft = nb_kernels * kernel_size_fft * kb_per_float * nb_sols

    kb_per_world = nb_cells * kb_per_float
    mb_all_world = kb_per_world * nb_init * nb_sols / 1024

    kb_per_seq = kb_per_world * config['run_params']['max_run_iter']
    mb_per_seq = kb_per_seq / 1024

    # We store 12 metrics for now
    stats_kb_per_seq = 12 * config['run_params']['max_run_iter'] * kb_per_float
    stats_kb_total = stats_kb_per_seq * nb_init * nb_sols

    # we need memory for the init cells, final cells, all statistics, kernel/gfn parameters
    # and the intermediary field and potential allocated
    mb_qd_conv_estimate = mb_all_world * 2 + kb_kernels_conv / 1024 + stats_kb_total / 1024
    mb_qd_fft_estimate = mb_all_world * 2 + kb_kernels_fft / 1024 + stats_kb_total / 1024

    return {
        'kb_per_world': kb_per_world,
        'mb_all_world': mb_all_world,
        'mb_per_seq': mb_per_seq,
        'kb_kernels_conv': kb_kernels_conv,
        'kb_kernels_fft': kb_kernels_fft,
        'stats_kb_per_seq': stats_kb_per_seq,
        'stats_kb_total': stats_kb_total,
        'mb_qd_conv_estimate': mb_qd_conv_estimate,
        'mb_qd_fft_estimate': mb_qd_fft_estimate,
    }
