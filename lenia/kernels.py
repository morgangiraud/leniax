import os
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import List, Dict, Tuple

from . import utils as lenia_utils


@dataclass()
class KernelMapping(object):
    cin_kernels: List[List[int]]
    cin_growth_fns: Dict[str, List]
    kernels_weight_per_channel: List[List[float]]
    true_channels: jnp.ndarray

    def __init__(self, nb_channels: int, nb_kernels: int):
        self.cin_kernels = [[] for _ in range(nb_channels)]
        self.cin_growth_fns = {
            'gf_id': [],
            'm': [],
            's': [],
        }

        self.kernels_weight_per_channel = [[0.] * nb_kernels for _ in range(nb_channels)]

    def get_gfn_params(self) -> jnp.ndarray:
        return jnp.vstack([self.cin_growth_fns['m'], self.cin_growth_fns['s']]).T

    def get_kernels_weight_per_channel(self) -> jnp.ndarray:
        return jnp.array(self.kernels_weight_per_channel)


def poly_quad4(x: jnp.ndarray, q: float = 4):
    out = 4 * x * (1 - x)
    out = out**q

    return out


def gauss_bump(x: jnp.ndarray, q: float = 1):
    out = q - 1 / (x * (1 - x))
    out = jnp.exp(q * out)

    return out


def gauss(x: jnp.ndarray, q: float = 1):
    out = -((x - q) / (0.3 * q))**2
    out = jnp.exp(out / 2)

    return out


def step4(x: jnp.ndarray, q: float = 1 / 4):
    return (x >= q) * (x <= 1 - q)


def staircase(x: jnp.ndarray, q: float = 1 / 4):
    return (x >= q) * (x <= 1 - q) + (x < q) * 0.5


kernel_core = {0: poly_quad4, 1: gauss_bump, 2: step4, 3: staircase, 4: gauss}


def get_kernels_and_mapping(kernels_params: Dict, world_size: List[int], nb_channels: int,
                            R: float) -> Tuple[jnp.ndarray, KernelMapping]:
    """
        Build the kernel and the mapping used in the update function by JAX.
        To take advantages of JAX vmap and conv_generalized function, we stack all kernels
        and keep track of the kernel graph in mapping:
        - Each kernel is applied to one channel and the result is added to one other channel
    """

    kernels_list = []
    mapping = KernelMapping(nb_channels, len(kernels_params))
    for kernel_idx, param in enumerate(kernels_params):
        kernels_list.append(get_kernel(param, world_size, R))

        channel_in = param["c_in"]
        channel_out = param["c_out"]
        mapping.cin_kernels[channel_in].append(kernel_idx)
        mapping.kernels_weight_per_channel[channel_out][kernel_idx] = param["h"]

        mapping.cin_growth_fns["gf_id"].append(param["gf_id"])
        mapping.cin_growth_fns["m"].append(param["m"])
        mapping.cin_growth_fns["s"].append(param["s"])
    # ! vstack concantenate on the first dimension
    # Currently it works, because we only considere 1-channel kernels
    kernels = jnp.vstack(kernels_list)  # [nb_kernels, H, W]
    kernels = crop_zero(kernels)  # [nb_kernels, K_h=max_k_h, K_w=max_k_w]

    K_h = kernels.shape[-2]
    K_w = kernels.shape[-1]

    max_k_per_channel = max(map(len, mapping.cin_kernels))
    kernels_per_channel = []
    true_channels = []
    for kernel_list in mapping.cin_kernels:
        K_tmp = kernels[jnp.array(kernel_list)]  # [O=nb_kernel_per_channel, K_h, K_w]

        # We create the mask
        diff = max_k_per_channel - K_tmp.shape[0]
        true_channels.append(jnp.array([True] * K_tmp.shape[0] + [False] * diff))
        if diff > 0:
            K_tmp = jnp.vstack([K_tmp, jnp.zeros([diff, K_h, K_w])])  # [O=max_k_per_channel, K_h, K_w]

        K_tmp = K_tmp[:, jnp.newaxis, ...]  # [O=max_k_per_channel, I=1, K_h, K_w]
        kernels_per_channel.append(K_tmp)
    K = jnp.vstack(kernels_per_channel)  # [O=nb_channels*max_k_per_channel, I=1, K_h, K_w]
    assert K.shape[0] == nb_channels * max_k_per_channel

    mapping.true_channels = jnp.concatenate(true_channels)

    return K, mapping


def get_kernel(kernel_params: Dict, world_size: list, R: float) -> jnp.ndarray:
    """ Build one kernel """
    midpoint = jnp.asarray([size // 2 for size in world_size])  # [nb_dims]
    midpoint = midpoint.reshape([-1] + [1] * len(world_size))  # [nb_dims, 1, 1, ...]
    coords = jnp.indices(world_size)  # [nb_dims, dim_0, dim_1, ...]
    centered_coords = (coords - midpoint) / (kernel_params['r'] * R)  # [nb_dims, dim_0, dim_1, ...]
    # Distances from the center of the grid

    distances = jnp.sqrt(jnp.sum(centered_coords**2, axis=0))  # [dim_0, dim_1, ...]

    kernel = kernel_shell(distances, kernel_params)
    kernel = kernel / kernel.sum()
    kernel = kernel[jnp.newaxis, ...]  # [1, dim_0, dim_1, ...]

    return kernel


def kernel_shell(distances: jnp.ndarray, kernel_params: Dict) -> jnp.ndarray:
    bs = jnp.asarray(lenia_utils.st2fracs2float(kernel_params['b']))
    nb_b = bs.shape[0]

    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we keep only value with distance under 1
    kernel_func = kernel_core[kernel_params['k_id']]
    kernel_q = kernel_params['q']
    kernel = (distances < 1) * kernel_func(B_dist % 1, kernel_q) * bs_mat  # type: ignore

    return kernel


def crop_zero(kernels: jnp.ndarray) -> jnp.ndarray:
    assert len(kernels.shape) == 3  # nb_dims == 2

    kernels = kernels[:, ~jnp.all(kernels == 0, axis=(0, 2))]  # remove 0 columns
    kernels = kernels[:, :, ~jnp.all(kernels == 0, axis=(0, 1))]  # remove 0 lines

    return kernels


###
# Viz
###


def draw_kernels(K, save_dir, colormap):
    nb_dims = len(K.shape[2:])
    data = jnp.ones(K.shape[2:])
    padding = [(dim, dim) for dim in data.shape]
    data = jnp.pad(data, padding, mode='constant')

    pixel_size = 1
    pixel_border_size = 0
    for i in range(K.shape[1]):
        id = str(i).zfill(2)
        current_k = K[0, i:i + 1]
        img = lenia_utils.get_image(
            lenia_utils.normalize(current_k, 0, jnp.max(current_k)), pixel_size, pixel_border_size, colormap
        )
        with open(os.path.join(save_dir, f"kernel{id}.png"), 'wb') as f:
            img.save(f, format='png')

        out = lax.conv(data[jnp.newaxis, jnp.newaxis], current_k[jnp.newaxis], [1] * nb_dims, 'valid')
        img = lenia_utils.get_image(out[0], pixel_size, pixel_border_size, colormap)
        with open(os.path.join(save_dir, f"kernel{id}_conv.png"), 'wb') as f:
            img.save(f, format='png')
