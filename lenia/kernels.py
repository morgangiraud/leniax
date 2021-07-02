import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .utils import st2fracs2float


@dataclass()
class KernelMapping(object):
    cin_kernels: List[List[int]]
    cout_kernels: List[List[int]]
    cout_kernels_h: List[List[float]]
    true_channels: List[bool]
    cin_growth_fns: Dict[str, List]

    def __init__(self, nb_channels: int):
        self.cin_kernels = [[] for i in range(nb_channels)]
        self.cout_kernels = [[] for i in range(nb_channels)]
        self.cout_kernels_h = [[] for i in range(nb_channels)]
        self.true_channels = []
        self.cin_growth_fns = {
            'gf_id': [],
            'm': [],
            's': [],
        }


def poly_quad4(x: jnp.ndarray):
    out = 4 * x * (1 - x)
    out = out**4

    return out


def poly_quad2(x: jnp.ndarray):
    out = 2 * x * (1 - x)
    out = out**2

    return out


def gauss_bump4(x: jnp.ndarray):
    out = 4 - 1 / (x * (1 - x))
    out = jnp.exp(out)

    return out


def step4(x: jnp.ndarray, q: float = 1 / 4):
    return (x >= q) * (x <= 1 - q)


def staircase(x: jnp.ndarray, q: float = 1 / 4):
    return (x >= q) * (x <= 1 - q) + (x < q) * 0.5


kernel_core = {0: poly_quad4, 1: poly_quad2, 2: gauss_bump4, 3: step4, 4: staircase}


def get_kernels_and_mapping(kernels_params: Dict, world_size: List[int], nb_channels: int,
                            R: float) -> Tuple[jnp.ndarray, KernelMapping]:
    """
        Build the kernel and the mapping used in the update function by JAX.
        To take advantages of JAX vmap and conv_generalized function, we stack all kernels
        and keep track of the kernel graph in mapping:
        - Each kernel is applied to one channel and the result is added to one other channel
    """

    kernels_list = []
    mapping = KernelMapping(nb_channels)
    for kernel_idx, param in enumerate(kernels_params):
        kernels_list.append(get_kernel(param, world_size, R))

        channel_in = param["c_in"]
        channel_out = param["c_out"]
        mapping.cin_kernels[channel_in].append(kernel_idx)
        mapping.cout_kernels[channel_out].append(kernel_idx)
        mapping.cout_kernels_h[channel_out].append(param["h"])

        mapping.cin_growth_fns["gf_id"].append(param["gf_id"])
        mapping.cin_growth_fns["m"].append(param["m"])
        mapping.cin_growth_fns["s"].append(param["s"])
    # ! vstack concantenate on the first dimension
    # Currently it works, ebcause we only considere 1-channel kernels
    kernels = jnp.vstack(kernels_list)  # [nb_kernels, H, W]
    kernels = remove_zero_dims(kernels)  # [nb_kernels, K_h=max_k_h, K_w=max_k_w]

    K_h = kernels.shape[-2]
    K_w = kernels.shape[-1]

    max_k_per_channel = max(map(len, mapping.cin_kernels))
    kernels_per_channel = []
    for kernel_list in mapping.cin_kernels:
        K_tmp = kernels[jnp.array(kernel_list)]  # [O=nb_kernel_per_channel, K_h, K_w]

        # We create the mask
        diff = max_k_per_channel - K_tmp.shape[0]
        mapping.true_channels.append(jnp.array([True] * K_tmp.shape[0] + [False] * diff))
        if diff > 0:
            K_tmp = jnp.vstack([K_tmp, jnp.zeros([diff, K_h, K_w])])  # [O=max_k_per_channel, K_h, K_w]

        K_tmp = K_tmp[:, jnp.newaxis, ...]  # [O=max_k_per_channel, I=1, K_h, K_w]
        kernels_per_channel.append(K_tmp)
    K = jnp.stack(kernels_per_channel)  # [vmap_dim=nb_channels, O=max_k_per_channel, I=1, K_h, K_w]
    assert K.shape[0] == nb_channels

    mapping.true_channels = jnp.concatenate(mapping.true_channels)
    mapping.cout_kernels = jnp.array(
        list(map(lambda arr: arr + [-1] * (max_k_per_channel - len(arr)), mapping.cout_kernels))
    )
    mapping.cout_kernels_h = jnp.array(
        list(map(lambda arr: arr + [0] * (max_k_per_channel - len(arr)), mapping.cout_kernels_h))
    )

    return K, mapping


def get_kernel(kernel_params: Dict, world_size: list, R: float) -> jnp.ndarray:
    """ Build one kernel """
    midpoint = jnp.asarray([size // 2 for size in world_size])
    coords = jnp.indices(world_size)

    whitened_coords = [(coords[i] - midpoint[i]) / (kernel_params['r'] * R) for i in range(coords.shape[0])]
    distances = jnp.sqrt(sum([x**2 for x in whitened_coords]))  # Distances from the center of the grid

    kernel = kernel_shell(distances, kernel_params)
    kernel = kernel / kernel.sum()
    kernel = kernel[jnp.newaxis, ...]

    return kernel


def kernel_shell(distances: jnp.ndarray, kernel_params: Dict) -> jnp.ndarray:
    kernel_func = kernel_core[kernel_params['k_id']]

    bs = jnp.asarray(st2fracs2float(kernel_params['b']))
    nb_b = bs.shape[0]

    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we keep only value with distance under 1
    kernel = (distances < 1) * kernel_func(jnp.minimum(B_dist % 1, 1)) * bs_mat  # type: ignore

    return kernel


def remove_zero_dims(kernels: jnp.ndarray) -> jnp.ndarray:
    assert len(kernels.shape) == 3  # nb_dims == 2

    kernels = kernels[:, ~jnp.all(kernels == 0, axis=(0, 2))]  # remove 0 columns
    kernels = kernels[:, :, ~jnp.all(kernels == 0, axis=(0, 1))]  # remove 0 lines

    return kernels
