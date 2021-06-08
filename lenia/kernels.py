import jax.numpy as jnp
from typing import List, Dict, Any

from .utils import st2fracs2float


def poly_quad4(x: jnp.ndarray) -> jnp.ndarray:
    x = 4 * x * (1 - x)
    x = x**4

    return x


def poly_quad2(x: jnp.ndarray) -> jnp.ndarray:
    x = 2 * x * (1 - x)
    x = x**2

    return x


def gauss_bump4(x: jnp.ndarray) -> jnp.ndarray:
    x = 4 - 1 / (x * (1 - x))
    x = jnp.exp(x)

    return x


def step4(x: jnp.ndarray, q: float = 1 / 4) -> jnp.ndarray:
    return (x >= q) * (x <= 1 - q)


def staircase(x: jnp.ndarray, q: float = 1 / 4) -> jnp.ndarray:
    return (x >= q) * (x <= 1 - q) + (x < q) * 0.5


kernel_core = {0: poly_quad4, 1: poly_quad2, 2: gauss_bump4, 3: step4, 4: staircase}


def get_kernels_and_mapping(kernels_params: Dict, world_size: List[int], nb_channels: int, R: float):
    kernels_list = []
    mapping: Dict[str, Any] = {
        "cin_kernels": [[] for i in range(nb_channels)],
        "cout_kernels": [[] for i in range(nb_channels)],
        "cout_kernels_h": [[] for i in range(nb_channels)],
        "cin_growth_fns": {
            'gf_id': [],
            'm': [],
            's': [],
        },
        "true_channels": []
    }
    for kernel_idx, param in enumerate(kernels_params):
        kernels_list.append(get_kernel(param, world_size, R))

        channel_in = param["c_in"]
        channel_out = param["c_out"]
        mapping["cin_kernels"][channel_in].append(kernel_idx)
        mapping["cout_kernels"][channel_out].append(kernel_idx)
        mapping["cout_kernels_h"][channel_out].append(param["h"])

        mapping["cin_growth_fns"]["gf_id"].append(param["gf_id"])
        mapping["cin_growth_fns"]["m"].append(param["m"])
        mapping["cin_growth_fns"]["s"].append(param["s"])

    max_depth_per_channel = max(map(len, mapping["cin_kernels"]))

    kernels = jnp.vstack(kernels_list)
    kernels = remove_zeros(kernels)

    K_h = kernels.shape[-2]
    K_w = kernels.shape[-1]

    kernels_per_channel = []
    for kernel_list in mapping["cin_kernels"]:
        K_tmp = kernels[jnp.array(kernel_list)]  # [O=nb_k_per_channel, K_h, K_w]

        diff = max_depth_per_channel - K_tmp.shape[0]
        mapping["true_channels"].append(jnp.array([True] * K_tmp.shape[0] + [False] * diff))
        if diff > 0:
            K_tmp = jnp.vstack([K_tmp, jnp.zeros([diff, K_h, K_w])])  # [O=max_depth_per_channel, K_h, K_w]

        K_tmp = K_tmp[:, jnp.newaxis, ...]  # [O=max_depth_per_channel, I=1, K_h, K_w]
        kernels_per_channel.append(K_tmp)
    K = jnp.stack(kernels_per_channel)  # [vmap_dim, O=max_depth_per_channel, I=1, K_h, K_w]

    mapping["true_channels"] = jnp.concatenate(mapping["true_channels"])
    mapping["cout_kernels"] = jnp.array(
        list(map(lambda arr: arr + [-1] * (max_depth_per_channel - len(arr)), mapping["cout_kernels"]))
    )
    mapping["cout_kernels_h"] = jnp.array(
        list(map(lambda arr: arr + [0] * (max_depth_per_channel - len(arr)), mapping["cout_kernels_h"]))
    )

    return K, mapping


def get_kernel(kernel_params: Dict, world_size: list, R: float) -> jnp.ndarray:
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


def remove_zeros(kernels: jnp.ndarray) -> jnp.ndarray:
    assert len(kernels.shape) == 3  # nb_dims == 2

    kernels = kernels[:, ~jnp.all(kernels == 0, axis=(0, 2))]  # remove 0 columns
    kernels = kernels[:, :, ~jnp.all(kernels == 0, axis=(0, 1))]  # remove 0 lines

    return kernels
