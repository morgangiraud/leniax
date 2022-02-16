import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import jax.numpy as jnp

from . import utils as leniax_utils
from .kernel_functions import register as kf_register


@dataclass()
class KernelMapping(object):
    """Explicit mapping of the computation graph

    Attributes:
        cin_kernels: list of kernel indexes grouped by channel of shape ``[nb_channels, max_nb_kernels]``
        cin_k_params: list of kernel parameters grouped by channel of shape ``[nb_channels, max_nb_kernels]``
        cin_kfs: list of kernel functions grouped by channel of shape ``[nb_channels, max_nb_kernels]``
        cin_gfs: list of growth function slugs grouped by channel of shape ``[nb_channels, max_nb_kernels]``
        gf_params: list of growth function slug grouped by channel of shape ``[nb_channels, max_nb_kernels]``
        kernels_weight_per_channel: list of weights grouped by channel of shape ``[nb_channels, nb_kernels]``
    """
    cin_kernels: List[List[int]]
    cin_k_params: List[List]
    cin_kfs: List[List[str]]
    cin_gfs: List[List[str]]
    cin_gf_params: List[List]
    kernels_weight_per_channel: List[List[float]]
    true_channels: jnp.ndarray

    def __init__(self, nb_channels: int, nb_kernels: int):
        self.cin_kernels = [[] for _ in range(nb_channels)]
        self.cin_k_params = [[] for _ in range(nb_channels)]
        self.cin_kfs = [[] for _ in range(nb_channels)]
        self.cin_gfs = [[] for _ in range(nb_channels)]
        self.cin_gf_params = [[] for _ in range(nb_channels)]
        self.kernels_weight_per_channel = [[0.] * nb_kernels for _ in range(nb_channels)]

    def get_k_params(self) -> jnp.ndarray:
        """Get all kernel params, flattened

        Returns:
            An array of shape ``[nb_kernels, kernel_params_shape...]``
        """
        flat_k_params = [item for sublist in self.cin_k_params for item in sublist]
        return jnp.array(flat_k_params, dtype=jnp.float32)

    def get_gf_params(self) -> jnp.ndarray:
        """Get all growth function params, flattened

        Returns:
            An array of shape ``[nb_kernels, gf_params_shape...]``
        """
        flat_gf_params = [item for sublist in self.cin_gf_params for item in sublist]
        return jnp.array(flat_gf_params, dtype=jnp.float32)

    def get_kernels_weight_per_channel(self) -> jnp.ndarray:
        """Get all growth function params, flattened

        Returns:
            An array of shape ``[nb_channels, nb_kernels]``
        """

        return jnp.array(self.kernels_weight_per_channel, dtype=jnp.float32)


def get_kernels_and_mapping(
    kernels_params: List,
    world_size: List[int],
    nb_channels: int,
    R: float,
    fft: bool = True,
) -> Tuple[jnp.ndarray, KernelMapping]:
    """Contruxt the kernel array and the associated mapping

    Build the kernel and the mapping used in the update function by JAX.
    To take advantages of JAX vmap and conv_generalized function, we stack all kernels
    and keep track of the kernel graph in mapping:
    - Each kernel is applied to one channel and the result is added to one other channel

    Args:
        kernels_params: Kernels parameters.
        world_size: World size.
        nb_channels: Number of channels.
        R: World radius.
        fft: Set to ``True`` to compute a FFT kernel.
    """
    kernels_list = []
    mapping = KernelMapping(nb_channels, len(kernels_params))
    # We need to sort those kernels_params to make sure kernels_weight_per_channel is in the right order
    kernels_params.sort(key=lambda d: d['c_in'])
    for kernel_idx, param in enumerate(kernels_params):
        k = register[param['k_slug']](R, param['k_params'], param['kf_slug'], param['kf_params'])
        padding = [[0, 0]]
        for ws, ks in zip(world_size, k.shape[1:]):
            pad_needed = ws - ks
            if pad_needed % 2 == 0:
                pad = [(ws - ks) // 2, (ws - ks) // 2]
            else:
                pad = [(ws - ks) // 2, (ws - ks) // 2 + 1]
            padding.append(pad)
        kernels_list.append(jnp.pad(k, padding))

        channel_in = param["c_in"]
        mapping.cin_kernels[channel_in].append(kernel_idx)
        mapping.cin_gfs[channel_in].append(param["gf_slug"])
        mapping.cin_gf_params[channel_in].append(param["gf_params"])
        mapping.cin_kfs[channel_in].append(param["kf_slug"])
        mapping.cin_k_params[channel_in].append(param["k_params"])

        channel_out = param["c_out"]
        mapping.kernels_weight_per_channel[channel_out][kernel_idx] = param["h"]

    # ! vstack concantenate on the first dimension
    # Currently it works, because we only considere 1-channel kernels
    kernels = jnp.vstack(kernels_list)  # [nb_kernels, H, W]
    if fft is False:
        kernels = leniax_utils.crop_zero(kernels)  # [nb_kernels, K_h=max_k_h, K_w=max_k_w]

    K_h = kernels.shape[-2]
    K_w = kernels.shape[-1]

    max_k_per_channel = max(map(len, mapping.cin_kernels))
    kernels_per_channel = []
    true_channels = []
    for cin_kernel_list in mapping.cin_kernels:
        if len(cin_kernel_list):
            K_tmp = kernels[jnp.array(cin_kernel_list)]  # [O=nb_kernel_per_channel, K_h, K_w]
        else:
            K_tmp = jnp.zeros([0, K_h, K_w])

        # We create the mask
        nb_k = K_tmp.shape[0]
        nb_missing = max_k_per_channel - nb_k
        true_channels.append(jnp.array([True] * nb_k + [False] * nb_missing))
        if nb_missing > 0:
            K_tmp = jnp.vstack([K_tmp, jnp.zeros([nb_missing, K_h, K_w])])  # [O=max_k_per_channel, K_h, K_w]

        K_tmp = K_tmp[:, jnp.newaxis, ...]  # [O=max_k_per_channel, I=1, K_h, K_w]
        kernels_per_channel.append(K_tmp)
    mapping.true_channels = jnp.concatenate(true_channels)

    if fft is True:
        axes = list(range(-1, -len(world_size) - 1, -1))
        K = jnp.stack(kernels_per_channel)[jnp.newaxis, :, :, 0, ...]  # [1, nb_channels, max_k_per_channel, H, W]
        K = jnp.fft.fftshift(K, axes=axes)
        K = jnp.fft.fftn(K, axes=axes)
        assert K.shape[0] == 1
        assert K.shape[1] == nb_channels
        assert K.shape[2] == max_k_per_channel
    else:
        K = jnp.vstack(kernels_per_channel)  # [O=nb_channels*max_k_per_channel, I=1, K_h, K_w]
        assert K.shape[0] == nb_channels * max_k_per_channel
        assert K.shape[1] == 1

    return K, mapping


def raw(R, k_params: List, kf_slug: str, kf_params: jnp.ndarray) -> jnp.ndarray:
    """Returns k_params as an array

    Args:
        R: World radius **(not used)**
        k_params: Kernel parameters
        kf_slug: Kernel function slug
        kf_params: Kernel function params

    Returns:
        A kernel of shape ``k_params``
    """
    return jnp.array(k_params, dtype=jnp.float32)


def circle_2d(R, k_params: List, kf_slug: str, kf_params: jnp.ndarray) -> jnp.ndarray:
    """Build a circle kernel

    Args:
        R: World radius
        k_params: Kernel parameters
        kf_slug: Kernel function slug
        kf_params: Kernel function params

    Returns:
        A kernel of shape ``[1, world_dims...]``
    """
    r = k_params[0]
    bs = jnp.array(k_params[1], dtype=jnp.float32)
    k_radius_px = math.ceil(r * R)

    midpoint = jnp.array([k_radius_px, k_radius_px])
    midpoint = midpoint.reshape([-1] + [1, 1])  # [nb_dims, 1, 1]
    coords = jnp.indices([k_radius_px * 2, k_radius_px * 2])  # [nb_dims, dim_0, dim_1]
    centered_coords = (coords - midpoint) / (r * R)  # [nb_dims, dim_0, dim_1]
    # Distances from the center of the grid
    distances = jnp.sqrt(jnp.sum(centered_coords**2, axis=0))  # [dim_0, dim_1]

    nb_b = bs.shape[0]
    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we compute B_dist modulo 1 to make sure value lies in the good range
    # Notice that those kernel functions should be positive and fk(0) = 0, fk(1) = 0
    # We then crop to distances < 1 which defines the "size" of the kernel
    raw_kernel = kf_register[kf_slug](kf_params, B_dist % 1)
    kernel = (distances < 1) * raw_kernel * bs_mat  # type: ignore
    kernel = kernel / kernel.sum()

    kernel = kernel[jnp.newaxis, ...]  # [1, dim_0, dim_1, ...]

    return kernel


def ellipse_2d(R, k_params: List, kf_slug: str, kf_params: jnp.ndarray) -> jnp.ndarray:
    """Build a circle kernel

    Args:
        R: World radius
        k_params: Kernel parameters
        kf_slug: Kernel function slug
        kf_params: Kernel function params

    Returns:
        A kernel of shape ``[1, world_dims...]``
    """
    r = k_params[0]
    k_radius_px = math.ceil(r * R)
    bs = jnp.array(k_params[1], dtype=jnp.float32)
    a = k_params[2]
    b = k_params[3]
    theta = k_params[4] * jnp.pi

    midpoint = jnp.array([k_radius_px, k_radius_px])
    midpoint = midpoint.reshape([-1] + [1, 1])  # [nb_dims, 1, 1]
    coords = jnp.indices([k_radius_px * 2, k_radius_px * 2])  # [nb_dims, dim_0, dim_1]
    centered_coords = (coords - midpoint) / (r * R)  # [nb_dims, dim_0, dim_1]
    # Rotated distances from the center of the grid
    rot_centered_coords = jnp.stack([
        centered_coords[0] * jnp.cos(theta) + centered_coords[1] * jnp.sin(theta),
        -centered_coords[0] * jnp.sin(theta) + centered_coords[1] * jnp.cos(theta)
    ])
    distances = jnp.sqrt((rot_centered_coords[0] / a)**2 + (rot_centered_coords[1] / b)**2)

    nb_b = bs.shape[0]
    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we compute B_dist modulo 1 to make sure value lies in the good range
    # Notice that those kernel functions should be positive and fk(0) = 0, fk(1) = 0
    # We then crop to distances < 1 which defines the "size" of the kernel
    raw_kernel = kf_register[kf_slug](kf_params, B_dist % 1)
    kernel = (distances < 1) * raw_kernel * bs_mat  # type: ignore
    kernel = kernel / kernel.sum()

    grad_coord = rot_centered_coords[0].at[rot_centered_coords[0] < -0.01].set(-1)
    grad_coord = grad_coord.at[grad_coord > 0.01].set(1)
    kernel = kernel * grad_coord

    kernel = kernel[jnp.newaxis, ...]  # [1, dim_0, dim_1, ...]

    return kernel

def oriented_ellipse_2d(R, k_params: List, kf_slug: str, kf_params: jnp.ndarray) -> jnp.ndarray:
    """Build a circle kernel

    Args:
        R: World radius
        k_params: Kernel parameters
        kf_slug: Kernel function slug
        kf_params: Kernel function params

    Returns:
        A kernel of shape ``[1, world_dims...]``
    """
    r = k_params[0]
    k_radius_px = math.ceil(r * R)
    bs = jnp.array(k_params[1], dtype=jnp.float32)
    a = k_params[2]
    b = k_params[3]
    theta = k_params[4] * jnp.pi

    midpoint = jnp.array([k_radius_px, k_radius_px])
    midpoint = midpoint.reshape([-1] + [1, 1])  # [nb_dims, 1, 1]
    coords = jnp.indices([k_radius_px * 2, k_radius_px * 2])  # [nb_dims, dim_0, dim_1]
    centered_coords = (coords - midpoint) / (r * R)  # [nb_dims, dim_0, dim_1]
    # Rotated distances from the center of the grid
    rot_centered_coords = jnp.stack([
        centered_coords[0] * jnp.cos(theta) + centered_coords[1] * jnp.sin(theta),
        -centered_coords[0] * jnp.sin(theta) + centered_coords[1] * jnp.cos(theta)
    ])
    distances = jnp.sqrt((rot_centered_coords[0] / a)**2 + (rot_centered_coords[1] / b)**2)

    nb_b = bs.shape[0]
    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we compute B_dist modulo 1 to make sure value lies in the good range
    # Notice that those kernel functions should be positive and fk(0) = 0, fk(1) = 0
    # We then crop to distances < 1 which defines the "size" of the kernel
    raw_kernel = kf_register[kf_slug](kf_params, B_dist % 1)
    kernel = (distances < 1) * raw_kernel * bs_mat  # type: ignore
    kernel = kernel / kernel.sum()

    grad_coord = rot_centered_coords[0].at[rot_centered_coords[0] < -0.01].set(-1)
    grad_coord = grad_coord.at[grad_coord > 0.01].set(1)
    kernel = kernel * grad_coord

    kernel = kernel[jnp.newaxis, ...]  # [1, dim_0, dim_1, ...]

    return kernel


register: Dict[str, Callable] = {
    'raw': raw,
    'circle_2d': circle_2d,
    'ellipse_2d': ellipse_2d,
    'oriented_ellipse_2d': oriented_ellipse_2d,
}
