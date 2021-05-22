import jax.numpy as jnp
from jax import lax

from . import utils
from .kernels import get_kernel
from .kernels import KERNEL_MODE_ALL_IN, KERNEL_MODE_ONE, KERNEL_MODE_ALL_IN_FFT, KERNEL_MODE_ONE_FFT
from .growth_functions import growth_func


def init(animal_conf, world_size: list, nb_channels: int, kernel_mode: int = KERNEL_MODE_ONE):
    assert len(world_size) == 2  # We linit ourselves to 2d worlds
    assert nb_channels > 0

    params = {
        'R': animal_conf['params']['R'],  # Ratio
        'T': animal_conf['params']['T'],  # ∆T
        'b': utils.st2fracs2float(animal_conf['params']['b']),  # Array of modes amplitude (tells the number of modes)
        'm': animal_conf['params']['m'],  # mean
        's': animal_conf['params']['s'],  # std
        'kn': animal_conf['params']['kn'],  # kernel id
        'gn': animal_conf['params']['gn'],  # growth function id
    }
    gfunc = growth_func[params['gn'] - 1]

    nb_dims = len(world_size)
    world_shape = [nb_channels] + world_size  # CHW
    cells = jnp.zeros(world_shape)
    animal_cells = utils.rle2arr(animal_conf['cells'], nb_dims, nb_channels)
    cells = utils.add_animal(cells, animal_cells)

    # We consider only 2 possible case
    # - Kernels with input_channels I = C: KERNEL_MODE_ALL_IN
    # - Kernels with input_channels I = 1: KERNEL_MODE_ONE
    if kernel_mode == KERNEL_MODE_ONE:
        nb_kernels = 2
        K_c = 1

        kernels = []
        kernels_fft = []
        for i in range(nb_kernels):
            kernel, kernel_fft = get_kernel(params, world_size, K_c)  # generate [K_c, K_h, K_w] kernels

            kernels.append(kernel[jnp.newaxis, ...])  # [1, K_c, K_h, K_w]
            kernels_fft.append(kernel_fft[jnp.newaxis, ...])
        kernel = jnp.concatenate(kernels, axis=0)  # [O, K_c, K_h, K_w]
        kernel_fft = jnp.concatenate(kernels_fft, axis=0)  # [O, K_c, K_h, K_w]

    elif kernel_mode == KERNEL_MODE_ALL_IN:
        # For now we limit ourselves to the following case
        # - same number of kernels and channels
        nb_kernels = nb_channels
        K_c = nb_channels
        raise Exception("kernel_mode KERNEL_MODE_ALL_IN not supported yet")
    else:
        raise Exception(f"kernel_mode {kernel_mode} not supported yet")

    return params, cells, gfunc, kernel, kernel_fft


def build_update_fn(params, growth_fn, kernel, kernel_mode=KERNEL_MODE_ONE, weights=None):
    T = params['T']
    m = params['m']
    s = params['s']

    if kernel_mode == KERNEL_MODE_ONE:
        get_potential_fn = get_potential_mode_one
    elif kernel_mode == KERNEL_MODE_ALL_IN:
        raise Exception('KERNEL_MODE_ALL_IN not supported yet')
    elif kernel_mode == KERNEL_MODE_ALL_IN_FFT:
        raise Exception('KERNEL_MODE_ALL_IN_FFT not supported yet')
    elif kernel_mode == KERNEL_MODE_ONE_FFT:
        get_potential_fn = get_potential_fft
    else:
        raise Exception(f"kernel mode {kernel_mode} not supported")

    def update(cells: jnp.array):
        potential = get_potential_fn(cells, kernel, weights)
        field = growth_fn(potential, m, s)
        cells = update_cells(cells, field, T)

        return cells

    return update


def get_potential_mode_one(cells: jnp.array, kernel: jnp.array, weights=None):
    # Cells: [C, H, W]
    # Kernel: [O, K_c = 1, K_h, K_w]
    assert kernel.shape[1] == 1
    if weights:
        assert len(weights) == kernel.shape[0]

    A = cells[jnp.newaxis, jnp.newaxis, ...]  # [1, 1, C, H, W]
    K = kernel[:, jnp.newaxis, :, :, :]  # [O, 1, K_c = 1, K_h, K_w]
    pad_w = kernel.shape[-1] // 2
    pad_h = kernel.shape[-2] // 2
    padded_A = jnp.pad(A, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='wrap')
    A_out = lax.conv(padded_A, K, (1, 1, 1), 'VALID')  # [1, O, C, H, W]
    A_out = jnp.average(A_out, axis=1, weights=weights)  # [1, C, H, W]
    potential = A_out[0]  # [C, H, W]

    return potential


def get_potential_fft(cells: jnp.array, kernel_fft: jnp.array, weights=None):
    # Cells: [C, H, W]
    # Kernel: [O, K_c = 1, K_h, K_w]
    assert cells.shape[0] == 1  # We limit ourselves to one channel for now

    world_fft = jnp.fft.fftn(cells)
    potentials = []
    for k_idx in range(kernel_fft.shape[0]):
        potential_fft = kernel_fft[k_idx] * world_fft
        potential = jnp.fft.fftshift(jnp.real(jnp.fft.ifftn(potential_fft)))
        potentials.append(potential)
    potential = jnp.mean(jnp.asarray(potentials), axis=0)

    return potential


def update_cells(cells, field, T):
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)

    return cells_new
