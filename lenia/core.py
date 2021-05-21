import jax.numpy as jnp
from jax import lax

from . import utils
from .kernels import get_kernel, KERNEL_MODE_ONE, KERNEL_MODE_ALL_IN
from .growth_functions import growth_func


def init(animal_conf, world_size, nb_channels, kernel_mode=KERNEL_MODE_ONE):
    assert len(world_size) == 2  # We linit ourselves to 2d worlds
    assert nb_channels > 0

    params = {
        'R': animal_conf['params']['R'],  # Ratio
        'T': animal_conf['params']['T'],  # ∆T
        'b': utils.st2fracs(animal_conf['params']['b']),  # Array of modes amplitude (tells the number of modes)
        'm': animal_conf['params']['m'],  # mean
        's': animal_conf['params']['s'],  # std
        'kn': animal_conf['params']['kn'],  # kernel id
        'gn': animal_conf['params']['gn'],  # growth function id
    }
    gfunc = growth_func[params['gn'] - 1]

    nb_dims = len(world_size)
    world_shape = world_size + [nb_channels]  # HWC
    cells = jnp.zeros(world_shape)
    animal_cells = utils.rle2arr(animal_conf['cells'], nb_dims, nb_channels)
    cells = utils.add_animal(cells, animal_cells)

    # We consider only 2 possible case
    # - Kernels with input_channels I = C: KERNEL_MODE_ALL_IN
    # - Kernels with input_channels I = 1: KERNEL_MODE_ONE
    if kernel_mode == KERNEL_MODE_ONE:
        # For now we limit ourselves to the following case
        # - same number of kernels and channels
        nb_kernels = nb_channels
        kernels = []
        kernels_fft = []
        for i in range(nb_kernels):
            kernel, kernel_fft = get_kernel(params, world_size)  # HW kernels

            kernel = kernel[jnp.newaxis, jnp.newaxis, ...]  # [I, K_c, K_h, K_w] kernels (I = 1, K_c = 1)
            kernels.append(kernel[jnp.newaxis, ...])

            kernels_fft.append(kernel_fft[..., jnp.newaxis])

        kernel = jnp.concatenate(kernels, axis=0)  # [O, I, K_c, K_h, K_w]
    elif kernel_mode == KERNEL_MODE_ALL_IN:
        raise Exception("kernel_mode KERNEL_MODE_ALL_IN not supported yet")
    else:
        raise Exception(f"kernel_mode {kernel_mode} not supported yet")

    return params, cells, gfunc, kernel, kernels_fft


def get_potential(cells, kernel, kernel_mode):
    if kernel_mode == KERNEL_MODE_ONE:
        # Cells: [H, W, C]
        # Kernel: [O, I = 1, K_c = 1, K_h, K_w]
        assert kernel.shape[1] == 1
        assert kernel.shape[2] == 1

        A = cells.transpose(2, 0, 1)[jnp.newaxis, jnp.newaxis, ...]  # [1, 1, C, H, W]
        pad_w = kernel.shape[-1] // 2
        pad_h = kernel.shape[-2] // 2
        padded_A = jnp.pad(A, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='wrap')
        A_out = lax.conv(padded_A, kernel, (1, 1, 1), 'VALID')
        A_out = A_out.mean(axis=1)  # [1, C, H, W]
        potential = A_out[0].transpose(1, 2, 0)  # [H, W, C]

    else:
        raise Exception("kernel_mode KERNEL_MODE_ALL_IN not supported yet")

    return potential


def update(params, cells, gfunc, kernel, kernel_mode):
    potential = get_potential(cells, kernel, kernel_mode)
    field = get_field(gfunc, params['m'], params['s'], potential)
    cells = update_cells(cells, field, params['T'])

    return cells


def update_cells(cells, field, T):
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)

    cells = cells_new

    return cells


def update_fft(params, cells, gfunc, kernel_fft):
    potential = get_potential_fft(cells, kernel_fft)
    field = get_field(gfunc, params['m'], params['s'], potential)
    cells = update_cells(cells, field, params['T'])

    return cells


# Potential
def get_potential_fft(cells, kernels_fft):
    assert cells.shape[-1] == 1  # We limit ourselves to one channel for now

    # Cells: [H, W, C]
    world_fft = jnp.fft.fftn(cells)
    potentials = []
    for kernel_fft in kernels_fft:
        potential_fft = kernel_fft * world_fft
        potential = jnp.fft.fftshift(jnp.real(jnp.fft.ifftn(potential_fft)))
        potentials.append(potential)
    potential = jnp.mean(jnp.asarray(potentials), axis=0)

    return potential


# Field
def get_field(gfunc, m, s, potential):
    field = gfunc(potential, m, s)

    return field
