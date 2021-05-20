import jax.numpy as jnp
from jax import lax

from . import utils

kernel_core = {
    0: lambda r: (4 * r * (1 - r))**4,  # polynomial (quad4)
    1: lambda r: jnp.exp(4 - 1 / (r * (1 - r))),  # exponential / gaussian bump (bump4)
    2: lambda r,
    q=1 / 4: (r >= q) * (r <= 1 - q),  # step (stpz1/4)
    3: lambda r,
    q=1 / 4: (r >= q) * (r <= 1 - q) + (r < q) * 0.5  # staircase (life)
}
growth_func = {
    0: lambda n,
    m,
    s: jnp.maximum(0, 1 - (n - m)**2 / (9 * s**2))**4 * 2 - 1,  # polynomial (quad4)
    1: lambda n,
    m,
    s: jnp.exp(-(n - m)**2 / (2 * s**2)) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n,
    m,
    s: (jnp.abs(n - m) <= s) * 2 - 1  # step (stpz)
}


def init(animal_conf, world_size, nb_dims):
    # channels = 1
    height, width = world_size
    # cells = jnp.zeros([channels, height, width])
    cells = jnp.zeros([height, width])

    animal_cells = utils.rle2arr(animal_conf['cells'], nb_dims)
    # if len(animal_cells.shape) == 2:
    #     animal_cells = animal_cells[jnp.newaxis, :, :]

    cells = utils.add_animal(cells, animal_cells)

    params = {
        'R': animal_conf['params']['R'],
        'T': animal_conf['params']['T'],
        'b': utils.st2fracs(animal_conf['params']['b']),
        'm': animal_conf['params']['m'],
        's': animal_conf['params']['s'],
        'kn': animal_conf['params']['kn'],
        'gn': animal_conf['params']['gn'],
    }

    gfunc = growth_func[params['gn'] - 1]
    kernel = get_kernel(params, world_size)
    kernel_FFT = jnp.fft.fftn(kernel)

    kernel = kernel[~jnp.all(kernel == 0, axis=1)]  # remove 0 lines
    kernel = kernel[:, ~jnp.all(kernel == 0, axis=0)]  # remove 0 columns
    kernel = kernel[:, :, jnp.newaxis, jnp.newaxis]

    return params, cells, gfunc, kernel, kernel_FFT


def get_kernel(params, world_size):
    midpoint = jnp.asarray([int(size / 2) for size in world_size])
    coords = jnp.indices(world_size)

    whitened_coords = [(coords[i] - midpoint[i]) / params['R'] for i in range(coords.shape[0])]
    distances = jnp.sqrt(sum([x**2 for x in whitened_coords]))  # Distances from the center of the grid

    kernel = kernel_shell(distances, params)
    kernel_sum = kernel.sum()
    kernel_norm = kernel / kernel_sum

    return kernel_norm


def kernel_shell(distances, params):
    B = len(params['b'])
    B_dist = B * distances
    bs = jnp.asarray([float(f) for f in params['b']])
    b = bs[jnp.minimum(jnp.floor(B_dist).astype(int), B - 1)]
    kernel_func = kernel_core[params['kn'] - 1]
    return (distances < 1) * kernel_func(jnp.minimum(B_dist % 1, 1)) * b


# Potential
def get_potential_fft(cells, kernel_FFT):
    world_FFT = jnp.fft.fftn(cells)
    potential_FFT = kernel_FFT * world_FFT
    potential = jnp.fft.fftshift(jnp.real(jnp.fft.ifftn(potential_FFT)))

    return potential


def get_potential(cells, kernel):
    # Cells shape should be [H, W] or [C, H, W]
    # Kernel shape should be [H, W, I, O]
    if len(cells.shape) == 2:
        # We add the batch and channel dims
        cells = cells[jnp.newaxis, jnp.newaxis, :, :]
    if len(cells.shape) == 3:
        # We add the batch dim
        cells = cells[jnp.newaxis]

    pad = kernel.shape[0] // 2
    padded_cells = jnp.pad(cells, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='wrap')
    potential = lax.conv(
        padded_cells,  # lhs = NCHW image tensor
        jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
        (1, 1),  # window strides
        'VALID'
    )  # padding mode
    potential = potential[0, 0]  # Remove batchsize & image dims

    return potential


# Field
def get_field(gfunc, m, s, potential):
    field = gfunc(potential, m, s)

    return field


def update_cells(cells, field, T):
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)

    cells = cells_new

    return cells


def update_fft(params, cells, gfunc, kernel_FFT):
    potential = get_potential_fft(cells, kernel_FFT)
    field = get_field(gfunc, params['m'], params['s'], potential)
    cells = update_cells(cells, field, params['T'])

    return cells


def update(params, cells, gfunc, kernel):
    potential = get_potential(cells, kernel)
    field = get_field(gfunc, params['m'], params['s'], potential)
    cells = update_cells(cells, field, params['T'])

    return cells
