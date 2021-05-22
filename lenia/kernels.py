import jax.numpy as jnp

KERNEL_MODE_ALL_IN = 0
KERNEL_MODE_ONE = 1
KERNEL_MODE_ALL_IN_FFT = 2
KERNEL_MODE_ONE_FFT = 3


def poly_quad4(x):
    x = 4 * x * (1 - x)
    x = x**4

    return x


def poly_quad2(x):
    x = 2 * x * (1 - x)
    x = x**2

    return x


def gauss_bump4(x):
    x = 4 - 1 / (x * (1 - x))
    x = jnp.exp(x)

    return x


def step4(x, q=1 / 4):
    return (x >= q) * (x <= 1 - q)


def staircase(x, q=1 / 4):
    (x >= q) * (x <= 1 - q) + (x < q) * 0.5


kernel_core = {0: poly_quad4, 1: poly_quad2, 2: gauss_bump4, 3: step4, 4: staircase}


def get_kernel(params, world_size: list, K_c: int):
    midpoint = jnp.asarray([size // 2 for size in world_size])
    coords = jnp.indices(world_size)

    whitened_coords = [(coords[i] - midpoint[i]) / params['R'] for i in range(coords.shape[0])]
    distances = jnp.sqrt(sum([x**2 for x in whitened_coords]))  # Distances from the center of the grid

    kernel = kernel_shell(distances, params)
    kernel = kernel / kernel.sum()

    kernel_fft = jnp.fft.fftn(kernel)

    kernel = kernel[~jnp.all(kernel == 0, axis=1)]  # remove 0 lines
    kernel = kernel[:, ~jnp.all(kernel == 0, axis=0)]  # remove 0 columns

    kernel = jnp.repeat(kernel[jnp.newaxis, ...], K_c, axis=0)
    kernel_fft = jnp.repeat(kernel_fft[jnp.newaxis, ...], K_c, axis=0)

    return kernel, kernel_fft


def kernel_shell(distances: jnp.array, params: list) -> jnp.array:
    kernel_func = kernel_core[params['kn'] - 1]

    nb_b = len(params['b'])
    bs = jnp.asarray(params['b'])

    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we keep only value with distance under 1
    kernel = (distances < 1) * kernel_func(jnp.minimum(B_dist % 1, 1)) * bs_mat

    return kernel
