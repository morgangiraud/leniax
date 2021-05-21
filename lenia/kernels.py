import jax.numpy as jnp

KERNEL_MODE_ALL_IN = 0
KERNEL_MODE_ONE = 1


def poly_quad4(x):
    x = 4 * x * (1 - x)
    x = x**4

    return x


def gauss_bump4(x):
    x = 4 - 1 / (x * (1 - x))
    x = jnp.exp(x)

    return x


def step4(x, q=1 / 4):
    return (x >= q) * (x <= 1 - q)


def staircase(x, q=1 / 4):
    (x >= q) * (x <= 1 - q) + (x < q) * 0.5


kernel_core = {0: poly_quad4, 1: gauss_bump4, 2: step4, 3: staircase}


def get_kernel(params, world_size: list):
    midpoint = jnp.asarray([size // 2 for size in world_size])
    coords = jnp.indices(world_size)

    whitened_coords = [(coords[i] - midpoint[i]) / params['R'] for i in range(coords.shape[0])]
    distances = jnp.sqrt(sum([x**2 for x in whitened_coords]))  # Distances from the center of the grid

    kernel = kernel_shell(distances, params)
    kernel = kernel / kernel.sum()

    kernel_fft = jnp.fft.fftn(kernel)

    kernel = kernel[~jnp.all(kernel == 0, axis=1)]  # remove 0 lines
    kernel = kernel[:, ~jnp.all(kernel == 0, axis=0)]  # remove 0 columns

    return kernel, kernel_fft


def kernel_shell(distances: jnp.array, params: list) -> jnp.array:
    kernel_func = kernel_core[params['kn'] - 1]

    nb_b = len(params['b'])
    bs = jnp.asarray([float(f) for f in params['b']])

    B_dist = nb_b * distances  # scale distances by the number of modes
    bs_mat = bs[jnp.minimum(jnp.floor(B_dist).astype(int), nb_b - 1)]  # Define postions for each mode

    # All kernel functions are defined in [0, 1] so we keep only value with distance under 1
    kernel = (distances < 1) * kernel_func(jnp.minimum(B_dist % 1, 1)) * bs_mat

    return kernel
