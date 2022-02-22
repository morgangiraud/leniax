import jax
import math

from leniax import core as leniax_core


def make_run_fn(rng_key, config, multiplier):
    config['bench']['nb_k'] *= multiplier
    config['world_params']['R'] = int(config['world_params']['R'] * math.log(multiplier + 2))
    config['world_params']['nb_channels'] *= multiplier
    config['run_params']['nb_init_search'] *= multiplier
    config['run_params']['max_run_iter'] *= multiplier

    R = config['world_params']['R']
    C = config['world_params']['nb_channels']
    N_init = config['run_params']['nb_init_search']
    world_size = config['render_params']['world_size']
    subkeys = jax.random.split(rng_key)

    state = jax.random.uniform(subkeys[0], [N_init, C] + world_size)

    max_k_per_channel = 1
    K = jax.random.uniform(subkeys[1], [C * max_k_per_channel, 1, 2 * R, 2 * R])
    K /= K.sum()

    pad_l = [(0, 0), (0, 0)]
    for dim in K.shape[2:]:
        if dim % 2 == 0:
            pad_l += [(dim // 2, dim // 2 - 1)]
        else:
            pad_l += [(dim // 2, dim // 2)]
    padding = tuple(pad_l)

    true_channels = None

    K.block_until_ready()

    def bench_fn():
        potential = leniax_core.get_potential(state, K, padding, true_channels)
        potential.block_until_ready()
        return potential

    return bench_fn
