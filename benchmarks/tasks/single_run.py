import functools
import jax
import math
import copy

from leniax import helpers as leniax_helpers
from leniax import statistics as leniax_stat
from leniax import runner as leniax_runner
from leniax.kernels import get_kernels_and_mapping


def make_run_fn(rng_key, config, multiplier):
    config['bench']['nb_k'] *= multiplier
    config['world_params']['R'] = int(config['world_params']['R'] * math.log(multiplier + 2))
    config['world_params']['nb_channels'] *= multiplier
    config['run_params']['nb_init_search'] *= multiplier
    config['run_params']['max_run_iter'] *= multiplier

    nb_k = config['bench']['nb_k']
    k_template = config['kernels_params'][0]
    for k in range(nb_k - 1):
        new_k = copy.deepcopy(k_template)
        new_k['c_in'] = k + 1
        new_k['c_out'] = k + 1
        config['kernels_params'].append(new_k)

    R = config['world_params']['R']
    T = config['world_params']['T']
    C = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    nb_init = config['run_params']['nb_init_search']
    max_run_iter = config['run_params']['max_run_iter']
    cells_shape = [
        nb_init,
        C,
    ] + world_size

    rng_key, subkey = jax.random.split(rng_key)
    cells = jax.random.uniform(subkey, cells_shape)
    K, mapping = get_kernels_and_mapping(config['kernels_params'], world_size, C, R, config['bench']['fft'])
    gf_params = mapping.get_gf_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()

    world_params = config['world_params']
    get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True

    update_fn = leniax_helpers.build_update_fn(
        K.shape, mapping, get_state_fn_slug, weighted_average, config['bench']['fft']
    )
    compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    if config['bench']['with_jit'] is True:
        run_fn = functools.partial(
            leniax_runner.run_scan,
            rng_key=rng_key,
            cells0=cells,
            K=K,
            gf_params=gf_params,
            kernels_weight_per_channel=kernels_weight_per_channel,
            T=T,
            max_run_iter=max_run_iter,
            R=R,
            update_fn=update_fn,
            compute_stats_fn=compute_stats_fn
        )
    else:
        run_fn = functools.partial(
            leniax_runner.run,
            rng_key=rng_key,
            cells=cells,
            K=K,
            gf_params=gf_params,
            kernels_weight_per_channel=kernels_weight_per_channel,
            T=T,
            max_run_iter=max_run_iter,
            R=R,
            update_fn=update_fn,
            compute_stats_fn=compute_stats_fn,
            stat_trunc=False
        )

    return run_fn
