import time
import os
import copy
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import jax
import jax.numpy as jnp

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
from leniax import core as leniax_core

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "wanderer"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_helpers.get_container(omegaConf, config_path)
    leniax_utils.print_config(config)

    config['run_params']['max_run_iter'] = 1024

    # Creating the training set
    config_v2 = copy.deepcopy(config)

    print("Rendering v2: start")
    start_time = time.time()
    all_cells_v2, _, _, stats_dict = leniax_helpers.init_and_run(
        config_v2, with_jit=False, fft=True, use_init_cells=False
    )  # [nb_max_iter, N=1, C, world_dims...]
    all_cells_v2 = all_cells_v2[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells_v2)
    print(f"Rendering v2: {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    config_v1 = copy.deepcopy(config)
    config_v1['world_params']['update_fn_version'] = 'v1'
    config_v1['kernels_params']['k'][0]['gf_id'] = 0
    config_v1['kernels_params']['k'][0]['k_id'] = 0
    config_v1['kernels_params']['k'][0]['q'] = 1.

    _, K, mapping = leniax_core.init(config, True, False)
    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    dt = 1. / config_v1['world_params']['T']
    update_fn = leniax_helpers.build_update_fn(K.shape, mapping, 'v1', True, True)

    def apply_3_updates(cells, K, gfn_params, kernels_weight_per_channel, dt, target):
        cells, _, _ = update_fn(cells, K, gfn_params, kernels_weight_per_channel, dt)
        cells, _, _ = update_fn(cells, K, gfn_params, kernels_weight_per_channel, dt)
        cells_out, _, _ = update_fn(cells, K, gfn_params, kernels_weight_per_channel, dt)
        error = jnp.mean((cells_out - target)**2) / 2.

        return error

    update_fn_gfngrad = jax.value_and_grad(apply_3_updates, argnums=(1, 2))
    inputs = all_cells_v2[:-3]
    target = all_cells_v2[3:]
    lr = 5e-2
    for i in range(200):
        error, (K_grad,
                gfnparams_grad) = update_fn_gfngrad(inputs, K, gfn_params, kernels_weight_per_channel, dt, target)
        gfn_params = gfn_params - lr * gfnparams_grad
        K = K - K_grad
        if i % 10 == 0:
            print(error, gfn_params)


if __name__ == '__main__':
    run()
