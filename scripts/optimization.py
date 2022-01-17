import functools
import time
import os
import copy
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import jax
import jax.numpy as jnp
from jax.experimental import optimizers as jax_opt
import matplotlib.pyplot as plt

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.runner as leniax_runner
from leniax.kernels import get_kernels_and_mapping

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "wanderer"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.print_config(config)

    config['run_params']['max_run_iter'] = 1024

    # Creating the training set
    config_v2 = copy.deepcopy(config)

    print("Rendering v2: start")
    start_time = time.time()
    all_cells_v2, _, _, _ = leniax_helpers.init_and_run(
        config_v2, with_jit=False, fft=True, use_init_cells=False, stat_trunc=True
    )  # [nb_max_iter, N=1, C, world_dims...]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells_v2)
    print(f"Rendering v2: {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    config_v1 = copy.deepcopy(config)
    config_v1['world_params']['update_fn_version'] = 'v1'
    config_v1['kernels_params']['k'][0]['gf_id'] = 0
    config_v1['kernels_params']['k'][0]['k_id'] = 0
    config_v1['kernels_params']['k'][0]['q'] = 1.

    K, mapping = get_kernels_and_mapping(
        config_v1['kernels_params']['k'],
        config_v1['render_params']['world_size'],
        1,
        R=config['world_params']['R'],
        fft=False
    )
    # kernel_shape = [
    #     1,
    #     config_v1['world_params']['nb_channels'],
    #     K.shape[0] // config_v1['world_params']['nb_channels'],
    # ] + config_v1['render_params']['world_size']
    kernel_shape = K.shape
    update_fn = leniax_helpers.build_update_fn(kernel_shape, mapping, 'v1', True, fft=False)

    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    T = jnp.array(config_v1['world_params']['T'], dtype=jnp.float32)
    max_run_iter = 1

    inputs = all_cells_v2[:-max_run_iter]
    target = (all_cells_v2[max_run_iter:], None, None)

    K_params = K

    # def K_fn(K_params):
    #     padding = jnp.array([[0, 0]] * 2 + [[(128 - dim) // 2, (128 - dim) // 2 + 1] for dim in K_params.shape[2:]])
    #     K = jnp.pad(K_params, padding, mode='constant')[jnp.newaxis, ...]
    #     K = jnp.fft.fftshift(K, axes=[-2, -1])
    #     K = jnp.fft.fftn(K, axes=[-2, -1])
    #     return K
    def K_fn(K_params):
        return K_params

    def error_fn(inputs, target):
        all_cells = inputs[0]
        all_cells_target = target[0]

        diffs = 0.5 * jnp.square(all_cells - all_cells_target)
        errors = jnp.sum(diffs, axis=(1, 2, 3))
        batch_error = jnp.mean(errors)

        return batch_error

    run_fn = functools.partial(
        leniax_runner.run_diff, max_run_iter=max_run_iter, K_fn=K_fn, update_fn=update_fn, error_fn=error_fn
    )
    run_fn_grads = jax.value_and_grad(run_fn, argnums=(1, 2, 4))

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    lr = 1e-4
    opt_init, opt_update, get_params = jax_opt.adam(lr)
    opt_state = opt_init((K_params, gfn_params, T))
    for i in range(5000):
        error, grads = run_fn_grads(inputs, K_params, gfn_params, kernels_weight_per_channel, T, target=target)
        opt_state = opt_update(i, grads, opt_state)

        if i % 5 == 0:
            K_params, gfn_params, T_params = get_params(opt_state)
            print(error, K_params.mean(), K_params.std(), gfn_params, T_params)

            fig = plt.figure(figsize=(6, 6))
            axes = []
            axes.append(fig.add_subplot(1, 2, 1))
            axes[-1].title.set_text("kernel K1")
            plt.imshow(K_params[0, 0], cmap='viridis', interpolation="nearest", vmin=0, vmax=K_params.max())
            axes.append(fig.add_subplot(1, 2, 2))
            axes[-1].title.set_text("kernel K1 grad")
            neg_grad = -grads[0]
            plt.imshow(neg_grad[0, 0], cmap='hot', interpolation="nearest", vmin=neg_grad.min(), vmax=neg_grad.max())

            plt.colorbar()
            plt.tight_layout()
            fullpath = f"{save_dir}/Ks{str(i).zfill(4)}.png"
            fig.savefig(fullpath)
            plt.close(fig)


if __name__ == '__main__':
    run()
