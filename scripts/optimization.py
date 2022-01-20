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
import leniax.initializations as leniax_initializations
from leniax.kernels import get_kernels_and_mapping

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))

config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "wanderer"


def K_fn_fft(K_params):
    padding = jnp.array([[0, 0]] * 2 + [[(128 - dim) // 2, (128 - dim) // 2 + 1] for dim in K_params.shape[2:]])
    K = jnp.pad(K_params, padding, mode='constant')[jnp.newaxis, ...]
    K = jnp.fft.fftshift(K, axes=[-2, -1])
    K = jnp.fft.fftn(K, axes=[-2, -1])
    return K


def K_fn_id(K_params):
    return K_params


def error_fn_triple(preds, target):
    all_cells = preds[0]
    all_cells_target = target[0]
    diffs = 0.5 * jnp.square(all_cells - all_cells_target)
    errors = jnp.sum(diffs, axis=(1, 2, 3))
    cells_batch_error = jnp.mean(errors)

    all_fields = preds[1]
    all_fields_target = target[1]
    diffs = 0.5 * jnp.square(all_fields - all_fields_target)
    errors = jnp.mean(diffs, axis=(1, 2, 3))
    fields_batch_error = jnp.mean(errors)

    all_potentials = preds[2]
    all_potentials_target = target[2]
    diffs = 0.5 * jnp.square(all_potentials - all_potentials_target)
    errors = jnp.sum(diffs, axis=(1, 2, 3))
    potentials_batch_error = jnp.mean(errors)

    batch_error = (cells_batch_error + fields_batch_error + potentials_batch_error) / 3
    return batch_error


def error_fn_cells(preds, target):
    all_cells = preds[0]
    all_cells_target = target[0]
    diffs = 0.5 * jnp.square(all_cells - all_cells_target)
    errors = jnp.sum(diffs, axis=(1, 2, 3))
    cells_batch_error = jnp.mean(errors)

    batch_error = cells_batch_error
    return batch_error


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    config['run_params']['nb_init_search'] = 2
    config['run_params']['max_run_iter'] = 32
    leniax_utils.print_config(config)

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    # Seed the env
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    ###
    # Creating the training set
    ###
    config_target = copy.deepcopy(config)

    print("Creating training set init_cells")
    nb_init_search = config_target['run_params']['nb_init_search']
    world_params = config_target['world_params']
    nb_channels = world_params['nb_channels']
    world_size = config_target['render_params']['world_size']
    R = config['world_params']['R']

    _, subkey = jax.random.split(rng_key)
    nb_channels_to_init = nb_channels * nb_init_search
    rng_key, noises = leniax_initializations.perlin(subkey, nb_channels_to_init, world_size, R, config_target['kernels_params']['k'][0])
    all_init_cells = noises.reshape([nb_init_search, 1] + world_size)
    config_target['run_params']['init_cells'] = all_init_cells

    print("Rendering target: start")
    start_time = time.time()
    all_cells_target, all_fields_target, all_potentials_target, _ = leniax_helpers.init_and_run(
        config_target, use_init_cells=True, with_jit=True, fft=True, stat_trunc=False
    )  # [nb_max_iter, N=1, C, world_dims...]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells_target)
    print(f"Rendering target: {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    nb_steps_to_train = 1
    training_data_shape = [-1, nb_channels] + world_size
    training_data = {
        'input_cells': all_cells_target[:-nb_steps_to_train].reshape(training_data_shape),
        'target_cells': all_cells_target[nb_steps_to_train:].reshape(training_data_shape),
        'target_fields': all_fields_target[nb_steps_to_train:].reshape(training_data_shape),
        'target_potentials': all_potentials_target[nb_steps_to_train:].reshape(training_data_shape),
    }

    ###
    # Preparing the training pipeline
    ###
    config_train = copy.deepcopy(config)
    # config_train['world_params']['update_fn_version'] = 'v1'
    # config_train['kernels_params']['k'][0]['b'] = [1., 1.]
    # config_train['kernels_params']['k'][0]['gf_id'] = 0
    # config_train['kernels_params']['k'][0]['m'] = 0.15
    # config_train['kernels_params']['k'][0]['s'] = 0.015
    # config_train['kernels_params']['k'][0]['k_id'] = 0
    # config_train['kernels_params']['k'][0]['q'] = 1.

    K, mapping = get_kernels_and_mapping(config_train['kernels_params']['k'], world_size, nb_channels, R, fft=False)
    #     # kernel_shape = [
    #     #     1,
    #     #     config_train['world_params']['nb_channels'],
    #     #     K.shape[0] // config_train['world_params']['nb_channels'],
    #     # ] + config_train['render_params']['world_size']
    kernel_shape = K.shape
    rng_key, subkey = jax.random.split(rng_key)
    K_params = jax.random.uniform(subkey, shape=kernel_shape)
    K_params = K_params / K_params.sum()

    update_fn = leniax_helpers.build_update_fn(kernel_shape, mapping, 'v1', True, fft=False)
    run_fn = functools.partial(
        leniax_runner.run_diff,
        max_run_iter=nb_steps_to_train,
        K_fn=K_fn_id,
        update_fn=update_fn,
        error_fn=error_fn_cells
    )
    run_fn_value_and_grads = jax.value_and_grad(run_fn, argnums=(1))

    lr = 1e-5
    opt_init, opt_update, get_params = jax_opt.adam(lr)
    opt_state = opt_init((K_params, ))

    gfn_params = mapping.get_gfn_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    T = jnp.array(config_train['world_params']['T'], dtype=jnp.float32)
    # for i in range(50000):
    for i in range(50000):
        x = training_data['input_cells']
        y = (training_data['target_cells'], training_data['target_fields'], training_data['target_potentials'])
        error, grads = run_fn_value_and_grads(x, K_params, gfn_params, kernels_weight_per_channel, T, target=y)
        if type(grads) != tuple:
            grads = (grads, )
        opt_state = opt_update(i, grads, opt_state)

        if i % 100 == 0:
            opt_params = get_params(opt_state)

            K_params = opt_params[0]
            K_params_neg_grad = -grads[0]
            print(i, error)

            fig = plt.figure(figsize=(6, 6))
            axes = []
            axes.append(fig.add_subplot(1, 2, 1))
            axes[-1].title.set_text("kernel K1")
            plt.imshow(K_params[0, 0], cmap='viridis', interpolation="nearest", vmin=0, vmax=K_params.max())
            plt.colorbar()
            axes.append(fig.add_subplot(1, 2, 2))
            axes[-1].title.set_text("kernel K1 grad")
            plt.imshow(
                K_params_neg_grad[0, 0],
                cmap='hot',
                interpolation="nearest",
                vmin=K_params_neg_grad.min(),
                vmax=K_params_neg_grad.max()
            )
            plt.colorbar()
            plt.tight_layout()
            fullpath = f"{save_dir}/Ks{str(i).zfill(4)}.png"
            fig.savefig(fullpath)
            plt.close(fig)


if __name__ == '__main__':
    run()
