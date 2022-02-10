"""Leniax: Kernel optimization example.

This example shows how to use JAX to learn the original kernel from noise
of an existing simulation.

Usage:
    ``python examples/run.py -cn config_name -cp config_path run_params.nb_init_search=8 run_params.max_run_iter=512``
"""
import functools
import time
import os
import io
import copy
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np
import jax.example_libraries.optimizers as jax_opt
import matplotlib.pyplot as plt

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.runner as leniax_runner
import leniax.initializations as leniax_init
from leniax.kernels import get_kernels_and_mapping

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "orbium"


###
# A few helper functions used in the learning pipeline.
###
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


def error_fn_cells(rng_key, preds, target):
    all_cells = preds[0]
    all_cells_target = target[0]
    diffs = 0.5 * jnp.square(all_cells - all_cells_target)
    errors = jnp.sum(diffs, axis=(1, 2, 3))
    cells_batch_loss = jnp.mean(errors)

    batch_loss = cells_batch_loss

    return rng_key, batch_loss


###
# We use hydra to load Leniax configurations.
# It alloes to do many things among which, we can override configuraiton parameters.
# For example to render a Lenia in a bigger size:
# python examples/run.py render_params.world_size='[512, 512]' world_params.scale=4
###
cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "orbium"


@hydra.main(config_path=config_path, config_name=config_name)
def run(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)
    leniax_utils.print_config(config)

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    ###
    # Creating the training set
    #
    # First we gather data of a full simulation.
    ###
    config_target = copy.deepcopy(config)

    logging.info("Creating training set init_cells")
    nb_init_search = config_target['run_params']['nb_init_search']
    world_params = config_target['world_params']
    nb_channels = world_params['nb_channels']
    world_size = config_target['render_params']['world_size']
    R = config['world_params']['R']
    kernels_params = config_target['kernels_params']
    init_slug = config['algo']['init_slug']

    _, subkey = jax.random.split(rng_key)
    nb_init = nb_channels * nb_init_search
    rng_key, noises = leniax_init.register[init_slug](subkey, nb_init, world_size, R, kernels_params[0]['gf_params'])
    all_init_cells = noises.reshape([nb_init_search, 1] + world_size)
    config_target['run_params']['init_cells'] = all_init_cells

    logging.info("Rendering target: start")
    start_time = time.time()
    rng_key, all_cells_target, all_fields_target, all_potentials_target, _ = leniax_helpers.init_and_run(
        rng_key, config_target, use_init_cells=True, with_jit=True, fft=True, stat_trunc=False
    )  # [nb_max_iter, N=1, C, world_dims...]
    total_time = time.time() - start_time
    nb_iter_done = len(all_cells_target)
    logging.info(
        f"Rendering target: {nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps"
    )

    ###
    # Data pipeline
    #
    # Second, we build our tensorflow dataset for our optimization.
    ###
    nb_steps_to_train = 1
    training_data_shape = [-1, nb_channels] + world_size
    # We ensure that the training pipeline is on CPU
    tf.config.experimental.set_visible_devices([], "GPU")
    training_data = {
        'input_cells': all_cells_target[:-nb_steps_to_train].reshape(training_data_shape),
        'target_cells': all_cells_target[nb_steps_to_train:].reshape(training_data_shape),
        'target_fields': all_fields_target[nb_steps_to_train:].reshape(training_data_shape),
        'target_potentials': all_potentials_target[nb_steps_to_train:].reshape(training_data_shape),
    }
    dataset = tf.data.Dataset.from_tensor_slices(training_data)
    bs = 64
    dataset = dataset.repeat(10).shuffle(bs * 2).batch(bs).as_numpy_iterator()

    ###
    # Preparing the training pipeline
    #
    # - We generate a random kernel.
    # - We build the leniax update function.
    # - We initialize our optimizers
    ###
    config_train = copy.deepcopy(config)
    K, mapping = get_kernels_and_mapping(config_train['kernels_params'], world_size, nb_channels, R, fft=False)
    kernel_shape = K.shape
    rng_key, subkey = jax.random.split(rng_key)
    K_params = jax.random.uniform(subkey, shape=kernel_shape)
    K_params = K_params / K_params.sum()

    update_fn = leniax_helpers.build_update_fn(kernel_shape, mapping, 'v1', True, fft=False)
    run_fn = functools.partial(
        leniax_runner.run_diff, nb_steps=nb_steps_to_train, K_fn=K_fn_id, update_fn=update_fn, error_fn=error_fn_cells
    )
    # We build the JAX gradient function
    run_fn_value_and_grads = jax.value_and_grad(run_fn, argnums=(2), has_aux=True)

    gf_params = mapping.get_gf_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    T = jnp.array(config_train['world_params']['T'], dtype=jnp.float32)

    train_log_dir = os.path.join(save_dir, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    K_lr = 1e-4
    K_opt_init, K_opt_update, K_get_params = jax_opt.adam(K_lr)
    K_opt_state = K_opt_init(K_params)
    for i, training_data in enumerate(dataset):
        x = training_data['input_cells']
        y = (training_data['target_cells'], training_data['target_fields'], training_data['target_potentials'])

        # In JAX, you need to manually retrieve the current state of your parameters at each timestep,
        # so you can use them in your update function
        K_params = K_get_params(K_opt_state)

        # We compute the gradients on a batch.
        (loss, rng_key), grads = run_fn_value_and_grads(rng_key, x, K_params, gf_params, kernels_weight_per_channel, T, target=y)

        # Finally we update our parameters
        K_opt_state = K_opt_update(i, grads[0], K_opt_state)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        if i % 20 == 0:
            logging.info(F"Iteration {i}, loss: {loss}")

            np.save(f"{save_dir}/K-{str(i).zfill(4)}.npy", K_params, allow_pickle=True, fix_imports=True)

            fig = plt.figure(figsize=(6, 6))
            axes = []
            axes.append(fig.add_subplot(1, 2, 1))
            axes[-1].title.set_text("kernel K1")
            plt.imshow(K_params[0, 0], cmap='viridis', interpolation="nearest", vmin=0, vmax=K_params.max())
            plt.colorbar()

            K_params_neg_grad = -grads
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
            with train_summary_writer.as_default():
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                tf.summary.image("Training data", image, step=i)
            plt.close(fig)


if __name__ == '__main__':
    run()
