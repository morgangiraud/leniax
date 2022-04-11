"""Leniax: Kernel optimization example.

This example shows how to use JAX to learn the original kernel from noise
of an existing simulation.

Usage:
    ``python examples/run.py -cn config_name -cp config_path run_params.nb_init_search=8 run_params.max_run_iter=512``
"""
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
import numpy as np
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import checkpoints, train_state
import matplotlib.pyplot as plt
from typing import Tuple, Callable

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.initializations as leniax_init
from leniax.kernels import get_kernels_and_mapping
from leniax.runner import make_gradient_fn, make_pipeline_fn

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '1c-1k')
config_name = "orbium"


class Step(nn.Module):
    kernel_shape: Tuple[int, ...]
    kernel_init: Callable

    gf_params: jnp.ndarray
    kernels_weight_per_channel: jnp.ndarray
    update_fn: Callable

    @nn.compact
    def __call__(self, rng_key, cells_state, dt):
        K = self.param('K', self.kernel_init, self.kernel_shape)

        state, field, potential = self.update_fn(rng_key, cells_state, K, self.gf_params, self.kernels_weight_per_channel, dt=dt)

        return state, field, potential


def kernel_init(rng_key, kernel_shape):
    K = jax.random.uniform(rng_key, shape=kernel_shape)
    K = K / K.sum()

    return K


def all_states_loss_fn(rng_key, preds, target):
    x = preds['cells']
    x_true = target['cells']

    diffs = 0.5 * jnp.square(x - x_true)
    errors = jnp.sum(diffs, axis=(1, 2, 3))
    loss = jnp.mean(errors)

    return loss, x


def save_checkpoint(save_dir, state):
    step = int(state.step)
    checkpoints.save_checkpoint(save_dir, state, step, keep=2)


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

    nb_init = nb_channels * nb_init_search
    rng_key, noises = leniax_init.register[init_slug](rng_key, nb_init, world_size, R, kernels_params[0]['gf_params'])
    all_init_cells = noises.reshape([nb_init_search, nb_channels] + world_size)
    config_target['run_params']['init_cells'] = all_init_cells

    logging.info("Rendering target: start")
    start_time = time.time()
    rng_key, subkey = jax.random.split(rng_key)
    all_cells_target, all_fields_target, all_potentials_target, _ = leniax_helpers.init_and_run(
        subkey, config_target, use_init_cells=True, with_jit=True, fft=True, stat_trunc=False
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
    dataset = dataset.repeat(4).shuffle(bs * 2).batch(bs).as_numpy_iterator()

    ###
    # Preparing the training pipeline
    #
    # - We generate a random kernel.
    # - We build the leniax update function.
    # - We initialize our optimizers
    ###
    config_train = copy.deepcopy(config)
    T = config['world_params']['T']
    K, mapping = get_kernels_and_mapping(config_train['kernels_params'], world_size, nb_channels, R, fft=False)
    gf_params = mapping.get_gf_params()
    kwpc = mapping.get_kernels_weight_per_channel()
    update_fn = leniax_helpers.build_update_fn(K.shape, mapping, 'v1', True, fft=False)
    train_log_dir = os.path.join(save_dir, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Initialize the model
    steper = Step(K.shape, kernel_init, gf_params, kwpc, update_fn)  # type: ignore
    rng_key, params_key, subkey = jax.random.split(rng_key, 2 + 1)
    variables = steper.init(params_key, subkey, jnp.ones(all_init_cells.shape), 1.)
    vars, params = variables.pop('params')
    del variables  # Delete variables to avoid wasting resources

    # Make the optimizer
    lr = 1e-4
    optimizer = optax.adam(learning_rate=lr)

    # Initialize the state
    t_state = train_state.TrainState.create(
        apply_fn=steper.apply,
        params=params,
        tx=optimizer,
    )

    pipeline_fn = make_pipeline_fn(
        nb_steps_to_train,
        1 / T,
        t_state.apply_fn,
        all_states_loss_fn,
        keep_intermediary_data=False,
        keep_all_timesteps=False
    )
    gradient = make_gradient_fn(pipeline_fn, normalize=True)

    for i, training_data in enumerate(dataset):
        x = training_data['input_cells']
        y = {
            'cells': training_data['target_cells'],
            'fields': training_data['target_fields'],
            'potentials': training_data['target_potentials'],
        }

        # We compute the gradients on a batch.
        rng_key, subkey = jax.random.split(rng_key)
        (loss, _), grads = gradient(subkey, t_state.params, vars, x, y)

        # Finally we update our parameters
        t_state = t_state.apply_gradients(grads=grads)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        if i % 20 == 0:
            logging.info(F"Iteration {i}, loss: {loss}")
            params = t_state.params
            K_params = params['K']

            np.save(f"{save_dir}/K-{str(i).zfill(4)}.npy", K_params, allow_pickle=True, fix_imports=True)

            fig = plt.figure(figsize=(6, 6))
            axes = []
            axes.append(fig.add_subplot(1, 2, 1))
            axes[-1].title.set_text("kernel K1")
            plt.imshow(K_params[0, 0], cmap='viridis', interpolation="nearest", vmin=0, vmax=K_params.max())
            plt.colorbar()

            K_params_neg_grad = -grads['K']
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

        if i % 512 == 0:
            save_checkpoint(save_dir, t_state)


if __name__ == '__main__':
    run()
