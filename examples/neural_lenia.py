"""Leniax: Neural Lenia example.

In this example, we will try to replicate the cannonical task of learning the
gecko emoji.

Usage:
    ``python examples/neural_lenia.py -cn config_name -cp config_path``
"""
import os
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import PIL.Image
import tensorflow as tf
import jax
import numpy as np
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import checkpoints, train_state
from typing import Tuple, Callable

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.colormaps as leniax_colormaps
from leniax.runner import make_gradient_fn, make_pipeline_fn
from leniax.video import render_video

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)


def get_jax_living_mask(x):
    alpha = jnp.take(x, jnp.array([3]), axis=-1)
    is_alive = nn.max_pool(alpha, (3, 3), (1, 1), 'SAME') > 0.1
    return is_alive


def K_init(rng, shape):
    K = jax.random.uniform(rng, shape)
    K = K / K.sum(axis=[-1, -2, -3], keepdims=True)

    return K


class LeniaModel(nn.Module):
    features: Tuple[int, ...]
    fire_rate: float
    K_shape: Tuple[int, int, int, int]
    get_potential_fn: Callable

    @nn.compact
    def __call__(self, rng_key, cells_state, dt=1.0):
        pre_life_mask = get_jax_living_mask(cells_state)

        K = self.param('K', K_init, self.K_shape)
        potential = self.get_potential_fn(cells_state, K)

        y = potential
        for feat in self.features[:-1]:
            y = nn.Conv(features=feat, kernel_size=(1, 1))(y)
            y = nn.relu(y)
        dx = nn.Conv(features=self.features[-1], kernel_size=(1, 1), kernel_init=nn.initializers.zeros)(y)

        update_mask = jax.random.uniform(rng_key, dx[:, :, :, :1].shape) <= self.fire_rate
        field = dx * jnp.array(update_mask, jnp.float32)

        new_cells_state = cells_state + dt * field

        post_life_mask = get_jax_living_mask(new_cells_state)
        life_mask = pre_life_mask & post_life_mask
        new_cells_state = new_cells_state * jnp.array(life_mask, dtype=jnp.float32)

        return new_cells_state, potential, field


def all_states_loss_fn(rng_key, preds, targets):
    all_cells = preds['cells']  # [N_iter, N_init, H, W, C]
    nb_steps = all_cells.shape[0]
    x_true = targets['cells']  # [1, H, W, 4]

    iter_idx = jax.random.randint(rng_key, [], int(0.8 * nb_steps), nb_steps)
    x = all_cells[iter_idx, :, :, :, :4]  # [N_init, H, W, 4]

    diffs = jnp.square(x - x_true)  # [N_init, C, H, W]
    batch_diffs = jnp.mean(diffs, axis=[-3, -2, -1])  # [N_init]
    loss = jnp.mean(batch_diffs)  # []

    overflow_reg = jnp.mean(jnp.abs(x - jnp.clip(x, 0, 1)))

    return loss + 1e-5 * overflow_reg, all_cells[-1]


def eval_fake_loss_fn(rng_key, preds, targets):
    all_cells = preds['cells']
    single_run = all_cells[:, 0]
    rgb = single_run[:, :, :, :3]
    alpha = jnp.clip(single_run[:, :, :, 3:4], 0, 1)

    loss = 0.

    return loss, 1.0 - alpha + rgb  # white background


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
config_path = os.path.join(cdir, '..', 'experiments', '019_neural_lenia')
config_name = "nlenia"


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

    # We load some parameter from the configuration
    R = config['world_params']['R']
    C = config['world_params']['nb_channels']
    T = config['world_params']['T']
    world_size = config['render_params']['world_size']
    nb_init = config['run_params']['nb_init_search']
    max_iter = config['run_params']['max_run_iter']
    nb_epochs = config['run_params']['nb_epochs']
    cells_shape = [nb_init] + world_size + [C]

    # We load the target image
    img = PIL.Image.open(os.path.join(config_path, 'emoji_u1f98e.png'))
    p = 16
    img.thumbnail([world_size[0] - 2 * p, world_size[1] - 2 * p], PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.
    img[..., :3] *= img[..., 3:]  # premultiply RGB by Alpha
    img = jnp.pad(img, pad_width=[[p, p], [p, p], [0, 0]])
    img = img[jnp.newaxis]  # [1, H, W, C]
    targets = {'cells': img}

    # We use a simple init_cells: only pixel set to 1 in the center
    init_cells = jnp.zeros(cells_shape, dtype=jnp.float32)
    midpoint = jnp.array(world_size) // 2
    init_cells = init_cells.at[:, midpoint[0], midpoint[1], 3:].set(1.)

    train_log_dir = os.path.join(save_dir, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    colormaps = [leniax_colormaps.get(cmap_name) for cmap_name in config['render_params']['colormaps']]

    # Initialize the model
    K_shape = (2 * R, 2 * R, 1, 32)
    ca = LeniaModel(
        features=[128, C],
        fire_rate=.5,
        K_shape=K_shape,
        get_potential_fn=leniax_helpers.build_get_potential_fn(K_shape, fft=False, channel_first=False)
    )
    rng_key, subkey = jax.random.split(rng_key)
    variables = ca.init(rng_key, subkey, jnp.ones(cells_shape))
    vars, params = variables.pop('params')
    del variables  # Delete variables to avoid wasting resources

    # Make the optimizer
    lr = 1e-3
    lr_sched = optax.piecewise_constant_schedule(lr, {5000: 0.1})
    optimizer = optax.adamw(learning_rate=lr_sched)

    # Initialize the state
    t_state = train_state.TrainState.create(
        apply_fn=ca.apply,
        params=params,
        tx=optimizer,
    )

    # We build the simulation function
    train_pipeline_fn = make_pipeline_fn(
        max_iter, 1 / T, t_state.apply_fn, all_states_loss_fn, keep_intermediary_data=False, keep_all_timesteps=True
    )
    eval_pipeline_fn = make_pipeline_fn(
        4 * max_iter, 1 / T, t_state.apply_fn, eval_fake_loss_fn, keep_intermediary_data=False, keep_all_timesteps=True
    )
    gradient = make_gradient_fn(train_pipeline_fn, normalize=True)

    for i in range(nb_epochs):
        # We compute the gradients on a batch.
        rng_key, subkey = jax.random.split(rng_key)
        (loss, aux), grads = gradient(subkey, t_state.params, vars, init_cells, targets)

        # Finally we update our parameters
        t_state = t_state.apply_gradients(grads=grads)

        # We copy some end state to the initial cells
        # To help the network learn to stabilize its outputs
        final_cells = aux[0]
        init_cells = init_cells.at[-4:].set(final_cells[1:5])

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        if i % 20 == 0:
            logging.info(F"Iteration {i}, loss: {loss}")

            with train_summary_writer.as_default():
                vis0 = jnp.hstack(init_cells[:, :, :, :4])
                vis1 = jnp.hstack(final_cells[:, :, :, :4])
                vis = np.vstack([vis0, vis1])
                tf.summary.image("Batch", vis[jnp.newaxis], step=i)

        if i % 500 == 0:
            _, aux = eval_pipeline_fn(subkey, t_state.params, vars, init_cells[:1], targets)
            all_cells = aux[0]
            render_video(
                save_dir,
                jnp.transpose(all_cells, (0, 3, 1, 2)),
                config['render_params'],
                colormaps,
                prefix=f"{str(i).zfill(4)}_neural_lenia"
            )
            save_checkpoint(save_dir, t_state)


if __name__ == '__main__':
    run()
