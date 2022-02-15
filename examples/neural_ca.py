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
import tensorflow as tf
import jax
import numpy as np
import jax.numpy as jnp
import jax.example_libraries.optimizers as jax_opt
import flax.linen as nn
import PIL.Image
from typing import Tuple, Callable

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
from leniax.runner import make_gradient_fn, make_pipeline_fn

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)


def get_jax_living_mask(x):
    alpha = jnp.moveaxis(x[:, 3:4], 1, -1)
    is_alive = nn.max_pool(alpha, (3, 3), (1, 1), 'SAME') > 0.1
    return jnp.moveaxis(is_alive, -1, 1)


class CAModel(nn.Module):
    features: Tuple[int, ...]
    fire_rate: float
    K: jnp.ndarray
    get_potential_fn: Callable

    def setup(self):
        self.conv0 = nn.Conv(features=self.features[0], kernel_size=(1, 1))
        self.conv1 = nn.Conv(features=self.features[1], kernel_size=(1, 1), kernel_init=nn.initializers.zeros)

    def __call__(self, rng_key, x, dt=1.0):
        pre_life_mask = get_jax_living_mask(x)

        potential = self.get_potential_fn(x, self.K)

        y = jnp.moveaxis(potential, 1, -1)
        y = self.conv0(y)
        y = nn.relu(y)
        dx = self.conv1(y)
        dx = jnp.moveaxis(dx, -1, 1)

        rng_key, subkey = jax.random.split(rng_key)
        update_mask = jax.random.uniform(subkey, dx[:, :1, :, :].shape) <= self.fire_rate

        field = dx * jnp.array(update_mask, jnp.float32)
        x += dt * field

        post_life_mask = get_jax_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        return rng_key, x * jnp.array(life_mask, dtype=jnp.float32), potential, field


def all_states_loss_fn(rng_key, preds, targets):
    all_cells = preds[0]  # [N_iter, N_init, C, H, W]
    nb_steps = all_cells.shape[0]
    x_true = targets[0]  # [1, 4, H, W]

    rng_key, subkey = jax.random.split(rng_key)
    iter_idx = jax.random.randint(subkey, [], int(0.8 * nb_steps), nb_steps)
    x = all_cells[iter_idx, :, :4]  # [N_init, 4, H, W]

    diffs = jnp.square(x - x_true)  # [N_init, C, H, W]
    batch_diffs = jnp.mean(diffs, axis=[-3, -2, -1])  # [N_init]
    loss = jnp.mean(batch_diffs)  # []

    return loss, rng_key, x


###
# We use hydra to load Leniax configurations.
# It alloes to do many things among which, we can override configuraiton parameters.
# For example to render a Lenia in a bigger size:
# python examples/run.py render_params.world_size='[512, 512]' world_params.scale=4
###
cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'experiments', '019_neural_lenia')
config_name = "config"


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
    C = config['world_params']['nb_channels']
    T = config['world_params']['T']
    world_size = config['render_params']['world_size']
    nb_init = config['run_params']['nb_init_search']
    max_iter = config['run_params']['max_run_iter']
    nb_epochs = config['run_params']['nb_epochs']
    cells_shape = [nb_init, C] + world_size

    # We load the target image
    img = PIL.Image.open(os.path.join(config_path, 'emoji_u1f98e.png'))
    p = 16
    img.thumbnail([world_size[0] - 2 * p, world_size[1] - 2 * p], PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.
    img = jnp.pad(img, pad_width=[[p, p], [p, p], [0, 0]])
    img = jnp.moveaxis(img, -1, 0)[jnp.newaxis]  # [1, C, H, W]
    targets = (img, None, None)
    ###
    # We define the final kernel K
    ###
    k_dx = jnp.array(np.outer([1, 2, 1], [-1, 0, 1]), dtype=jnp.float32) / 8.0  # Sobel filter
    k_dy = k_dx.T
    k_identity = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    k = jnp.stack([k_identity, k_dx, k_dy])[:, jnp.newaxis]  # [K_o=3, K_i=1, H, W]
    K = jnp.repeat(k, C, axis=0)  # [K_o=3*c, K_i=1, H, W]

    # We use a simple init_cells: only pixel set to 1 in the center
    init_cells = jnp.zeros(cells_shape, dtype=jnp.float32)
    midpoint = jnp.array(world_size) // 2
    init_cells = init_cells.at[:, 3:, midpoint[0], midpoint[1]].set(1.)

    # We build the simulation function
    get_potential_fn = leniax_helpers.build_get_potential_fn(K.shape, fft=False)
    ca = CAModel(features=[128, C], fire_rate=.5, K=K, get_potential_fn=get_potential_fn)
    pipeline_fn = make_pipeline_fn(
        max_iter, 1 / T, ca.apply, all_states_loss_fn, keep_intermediary_data=False, keep_all_timesteps=True
    )
    gradient = make_gradient_fn(pipeline_fn, normalize=True)

    train_log_dir = os.path.join(save_dir, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    lr = 2e-3
    opt_init, opt_update, get_params = jax_opt.adam(lr)
    rng_key, subkey = jax.random.split(rng_key)
    variables = ca.init(rng_key, subkey, jnp.ones(cells_shape))
    vars, params = variables.pop('params')
    del variables  # Delete variables to avoid wasting resources
    opt_state = opt_init(params)

    for i in range(nb_epochs):
        # In JAX, you need to manually retrieve the current state of your parameters at each timestep,
        # so you can use them in your update function
        params = get_params(opt_state)

        # We compute the gradients on a batch.
        (rng_key, loss, rest), grads = gradient(rng_key, params, vars, init_cells, targets)

        # Finally we update our parameters
        opt_state = opt_update(i, grads, opt_state)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        if i % 20 == 0:
            logging.info(F"Iteration {i}, loss: {loss}")

            with train_summary_writer.as_default():
                final_cells = rest[0]
                vis0 = jnp.hstack(jnp.moveaxis(init_cells[:, :4], 1, -1))
                vis1 = jnp.hstack(jnp.moveaxis(final_cells[:, :4], 1, -1))
                vis = np.vstack([vis0, vis1])
                tf.summary.image("Batch", vis[jnp.newaxis], step=i)


if __name__ == '__main__':
    run()
