"""Leniax: Neural Lenia example.

In this example, we will try to replicate the cannonical task of learning the
gecko emoji.

Usage:
    ``python examples/neural_lenia.py -cn config_name -cp config_path``
"""
import functools
import os
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import tensorflow as tf
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as jax_opt
import flax.linen as nn
import PIL.Image

import leniax.utils as leniax_utils
import leniax.core as leniax_core
import leniax.helpers as leniax_helpers
import leniax.runner as leniax_runner

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)


class CA_field(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Lenia is of shape [N, C, world_dims...]
        N = x.shape[0]
        C = x.shape[1]
        x = x.reshape([N, 128, 128, C])

        x = nn.Conv(features=64, kernel_size=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=C // 3, kernel_size=(3, 3), kernel_init=nn.initializers.zeros)(x)

        x = x.reshape([N, C // 3, 128, 128])

        return x


def K_fn_id(K_params):
    return K_params


def loss_fn_cells(rng_key, preds, target):
    all_cells = preds[0]
    all_cells_target = target[0]

    nb_steps = all_cells.shape[0]
    rng_key, subkey = jax.random.split(rng_key)
    iter_idx = jax.random.randint(subkey, [], int(0.8 * nb_steps), nb_steps)

    x = jnp.clip(all_cells[iter_idx:iter_idx + 1, :, :4], 0., 1.)
    # alive = nn.max_pool(x[:, :, 3], (3, 3)) > 0.1
    # x = x * jnp.array(alive, dtype=jnp.float32)

    x_true = all_cells_target[iter_idx:iter_idx + 1]

    diffs = 0.5 * jnp.square(x - x_true)
    mean_loss = jnp.mean(diffs)

    return rng_key, mean_loss


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
    nb_steps = config['run_params']['max_run_iter']
    nb_epochs = config['run_params']['nb_epochs']
    cells_shape = [nb_init, C] + world_size

    ###
    # We define the final kernel K
    ###
    sobel_x = jnp.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=jnp.float32) / 8.
    k_identity = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    k = jnp.stack([sobel_x, sobel_x.T, k_identity])
    K = jnp.repeat(k, C, axis=0)[:, jnp.newaxis]

    # We use a simple init_cells: only pixel set to 1 in the center
    init_cells = jnp.zeros(cells_shape, dtype=jnp.float32)
    midpoint = jnp.array(world_size) // 2
    init_cells = init_cells.at[:, 3:, midpoint[0], midpoint[1]].set(1)

    # We build the simulation function
    model_field = CA_field()

    def get_field_fn(potential, params, kwpc):
        return model_field.apply(params, potential)

    update_fn = functools.partial(  # type: ignore
        leniax_core.update,
        get_potential_fn=leniax_helpers.build_get_potential_fn(K.shape, jnp.array([True] * K.shape[0]), fft=False),
        get_field_fn=get_field_fn,
        get_state_fn=leniax_core.register[config['world_params']['get_state_fn_slug']],
    )
    run_fn = functools.partial(
        leniax_runner.run_diff, nb_steps=nb_steps, K_fn=K_fn_id, update_fn=update_fn, error_fn=loss_fn_cells
    )
    # We build the JAX gradient function
    run_fn_value_and_grads = jax.value_and_grad(run_fn, argnums=(3), has_aux=True)

    train_log_dir = os.path.join(save_dir, 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    lr = 1e-3
    opt_init, opt_update, get_params = jax_opt.adam(lr)
    fake_batch = jnp.ones([nb_init, 3 * C] + world_size)
    gf_params = model_field.init(rng_key, fake_batch)
    opt_state = opt_init(gf_params)

    # We load the target image
    img = PIL.Image.open(os.path.join(config_path, 'emoji_u1f98e.png'))
    img.thumbnail(world_size, PIL.Image.ANTIALIAS)
    img = jnp.float32(img) / 255.
    img = img.reshape([4, 128, 128])[jnp.newaxis, jnp.newaxis]
    img = jnp.repeat(img, repeats=nb_steps, axis=0)
    img = jnp.repeat(img, repeats=nb_init, axis=1)
    x_target = (img, None, None)

    for i in range(nb_epochs):
        # In JAX, you need to manually retrieve the current state of your parameters at each timestep,
        # so you can use them in your update function
        gf_params = get_params(opt_state)

        # We compute the gradients on a batch.
        (loss, rng_key), grads = run_fn_value_and_grads(rng_key, init_cells, K, gf_params, jnp.array([]), 10., target=x_target)

        # Finally we update our parameters
        opt_state = opt_update(i, grads, opt_state)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)

        if i % 20 == 0:
            logging.info(F"Iteration {i}, loss: {loss}")

            x = init_cells
            for j in range(nb_steps + 1):
                out = update_fn(rng_key, x, K, gf_params, [], 1. / T)
                x = out[1]

                if j in [64, 80, 96]:
                    with train_summary_writer.as_default():
                        tf.summary.image(f"Final step training{j}", x[:1, :4].reshape(1, 128, 128, 4), step=i)


if __name__ == '__main__':
    run()
