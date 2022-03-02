import jax
import jax.numpy as jnp
import math
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Callable

from leniax.runner import make_pipeline_fn
from leniax.helpers import build_get_potential_fn


def K_init(rng, shape):
    K = jax.random.uniform(rng, shape)
    K = K / K.sum(axis=[-1, -2, -3], keepdims=True)

    return K


def all_states_loss_fn(rng_key, preds, targets):
    x = preds['cells']
    x_true = targets['cells']

    loss = jnp.mean(jnp.square(x - x_true))

    return loss, None


class BenchModel(nn.Module):
    features: Tuple[int, ...]
    K_shape: Tuple[int, int, int, int]
    get_potential_fn: Callable

    @nn.compact
    def __call__(self, rng_key, cells_state, dt=1.0):

        K = self.param('K', K_init, self.K_shape)
        potential = self.get_potential_fn(cells_state, K)

        y = potential
        for feat in self.features[:-1]:
            y = nn.Conv(features=feat, kernel_size=(1, 1))(y)
            y = nn.relu(y)
        field = nn.Conv(features=self.features[-1], kernel_size=(1, 1), kernel_init=nn.initializers.zeros)(y)

        new_cells_state = cells_state + dt * field

        return new_cells_state, potential, field


def make_run_fn(rng_key, config, multiplier):
    config['bench']['nb_k'] *= multiplier
    config['world_params']['R'] = int(config['world_params']['R'] * math.log(multiplier + 2))
    config['world_params']['nb_channels'] *= multiplier
    config['run_params']['nb_init_search'] *= multiplier
    config['run_params']['max_run_iter'] *= multiplier

    R = config['world_params']['R']
    T = config['world_params']['T']
    C = config['world_params']['nb_channels']
    N_init = config['run_params']['nb_init_search']
    world_size = config['render_params']['world_size']
    max_iter = config['run_params']['max_run_iter']
    cells_shape = [N_init] + world_size + [C]

    subkeys = jax.random.split(rng_key)
    init_cells = jax.random.uniform(subkeys[0], [N_init] + world_size + [C])
    targets = {'cells': jax.random.uniform(subkeys[1], [N_init] + world_size + [C])}

    K_shape = (2 * R, 2 * R, 1, 32)
    m = BenchModel(
        features=[4, C],
        K_shape=K_shape,
        get_potential_fn=build_get_potential_fn(K_shape, fft=False, channel_first=False)
    )
    rng_key, subkey = jax.random.split(rng_key)
    variables = m.init(rng_key, subkey, jnp.ones(cells_shape))
    vars, params = variables.pop('params')
    del variables  # Delete variables to avoid wasting resources

    t_state = train_state.TrainState.create(
        apply_fn=m.apply,
        params=params,
        tx=optax.adamw(learning_rate=1e-3),
    )

    pipeline_fn = make_pipeline_fn(
        max_iter, 1 / T, t_state.apply_fn, all_states_loss_fn, keep_intermediary_data=False, keep_all_timesteps=False
    )

    def bench_fn():
        loss, _ = pipeline_fn(subkey, t_state.params, vars, init_cells, targets)
        loss.block_until_ready()
        return loss

    return bench_fn
