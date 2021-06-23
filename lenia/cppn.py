import time
import os
import matplotlib.pyplot as plt
import jax
from jax.nn import sigmoid
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.nn.initializers import glorot_normal
from jax.experimental import stax, optimizers
from jax._src.nn.initializers import _compute_fans

from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = os.path.join(cdir, '..', 'tmp')

### 
# SIREN layers
###
def Dense(out_dim, W_init=glorot_normal(), b_init=glorot_normal()):
    """(Custom) Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        # the below line is different from the original jax's Dense
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(
            k2, (input_shape[-1], out_dim)
        )
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W) + b

    return init_fun, apply_fun


def sine_with_gain(gain):
    def func(input):
        return jnp.sin(input * gain)

    return func


def Sine(gain):
    return stax.elementwise(sine_with_gain(gain))

def Sigmoid():
    return stax.elementwise(sigmoid)


### 
# SIREN init
###
def siren_init(omega=30, in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(6 / fan_in) / omega
        return random.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init


def siren_init_first(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = 1 / fan_in
        return random.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init

def bias_uniform(in_axis=-2, out_axis=-1, dtype=jnp.float32):
    # this is what Pytorch default Linear uses.
    def init(key, shape, dtype=dtype):
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        variance = jnp.sqrt(1 / fan_in)
        return random.uniform(
            key, (int(fan_out),), dtype, minval=-variance, maxval=variance
        )

    return init

### 
# SIREN network
###
def create_mlp(input_dim, num_channels, output_dim, omega=30, rng_seed=None):
    modules = []
    modules.append(
        Dense(num_channels[0], W_init=siren_init_first(), b_init=bias_uniform())
    )
    modules.append(Sine(omega))

    for nc in num_channels:
        modules.append(
            Dense(nc, W_init=siren_init(omega=omega), b_init=bias_uniform())
        )
        modules.append(Sine(omega))

    modules.append(
        Dense(output_dim, W_init=siren_init(omega=omega), b_init=bias_uniform())
    )
    modules.append(Sigmoid())
    net_init_random, net_apply = stax.serial(*modules)

    in_shape = (-1, input_dim)  # [bs, input_dim]
    rng = create_random_generator(rng_seed)
    out_shape, net_params = net_init_random(rng, in_shape)

    return net_params, net_apply


def create_random_generator(rng_seed=None):
    if rng_seed is None:
        rng_seed = int(round(time.time()))
    rng = random.PRNGKey(rng_seed)

    return rng


class Siren:
    def __init__(self, input_dim, layers, output_dim, omega, rng_seed=None):
        net_params, net_apply = create_mlp(input_dim, layers, output_dim, omega, rng_seed)

        self.init_params = net_params
        self.net_apply = net_apply

    """ *_p methods are used for optimization """

    def f(self, net_params, x):
        return self.net_apply(net_params, x)


class ImageModel():
    def __init__(self, layers, n_channel, omega, rng_seed=None):
        self.net = Siren(2, layers, n_channel, omega, rng_seed)
        self.net_params = self.net.init_params
        self.loss_func = self.create_loss_func()

    def create_loss_func(self):
        @jit
        def loss_func(net_params, data):
            x = data["input"]
            y = data["output"]
            output = self.net.f(net_params, x)
            return jnp.mean((output - y) ** 2)

        return loss_func

    def update_net_params(self, params):
        self.net_params = params

    def forward(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return self.net.f(self.net_params, x)

###
# Optimizer
###
class JaxOptimizer:
    def __init__(self, model, lr):
        opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

        self.opt_state = opt_init(model.net_params)
        self.get_params = get_params
        
        grad_loss = jit(jax.grad(model.loss_func))

        @jit
        def step(i, _opt_state, data):
            _params = get_params(_opt_state)       
            g = grad_loss(_params, data)
            return opt_update(i, g, _opt_state)

        self._step = step
        self.iter_cnt = 0

    def step(self, data):
        self.opt_state = self._step(self.iter_cnt, self.opt_state, data)
        self.iter_cnt += 1

    def get_optimized_params(self):
        return self.get_params(self.opt_state)

if __name__ == '__main__':
    rng_seed = 1

    # Properties
    width = 128
    height = 128
    layer_outputs_sizes = [10, 10, 1] 
    n_channel = 1
    omega = 30
    batch_dim = width * height  # image size
    feature_dim = 2
    lr = 1e-3

    # Tooling
    model = ImageModel(layer_outputs_sizes, n_channel, omega, rng_seed)
    optimizer = JaxOptimizer(model, lr)

    # Inputs/target
    i = np.linspace(-1, 1, width)
    j = np.linspace(-1, 1, height)
    mesh = np.meshgrid(i, j, indexing="ij")
    xy = np.stack(mesh, axis=-1).reshape(-1, 2)
    
    with open(os.path.join(tmp_dir, 'cells.p'), 'rb') as f:
        cells = np.load(f)
        target_cell = cells[200].reshape(-1, 1)

    data = {
        "input": xy,
        "output": target_cell
    }

    for i in range(2**13):
        optimizer.step(data)
        if i % 64 == 0:
            params = optimizer.get_optimized_params()
            loss = model.loss_func(params, data)
            print(i, loss)

    params = optimizer.get_optimized_params()
    model.update_net_params(params)
    out = model.forward(data["input"])
    
    colormap = plt.get_cmap('plasma')

    out = out.reshape(1, height, width)
    img = lenia_utils.get_image(
        out, 1, 0, colormap
    )
    with open(os.path.join(tmp_dir, f"cppn.png"), 'wb') as f:
        img.save(f, format='png')

    target_out = target_cell.reshape(1, height, width)
    target_img = lenia_utils.get_image(
        target_out, 1, 0, colormap
    )
    with open(os.path.join(tmp_dir, f"cppn_target.png"), 'wb') as f:
        target_img.save(f, format='png')
    