"""Leniax core simulation functions
"""

import functools
import jax
from jax import lax, jit
import jax.numpy as jnp
from typing import Callable, Tuple, Optional

GetStateCallableType = Callable[[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


@functools.partial(jit, static_argnums=(6, 7, 8))
def update(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    K: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    dt: jnp.ndarray,
    get_potential_fn: Callable,
    get_field_fn: Callable,
    get_state_fn: GetStateCallableType,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update the cells state

    Jitted function with static argnums. Use functools.partial to set the different function.
    Avoid changing non-static argument shape for performance.

    Args:
        rng_key: JAX PRNG key.
        state: cells state ``[N, nb_channels, world_dims...]``
        K: Kernel ``[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]``
        gf_params: Growth function parmaeters ``[nb_kernels, params_shape...]``
        kernels_weight_per_channel: Kernels weight used in the averaginf function ``[nb_channels, nb_kernels]``
        dt: Update rate ``[N]``
        get_potential_fn: **(jit static arg)** Function used to compute the potential
        get_field_fn: **(jit static arg)** Function used to compute the field
        get_state_fn: **(jit static arg)** Function used to compute the new cell state

    Returns:
        A tuple of arrays representing a jax PRNG key and the updated state, the used potential and used field.
    """

    potential = get_potential_fn(state, K)
    field = get_field_fn(potential, gf_params, kernels_weight_per_channel)
    state = get_state_fn(rng_key, state, field, dt)

    return state, field, potential


@functools.partial(jit, static_argnums=(2, 3))
def get_potential_fft(
    state: jnp.ndarray,
    K: jnp.ndarray,
    tc_indices: Optional[Tuple] = None,
    channel_first: bool = True,
) -> jnp.ndarray:
    """Compute the potential using FFT

    The first dimension of cells and K is the vmap dimension

    Args:
        state: cells state ``[N, nb_channels, world_dims...]``
        K: Kernels ``[1, nb_channels, max_k_per_channel, world_dims...]``
        tc_indices: Optional ``1-dim`` array channels indices to keep
        channel_first: see ``https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html``

    Returns:
        An array containing the potential
    """
    dim_numbers = _conv_dimension_numbers(state.shape, channel_first)
    channel_dim = dim_numbers.lhs_spec[1]
    C = state.shape[channel_dim]

    if channel_first is True:
        world_dims = tuple(range(-1, -len(state.shape) + 1, -1))
    else:
        world_dims = tuple(range(-2, -len(state.shape), -1))

    fft_cells = jnp.fft.fftn(state, axes=world_dims)
    if channel_first is True:
        fft_cells = fft_cells[:, :, jnp.newaxis]  # [N, nb_channels, 1, world_dims...]
    else:
        fft_cells = fft_cells[..., jnp.newaxis]  # [N, world_dims..., nb_channels, 1]
    fft_out = fft_cells * K
    conv_out = jnp.real(jnp.fft.ifftn(fft_out, axes=world_dims))

    if channel_first is True:
        max_k_per_channel = K.shape[2]
        conv_shape = (-1, max_k_per_channel * C) + state.shape[2:]
    else:
        max_k_per_channel = K.shape[-1]
        conv_shape = state.shape[0:-1] + (max_k_per_channel * C, )
    conv_out_reshaped = conv_out.reshape(conv_shape)

    if tc_indices is not None:
        potential = jnp.take(conv_out_reshaped, jnp.array(tc_indices), axis=channel_dim)
    else:
        potential = conv_out_reshaped

    return potential


@functools.partial(jit, static_argnums=(2, 3, 4))
def get_potential(
    state: jnp.ndarray,
    K: jnp.ndarray,
    padding: Tuple,
    tc_indices: Optional[Tuple] = None,
    channel_first: bool = True,
) -> jnp.ndarray:
    """Compute the potential using lax.conv_general_dilated

    The first dimension of cells and K is the vmap dimension

    Args:
        state: cells state ``[N, nb_channels, world_dims...]``
        K: Kernels ``[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]``
        padding: array with padding informations, ``[nb_world_dims, 2]``
        tc_indices: Optional ``1-dim`` array channels indices to keep
        channel_first: see ``https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html``

    Returns:
        An array containing the potential
    """
    dim_numbers = _conv_dimension_numbers(state.shape, channel_first)
    channel_dim = dim_numbers.lhs_spec[1]
    C = state.shape[channel_dim]

    padded_state = jnp.pad(state, padding, mode='wrap')
    conv_out_reshaped = lax.conv_general_dilated(
        padded_state,
        K,
        (1, 1),
        'VALID',
        feature_group_count=C,
        dimension_numbers=dim_numbers,
    )

    if tc_indices is not None:
        potential = jnp.take(conv_out_reshaped, jnp.array(tc_indices), axis=channel_dim)
    else:
        potential = conv_out_reshaped

    return potential


def _conv_dimension_numbers(input_shape, channel_first=True):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    if channel_first is True:
        lhs_spec = tuple(range(0, ndim))
        rhs_spec = tuple(range(0, ndim))
    else:
        lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
        rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec

    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


@functools.partial(jit, static_argnums=(3, 4))
def get_field(
    potential: jnp.ndarray,
    gf_params: jnp.ndarray,
    kernels_weight_per_channel: jnp.ndarray,
    growth_fn_t: Tuple[Callable, ...],
    weighted_fn: Callable
) -> jnp.ndarray:
    """Compute the field

    Jitted function with static argnums. Use functools.partial to set the different function.
    Avoid changing non-static argument shape for performance.

    Args:
        potential: ``[N, nb_kernels, world_dims...]``
        gf_params: ``[nb_kernels, nb_gf_params]``
        kernels_weight_per_channel: Kernels weight used in the averaginf function ``[nb_channels, nb_kernels]``
        growth_fn_t: **(jit static arg)** Tuple of growth functions. ``length: nb_kernels``
        weighted_fn: **(jit static arg)** Function used to merge fields linked to the same channel

    Returns:
        An array containing the field
    """
    fields = []
    for i in range(len(growth_fn_t)):
        sub_potential = potential[:, i]
        current_gf_params = gf_params[i]
        growth_fn = growth_fn_t[i]

        sub_field = growth_fn(current_gf_params, sub_potential)

        fields.append(sub_field)
    fields_jnp = jnp.stack(fields, axis=1)  # [N, nb_kernels, world_dims...]

    fields_jnp = weighted_fn(fields_jnp, kernels_weight_per_channel)  # [N, C, H, W]

    return fields_jnp


def weighted_sum(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute the weighted sum of sub fields

    Args:
        fields: Raw sub fields ``[N, nb_kernels, world_dims...]``
        weights: Weights used to compute the sum ``[nb_channels, nb_kernels]`` 0.
            values are used to indicate that a given channels does not receive inputs from this kernel

    Returns:
        The unnormalized field
    """
    N = fields.shape[0]
    nb_kernels = fields.shape[1]
    world_dims = list(fields.shape[2:])
    nb_channels = weights.shape[0]

    fields_swapped = fields.swapaxes(0, 1)  # [nb_kernels, N, world_dims...]
    fields_reshaped = fields_swapped.reshape(nb_kernels, -1)

    out_tmp = jnp.matmul(weights, fields_reshaped)
    out_tmp = out_tmp.reshape([nb_channels, N] + world_dims)
    fields_out = out_tmp.swapaxes(0, 1)  # [N, nb_channels, world_dims...]

    return fields_out


def weighted_mean(fields: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute the weighted mean of sub fields

    Args:
        fields: Raw sub fields ``[N, nb_kernels, world_dims...]``
        weights: Weights used to compute the sum ``[nb_channels, nb_kernels]`` 0.
            values are used to indicate that a given channels does not receive inputs from this kernel

    Returns:
        The normalized field
    """
    fields_out = weighted_sum(fields, weights)
    fields_normalized = fields_out / weights.sum(axis=1)[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

    return fields_normalized


def get_state(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    field: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the new cells state using the original Lenia formula

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]``

    Returns:
        The new cells state

    Reference:
        https://arxiv.org/abs/1812.05433
    """
    new_state = state + dt * field

    # Straight-through estimator
    clipped_cells = jnp.clip(new_state, 0., 1.)
    zero = new_state - lax.stop_gradient(new_state)
    out = zero + lax.stop_gradient(clipped_cells)

    return out


def get_state_v2(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    field: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the new cells state using the asymptotic Lenia formula

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]``

    Returns:
        The new cells state

    Reference:
        https://direct.mit.edu/isal/proceedings/isal/91/102916
    """
    new_state = state * (1 - dt) + dt * field

    return new_state


def get_state_simple(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    field: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the new cells state by simply adding the field directly

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]`` **(not used)**

    Returns:
        The new cells state
    """
    new_state = state + dt * field

    return new_state


register = {
    'v1': get_state,
    'v2': get_state_v2,
    'simple': get_state_simple,
}
