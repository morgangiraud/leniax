"""Leniax core simulation functions
"""

import functools
import jax
from jax import lax, jit
import jax.numpy as jnp
from typing import Callable, Tuple

GetStateCallableType = Callable[[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                Tuple[jax.random.KeyArray, jnp.ndarray], ]


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
) -> Tuple[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    rng_key, state = get_state_fn(rng_key, state, field, dt)

    return rng_key, state, field, potential


def get_potential_fft(
    state: jnp.ndarray,
    K: jnp.ndarray,
    true_channels: jnp.ndarray,
    max_k_per_channel: int,
    C: int,
    wdims_axes: Tuple[int, ...]
) -> jnp.ndarray:
    """Compute the potential using FFT

    The first dimension of cells and K is the vmap dimension

    Args:
        state: cells state ``[N, nb_channels, world_dims...]``
        K: Kernels ``[1, nb_channels, max_k_per_channel, world_dims...]``
        true_channels: ``1-dim`` array of boolean values ``[nb_channels]``
        max_k_per_channel: Maximum number of kernels applied per channel
        C: nb_channels
        wdims_axes: Dimensions index representing the world dimensions (H, W, etc...)

    Returns:
        An array containing the potential
    """
    fft_cells = jnp.fft.fftn(state, axes=wdims_axes)[:, :, jnp.newaxis, ...]  # [N, nb_channels, 1, world_dims...]
    fft_out = fft_cells * K
    conv_out = jnp.real(jnp.fft.ifftn(fft_out, axes=wdims_axes))  # [N, nb_channels, max_k_per_channel, world_dims...]

    world_shape = state.shape[2:]
    final_shape = (-1, max_k_per_channel * C) + world_shape
    conv_out_reshaped = conv_out.reshape(final_shape)

    potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, world_dims...]

    return potential


def get_potential(state: jnp.ndarray, K: jnp.ndarray, padding: jnp.ndarray, true_channels: jnp.ndarray) -> jnp.ndarray:
    """Compute the potential using lax.conv_general_dilated

    The first dimension of cells and K is the vmap dimension

    Args:
        state: cells state ``[N, nb_channels, world_dims...]``
        K: Kernels ``[K_o=nb_channels * max_k_per_channel, K_i=1, kernel_dims...]``
        padding: array with padding informations, ``[nb_world_dims, 2]``
        true_channels: ``1-dim`` array of boolean values ``[nb_channels]``

    Returns:
        An array containing the potential
    """
    padded_state = jnp.pad(state, padding, mode='wrap')
    nb_channels = state.shape[1]
    conv_out_reshaped = lax.conv_general_dilated(padded_state, K, (1, 1), 'VALID', feature_group_count=nb_channels)
    potential = conv_out_reshaped[:, true_channels]  # [N, nb_kernels, H, W]

    return potential


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
) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Compute the new cells state using the original Lenia formula

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]``

    Returns:
        A 2-tuple representing a jax PRNG key and the new cells state

    Reference:
        https://arxiv.org/abs/1812.05433
    """
    new_state = state + dt * field

    # Straight-through estimator
    clipped_cells = jnp.clip(new_state, 0., 1.)
    zero = new_state - lax.stop_gradient(new_state)
    out = zero + lax.stop_gradient(clipped_cells)

    return rng_key, out


def get_state_v2(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    field: jnp.ndarray,
    dt: jnp.ndarray,
) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Compute the new cells state using the asymptotic Lenia formula

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]``

    Returns:
        A 2-tuple representing a jax PRNG key and the new cells state

    Reference:
        https://direct.mit.edu/isal/proceedings/isal/91/102916
    """
    new_state = state * (1 - dt) + dt * field

    return rng_key, new_state


def get_state_simple(
    rng_key: jax.random.KeyArray,
    state: jnp.ndarray,
    field: jnp.ndarray,
    dt: jnp.ndarray,
) -> Tuple[jax.random.KeyArray, jnp.ndarray]:
    """Compute the new cells state by simply adding the field directly

    Args:
        rng_key: JAX PRNG key.
        state: Current cells state ``[N, nb_channels, world_dims...]``
        field: Current field``[N, nb_channels, world_dims...]``
        dt: Update rate ``[N]`` **(not used)**

    Returns:
        A 2-tuple representing a jax PRNG key and the new cells state
    """
    new_state = state + dt * field

    return rng_key, new_state


register = {
    'v1': get_state,
    'v2': get_state_v2,
    'simple': get_state_simple,
}
