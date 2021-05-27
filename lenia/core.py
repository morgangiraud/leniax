from jax import vmap, lax, jit
import jax.numpy as jnp
# import numpy as np
from typing import Callable, List

from . import utils
from .kernels import get_kernels_and_mapping
from .growth_functions import growth_fns


def init(config: dict) -> tuple[jnp.array, jnp.array, dict]:
    nb_dims = config['world_params']['nb_dims']
    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    assert len(world_size) == nb_dims
    assert nb_channels > 0

    cells = config['run_params']['cells']
    if type(cells) is list:
        cells = utils.rle2arr(cells, nb_dims, nb_channels)
    cells = init_cells(world_size, nb_channels, [cells])

    kernels_params = config['kernels_params']['k']
    R = config['world_params']['R']
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)

    return cells, K, mapping


def init_cells(world_size: List[int], nb_channels: int, other_cells: List[jnp.array] = None) -> jnp.array:
    world_shape = [nb_channels] + world_size  # [C, H, W]
    cells = jnp.zeros(world_shape)
    if other_cells is not None:
        for c in other_cells:
            cells = utils.merge_cells(cells, c)

    # 2D convolutions needs an input with shape [N, C, H, W]
    # And we are going to map over first dimension with vmap so we need th following shape
    # [C, N=1, c=1, H, W]
    cells = cells[:, jnp.newaxis, jnp.newaxis]

    return cells


def init_and_run(config: dict) -> tuple[jnp.array, jnp.array, jnp.array]:
    cells, K, mapping = init(config)
    update_fn = jit(build_update_fn(config['world_params'], K, mapping))

    outputs = run(cells, update_fn, config['run_params']['max_run_iter'])

    return outputs


def run(cells: jnp.array, update_fn: Callable, max_run_iter: int) -> tuple[jnp.array, jnp.array, jnp.array]:
    assert max_run_iter > 0, f"max_run_iter must be positive, value given: {max_run_iter} "

    all_cells = [cells]
    all_fields = []
    all_potentials = []
    for i in range(max_run_iter):
        cells, field, potential = update_fn(cells)

        all_cells.append(cells)
        all_fields.append(field)
        all_potentials.append(potential)

        if cells.sum() == 0:
            break
        if cells.sum() > 2**10:  # heuristic to detect explosive behaviour
            break

    # To keep the same number of elements per array
    _, field, potential = update_fn(cells)
    all_fields.append(field)
    all_potentials.append(potential)

    return all_cells, all_fields, all_potentials


def run_init_search(rng_key, config):
    world_params = config['world_params']
    nb_channels = world_params['nb_channels']
    R = world_params['R']

    render_params = config['render_params']
    world_size = render_params['world_size']

    kernels_params = config['kernels_params']['k']

    run_params = config['run_params']
    nb_init_search = run_params['nb_init_search']
    max_run_iter = run_params['max_run_iter']

    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, R)
    update_fn = jit(build_update_fn(world_params, K, mapping))

    rng_key, noises = utils.generate_noise_using_numpy(nb_init_search, nb_channels, rng_key)

    runs = []
    for i in range(nb_init_search):
        cells_0 = init_cells(world_size, nb_channels, [noises[i]])

        all_cells, all_fields, all_potentials = run(cells_0, update_fn, max_run_iter)
        nb_iter_done = len(all_cells)

        runs.append({"N": nb_iter_done, "all_cells": all_cells})

    return rng_key, runs


def build_update_fn(world_params: List[int], K: jnp.array, mapping: dict) -> Callable:
    T = world_params['T']

    get_potential_fn = build_get_potential(K.shape, mapping["true_channels"])
    get_field_fn = build_get_field(mapping)

    def update(cells: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array]:
        potential = get_potential_fn(cells, K)
        field = get_field_fn(potential)
        cells = update_cells(cells, field, T)

        return cells, field, potential

    return update


def build_get_potential(kernel_shape: List[int], true_channels: List[bool]) -> Callable:
    vconv = vmap(lax.conv, (0, 0, None, None), 0)
    nb_channels = kernel_shape[0]
    depth = kernel_shape[1]
    pad_w = kernel_shape[-1] // 2
    pad_h = kernel_shape[-2] // 2

    def get_potential(cells: jnp.array, K: jnp.array) -> jnp.array:
        H, W = cells.shape[-2:]

        padded_cells = jnp.pad(cells, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='wrap')
        conv_out = vconv(padded_cells, K, (1, 1), 'VALID')  # [C, N=1, c=O, H, W]

        conv_out_reshaped = conv_out.reshape(nb_channels * depth, H, W)
        potential = conv_out_reshaped[true_channels]  # [nb_kernels, H, W]

        return potential

    return get_potential


def build_get_field(mapping: dict) -> Callable:
    cin_growth_fns = mapping["cin_growth_fns"]
    cout_kernels = mapping["cout_kernels"]
    cout_kernels_h = mapping["cout_kernels_h"]

    def average_per_channel(field, channel_list, weights):
        out = jnp.average(field[channel_list], axis=0, weights=weights)

        return out

    vaverage = vmap(average_per_channel, (None, 0, 0))

    def get_field(potential: jnp.array) -> jnp.array:
        fields = []
        for i in range(len(cin_growth_fns['gf_id'])):
            sub_potential = potential[i]
            gf_id = cin_growth_fns['gf_id'][i]
            m = cin_growth_fns['m'][i]
            s = cin_growth_fns['s'][i]

            fields.append(growth_fns[gf_id](sub_potential, m, s))
        field = jnp.stack(fields)  # Add a dimension

        field = vaverage(field, cout_kernels, cout_kernels_h)  # [C, H, W]
        field = field[:, jnp.newaxis, jnp.newaxis]

        return field

    return get_field


def update_cells(cells: jnp.array, field: jnp.array, T: float) -> jnp.array:
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)

    return cells_new
