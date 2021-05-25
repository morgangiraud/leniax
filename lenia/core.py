import jax.numpy as jnp
from jax import vmap, lax, jit
from typing import Callable, List, Tuple

from . import utils
from .kernels import get_kernels_and_mapping
from .growth_functions import growth_fns


def init(animal_conf: object, world_size: List[int], nb_channels: int) -> Tuple[jnp.array, jnp.array, object]:
    assert len(world_size) == 2  # We linit ourselves to 2d worlds
    assert nb_channels > 0

    world_params = animal_conf['world_params']
    assert nb_channels == world_params['N_c'], 'The animal and the world are not coming from the same world'

    nb_dims = len(world_size)
    assert nb_dims == world_params['N_d'], 'The animal and the world are not coming from the same world'

    animal_cells_code = animal_conf['cells']
    animal_cells = utils.rle2arr(animal_cells_code, nb_dims, nb_channels)
    cells = init_cells(world_size, nb_channels, [animal_cells])

    kernels_params = animal_conf['kernels_params']
    K, mapping = get_kernels_and_mapping(kernels_params, world_size, nb_channels, world_params["R"])

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
    assert len(cells.shape) == 5

    return cells


def build_update_fn(world_params: List[int], K: jnp.array, mapping: object):
    T = world_params['T']

    get_potential_fn = build_get_potential_generic(K.shape, mapping["true_channels"])
    get_field_fn = build_get_field_generic(mapping)

    def update(cells: jnp.array) -> Tuple[jnp.array, jnp.array, jnp.array]:
        potential = get_potential_fn(cells, K)
        field = get_field_fn(potential)
        cells = update_cells(cells, field, T)

        return cells, field, potential

    return update


def build_get_potential_generic(kernel_shape: List[int], true_channels: List[bool]) -> Callable:
    vconv = vmap(lax.conv, (0, 0, None, None), 0)
    nb_channels = kernel_shape[0]
    depth = kernel_shape[1]
    pad_w = kernel_shape[-1] // 2
    pad_h = kernel_shape[-2] // 2

    def get_potential_generic(cells: jnp.array, K: jnp.array) -> jnp.array:
        H, W = cells.shape[-2:]

        padded_cells = jnp.pad(cells, [(0, 0), (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], mode='wrap')
        conv_out = vconv(padded_cells, K, (1, 1), 'VALID')  # [C, N=1, c=O, H, W]

        conv_out_reshaped = conv_out.reshape(nb_channels * depth, H, W)
        potential = conv_out_reshaped[true_channels]  # [nb_kernels, H, W]

        return potential

    return get_potential_generic


def build_get_field_generic(mapping: object) -> Callable:
    cin_growth_fns = mapping["cin_growth_fns"]
    cout_kernels = mapping["cout_kernels"]
    cout_kernels_h = mapping["cout_kernels_h"]

    def average_per_channel(field, channel_list, weights):
        out = jnp.average(field[channel_list], axis=0, weights=weights)

        return out

    vaverage = vmap(average_per_channel, (None, 0, 0))

    def get_field_generic(potential: jnp.array) -> jnp.array:
        fields = []
        for i in range(len(cin_growth_fns['gf_id'])):
            sub_potential = potential[i]
            gf_id = cin_growth_fns['gf_id'][i]
            m = cin_growth_fns['m'][i]
            s = cin_growth_fns['s'][i]

            fields.append(growth_fns[gf_id](sub_potential, m, s))
        field = jnp.stack(fields)

        field = vaverage(field, cout_kernels, cout_kernels_h)  # [C, H, W]
        field = field[:, jnp.newaxis, jnp.newaxis]

        return field

    return get_field_generic


def update_cells(cells: jnp.array, field: jnp.array, T: float) -> jnp.array:
    dt = 1 / T

    cells_new = cells + dt * field
    cells_new = jnp.clip(cells_new, 0, 1)

    return cells_new


def init_and_run(world_params: object, world_size: List[int], animal_conf: object,
                 max_iter: int) -> Tuple[jnp.array, jnp.array, jnp.array]:
    nb_channels = world_params['N_c']
    cells, K, mapping = init(animal_conf, world_size, nb_channels)

    update_fn = jit(build_update_fn(world_params, K, mapping))

    return run(cells, update_fn, max_iter)


def run(cells: jnp.array, update_fn: Callable, max_iter: int) -> Tuple[jnp.array, jnp.array, jnp.array]:
    all_cells = []
    all_field = []
    all_potential = []
    for i in range(max_iter):
        cells, field, potential = update_fn(cells)

        all_cells.append(cells)
        all_field.append(field)
        all_potential.append(potential)

        if cells.sum() == 0:
            break
        if cells.sum() > 2**11:  # heuristic to detect explosive behaviour
            break

    return all_cells, all_field, all_potential
