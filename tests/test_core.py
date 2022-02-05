import time
import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

from leniax.utils import get_container
from leniax import core as leniax_core
from leniax import helpers as leniax_helpers
from leniax import kernels as leniax_kernels

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestCore(unittest.TestCase):
    def test_update_fn_jit_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = get_container(omegaConf, fixture_dir)

        cells, K, mapping = leniax_helpers.init(config)
        world_params = config['world_params']
        T = jnp.array(world_params['T'])
        update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
        update_fn = leniax_helpers.build_update_fn(K.shape, mapping, update_fn_version)

        cells1 = jnp.ones(cells.shape) * 0.2
        K1 = jnp.ones(K.shape) * 0.3
        gf_params1 = mapping.get_gf_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out1 = update_fn(cells1, K1, gf_params1, kernels_weight_per_channel1, 1. / T)
        out1[0].block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(cells.shape) * 0.5
        K2 = jnp.ones(K.shape) * 0.5
        mapping.cin_gf_params[0][0][0] = .3
        gf_params2 = mapping.get_gf_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2 = update_fn(cells2, K2, gf_params2, kernels_weight_per_channel2, 1. / T)
        out2[0].block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t

    def test_get_potential_jit_perf(self):
        H = 2
        W = 2
        C = 2
        cells = jnp.ones([1, C, H, W])
        cells = cells.at[0, 0].set(0.1)
        cells = cells.at[0, 1].set(0.2)

        nb_kernels = 3  # 0 -> 0, 0 -> 1, 1 -> 1
        max_depth = 2
        K_o = max_depth * C
        K_i = 1
        K_h = 3
        K_w = 3
        K = jnp.ones([K_o, K_i, K_h, K_w])
        K = K.at[0].set(0.1)
        K = K.at[1].set(0.2)
        K = K.at[2].set(0.3)
        K = K.at[3].set(0)

        true_channels = jnp.array([True, True, True, False])

        get_potential = leniax_helpers.build_get_potential_fn(K.shape, true_channels, False)
        jit_fn = jax.jit(get_potential)

        t0 = time.time()
        potential = jit_fn(cells, K)
        potential.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones([1, C, H, W])
        K2 = jnp.ones([K_o, K_i, K_h, K_w]) * 0.2
        t0 = time.time()
        potential2 = jit_fn(cells2, K2)
        potential2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert len(potential.shape) == 4
        assert potential.shape[1] == nb_kernels
        assert potential.shape[2] == H
        assert potential.shape[3] == W

        true_potential = jnp.ones([1, 3, 2, 2])
        true_potential = true_potential.at[0, 0].set(0.09)
        true_potential = true_potential.at[0, 1].set(0.18)
        true_potential = true_potential.at[0, 2].set(0.54)
        np.testing.assert_array_equal(potential, true_potential)

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_get_field_jit_perf(self):
        nb_channels = 2
        nb_kernels = 3
        mapping = leniax_kernels.KernelMapping(nb_channels, nb_kernels)
        mapping.kernels_weight_per_channel = [[.5, .4, .0], [0., 0., .2]]
        mapping.cin_gfs = [['poly_quad4', 'poly_quad4'], ['poly_quad4']]
        mapping.cin_gf_params = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
        average_weight = True

        get_field = leniax_helpers.build_get_field_fn(mapping.cin_gfs, average_weight)
        jit_fn = jax.jit(get_field)

        potential1 = jnp.ones([1, nb_kernels, 25, 25]) * .5
        gf_params1 = mapping.get_gf_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()
        t0 = time.time()
        out1 = jit_fn(potential1, gf_params1, kernels_weight_per_channel1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        potential2 = jnp.ones([1, nb_kernels, 25, 25]) * .5
        mapping.cin_gf_params = [[[0.2, 0.5], [0.3, 0.6]], [[0.4, 0.7]]]
        gf_params2 = mapping.get_gf_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2 = jit_fn(potential2, gf_params2, kernels_weight_per_channel2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_weighted_mean_jit_perf(self):
        jit_fn = jax.jit(leniax_core.weighted_mean)

        nb_kernels = 3
        field1 = jnp.ones([1, nb_kernels, 25, 25]) * .5
        weights1 = jnp.array([[.5, .4, .0], [.5, .2, .0]])

        t0 = time.time()
        out1 = jit_fn(field1, weights1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        field2 = jnp.ones([1, nb_kernels, 25, 25]) * .5
        weights2 = jnp.array([[.1, .2, .0], [.5, .2, .0]])

        t0 = time.time()
        out2 = jit_fn(field2, weights2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_update_state_v1_jit_perf(self):
        jit_fn = jax.jit(leniax_core.update_state)

        world_shape = [25, 25]

        cells1 = jnp.ones(world_shape) * .5
        field1 = jnp.ones(world_shape) * 3
        dt1 = jnp.array(1. / 3.)

        t0 = time.time()
        out1 = jit_fn(cells1, field1, dt1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(world_shape, ) * .2
        field2 = jnp.ones(world_shape) * 3
        dt2 = jnp.array(1. / 6.)

        t0 = time.time()
        out2 = jit_fn(cells2, field2, dt2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t
        np.testing.assert_array_almost_equal(out1, jnp.ones(world_shape))
        np.testing.assert_array_almost_equal(out2, jnp.ones(world_shape) * .7)

    def test_update_state_v2_jit_perf(self):
        jit_fn = jax.jit(leniax_core.update_state_v2)

        world_shape = [25, 25]

        cells1 = jnp.ones(world_shape) * .5
        field1 = jnp.ones(world_shape) * 3
        dt1 = jnp.array(1. / 3.)

        t0 = time.time()
        out1 = jit_fn(cells1, field1, dt1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(world_shape, ) * .2
        field2 = jnp.ones(world_shape) * 3
        dt2 = jnp.array(1. / 6.)

        t0 = time.time()
        out2 = jit_fn(cells2, field2, dt2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        print(delta_t_compiled, delta_t)

        assert delta_t_compiled < 1 / 20 * delta_t
