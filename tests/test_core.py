import time
import os
import unittest
import jax.numpy as jnp
import numpy as np

from lenia import core as lenia_core

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestCore(unittest.TestCase):
    def test_get_potential(self):
        H = 2
        W = 2
        C = 2
        cells = jnp.ones([C, 1, 1, H, W])
        cells = cells.at[0].set(0.1)
        cells = cells.at[1].set(0.2)

        nb_kernels = 3  # 0 -> 0, 0 -> 1, 1 -> 1
        max_depth = 2
        K_o = max_depth
        K_i = 1
        K_h = 3
        K_w = 3
        K = jnp.ones([C, K_o, K_i, K_h, K_w])
        K = K.at[0, 0].set(0.1)
        K = K.at[0, 1].set(0.2)
        K = K.at[1, 0].set(0.3)
        K = K.at[1, 1].set(0)

        true_channels = jnp.array([True, True, True, False])

        get_potential = lenia_core.build_get_potential_fn(K.shape, true_channels)
        potential = get_potential(cells, K)

        assert len(potential.shape) == 3
        assert potential.shape[0] == nb_kernels
        assert potential.shape[1] == H
        assert potential.shape[2] == W

        true_potential = jnp.ones([3, 2, 2])
        true_potential = true_potential.at[0].set(0.09)
        true_potential = true_potential.at[1].set(0.18)
        true_potential = true_potential.at[2].set(0.54)
        np.testing.assert_array_equal(potential, true_potential)

    def test_weighted_select_average_jit_perf(self):
        nb_kernels = 3
        field1 = jnp.ones([nb_kernels, 25, 25]) * .5
        weights1 = jnp.array([[.5, .4, .0], [.5, .2, .0]])

        t0 = time.time()
        _ = lenia_core.weighted_select_average(field1, weights1)
        delta_t = time.time() - t0

        field2 = jnp.ones([nb_kernels, 25, 25]) * .5
        weights2 = jnp.array([[.1, .2, .0], [.5, .2, .0]])

        t0 = time.time()
        _ = lenia_core.weighted_select_average(field2, weights2)
        delta_t_compiled = time.time() - t0

        assert 0.01 * delta_t > delta_t_compiled

    def test_update_cells_jit_perf(self):
        cells1 = jnp.ones([25, 25]) * .5
        field1 = jnp.ones([25, 25]) * 3
        dt1 = 1. / 3.

        t0 = time.time()
        out1 = lenia_core.update_cells(cells1, field1, dt1)
        delta_t = time.time() - t0

        cells2 = jnp.ones([25, 25], ) * .2
        field2 = jnp.ones([25, 25]) * 3
        dt2 = 1. / 6.

        t0 = time.time()
        out2 = lenia_core.update_cells(cells2, field2, dt2)
        delta_t_compiled = time.time() - t0

        assert 0.01 * delta_t > delta_t_compiled
        np.testing.assert_array_almost_equal(out1, jnp.ones([25, 25]))
        np.testing.assert_array_almost_equal(out2, jnp.ones([25, 25]) * .7)
