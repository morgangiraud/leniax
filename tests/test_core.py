import os
import unittest
import jax.numpy as jnp
import numpy as np

from lenia import core as lenia_core

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestCore(unittest.TestCase):
    def test_get_potential_generic(self):

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

        get_potential_generic = lenia_core.build_get_potential_generic(K.shape, true_channels)
        potential = get_potential_generic(cells, K)

        assert len(potential.shape) == 3
        assert potential.shape[0] == nb_kernels
        assert potential.shape[1] == H
        assert potential.shape[2] == W

        true_potential = jnp.ones([3, 2, 2])
        true_potential = true_potential.at[0].set(0.09)
        true_potential = true_potential.at[1].set(0.18)
        true_potential = true_potential.at[2].set(0.54)
        np.testing.assert_array_equal(potential, true_potential)
