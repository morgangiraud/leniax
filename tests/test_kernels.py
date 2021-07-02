import os
import unittest
import jax.numpy as jnp
import numpy as np

from lenia import kernels as lenia_kernels

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestKernels(unittest.TestCase):
    def test_remove_zero_dims(self):

        kernels = jnp.array([[
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0.],
            [0., 0., 1., 1.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
        ]])
        true_kernels = jnp.array([[
            [1., 0., 0.],
            [1., 1., 0.],
        ], [
            [0., 1., 1.],
            [1., 0., 0.],
        ]])

        kernels_out = lenia_kernels.remove_zero_dims(kernels)

        np.testing.assert_array_equal(kernels_out, true_kernels)
