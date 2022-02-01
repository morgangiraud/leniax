import os
import unittest
import numpy as np

from leniax import kernels as leniax_kernels

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestKernels(unittest.TestCase):
    def test_circle_2d(self):
        k_params = [1., 5., [1.]]
        kf_slug = 'poly_quad'
        kf_params = [4]

        kernel = leniax_kernels.circle_2d(k_params, kf_slug, kf_params)

        np.testing.assert_array_equal(kernel.shape, [1, 10, 10])
