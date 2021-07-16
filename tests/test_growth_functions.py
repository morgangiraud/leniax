import os
import time
import unittest
import jax
import jax.numpy as jnp

from lenia import growth_functions as lenia_gf

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestGrowthFunctions(unittest.TestCase):
    def test_jit_perf(self):
        jitted_poly_quad4 = jax.jit(lenia_gf.poly_quad4)

        a = jnp.array([[
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
        m = .5
        s = .1

        t0 = time.time()
        _ = jitted_poly_quad4(a, m, s)
        delta_t = time.time() - t0

        t0 = time.time()
        _ = jitted_poly_quad4(a, .1, .4)
        delta_t_compiled = time.time() - t0

        assert 0.01 * delta_t > delta_t_compiled
