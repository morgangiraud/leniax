import os
import time
import unittest
import jax
import jax.numpy as jnp

from leniax import growth_functions as leniax_gf

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestGrowthFunctions(unittest.TestCase):
    def test_jit_perf(self):
        jitted_poly_quad4 = jax.jit(leniax_gf.poly_quad4)

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
        params = jnp.array([.5, .1])

        t0 = time.time()
        _ = jitted_poly_quad4(params, a)
        delta_t = time.time() - t0

        t0 = time.time()
        _ = jitted_poly_quad4(params, a)
        delta_t_compiled = time.time() - t0

        print(delta_t, delta_t_compiled)

        assert 0.005 * delta_t > delta_t_compiled
