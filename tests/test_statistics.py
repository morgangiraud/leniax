import os
import unittest

import jax.numpy as jnp
import numpy as np

from leniax import statistics as leniax_stat

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestStatistics(unittest.TestCase):
    def test_mass_volume_heuristic(self):
        R = 13.
        mass_volume = jnp.array([800, 1600, 2400, 3200]) / R**2
        mass_volume_counter = jnp.array([
            10,
            70,
            leniax_stat.MASS_VOLUME_STOP_STEP - 1,
            leniax_stat.MASS_VOLUME_STOP_STEP,
        ])

        should_continue_cond, mass_volume_counter_next = leniax_stat.mass_volume_heuristic(
            mass_volume, mass_volume_counter, R
        )

        np.testing.assert_array_equal(should_continue_cond, [True, True, True, False])
        np.testing.assert_array_equal(mass_volume_counter_next, [1, 1, 128, 129])

    def test_monotonic_heuristic(self):
        mass = jnp.array([1.1, 0.9, 0.9, 0.9])
        previous_mass = jnp.array([1, 1, 1, 1])
        sign = jnp.sign(mass - previous_mass)
        previous_sign = jnp.array([1, 1, -1, -1])
        monotone_counter = jnp.array([
            70,
            leniax_stat.MONOTONIC_STOP_STEP,
            30,
            leniax_stat.MONOTONIC_STOP_STEP,
        ])

        should_continue_cond, monotone_counter_next = leniax_stat.monotonic_heuristic(
            sign, previous_sign, monotone_counter
        )

        np.testing.assert_array_equal(should_continue_cond, [True, True, True, False])
        np.testing.assert_array_equal(monotone_counter_next, [71, 1, 31, 81])
