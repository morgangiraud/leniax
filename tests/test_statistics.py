import os
import unittest

import jax.numpy as jnp
import numpy as np

from lenia import statistics as lenia_stats

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestStatistics(unittest.TestCase):
    def test_percent_activated_heuristic(self):
        percent_activated = jnp.array([0.1, 0.2, 0.3, 0.4])
        percent_activated_counter = jnp.array([10, 70, 49, 50])

        should_continue_cond, percent_activated_counter_next = lenia_stats.percent_activated_heuristic(
            percent_activated, percent_activated_counter
        )

        np.testing.assert_array_equal(should_continue_cond, [True, True, True, False])
        np.testing.assert_array_equal(percent_activated_counter_next, [0, 0, 50, 51])

    def test_monotonic_heuristic(self):
        mass = jnp.array([1.1, 0.9, 0.9, 0.9])
        previous_mass = jnp.array([1, 1, 1, 1])
        sign = jnp.sign(mass - previous_mass)
        previous_sign = jnp.array([1, 1, -1, -1])
        monotone_counter = jnp.array([70, 80, 30, 80])

        should_continue_cond, monotone_counter_next = lenia_stats.monotonic_heuristic(
            sign, previous_sign, monotone_counter
        )

        np.testing.assert_array_equal(should_continue_cond, [True, True, True, False])
        np.testing.assert_array_equal(monotone_counter_next, [71, 0, 31, 81])
