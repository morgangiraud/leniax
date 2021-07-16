import time
import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from lenia import core as lenia_core
from lenia import utils as lenia_utils
from lenia import statistics as lenia_stat
from lenia.kernels import KernelMapping

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

    def test_vrun(self):
        config = {
            'world_params': {
                'nb_dims': 2, 'nb_channels': 1, 'R': 13, 'T': 10
            },
            'render_params': {
                'win_size_power2': 9,
                'pixel_size_power2': 2,
                'size_power2': 7,
                'pixel_border_size': 0,
                'world_size': [32, 32],
                'pixel_size': 4
            },
            'kernels_params': {
                'k': [{
                    'r': 1, 'b': '1', 'm': 0.15, 's': 0.015, 'h': 1, 'k_id': 0, 'gf_id': 0, 'c_in': 0, 'c_out': 0
                }]
            },
            'run_params': {
                'code': '26f03e07-eb6c-4fcc-87bb-b4389793b63f',
                'seed': 1,
                'max_run_iter': 32,
                'nb_init_search': 16,
                'cells': 'MISSING'
            }
        }

        nb_init = config['run_params']['nb_init_search']
        cells = jnp.zeros([nb_init, 1] + config['render_params']['world_size'])
        init_cells = np.random.choice(2, size=[nb_init, 1, 3, 3])
        cells = lenia_utils.merge_cells(cells, init_cells)
        cells = cells[:, :, jnp.newaxis, jnp.newaxis, ...]  # [N_init, C, N=1, c=1, H, W]

        K = jnp.array([
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
        ], dtype=jnp.float32)[jnp.newaxis, jnp.newaxis, ...]

        mapping = KernelMapping(config['world_params']['nb_channels'])
        mapping.cin_kernels[0].append(0)
        mapping.cout_kernels[0].append(0)
        mapping.cout_kernels_h[0].append(1)
        mapping.true_channels.append(True)
        mapping.cin_growth_fns['gf_id'].append(0)
        mapping.cin_growth_fns['m'].append(0.15)
        mapping.cin_growth_fns['s'].append(0.015)

        update_fn = lenia_core.build_update_fn(config['world_params'], K, mapping)
        compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])
        max_run_iter = config['run_params']['max_run_iter']

        t0 = time.time()
        runs = []
        max_n = 0
        idx = 0
        for i in range(cells.shape[0]):
            all_cells, _, _, all_stats = lenia_core.run(cells[i], max_run_iter, update_fn, compute_stats_fn)
            nb_iter_done = all_cells.shape[0]  # contains the initialisation

            if max_n < nb_iter_done:
                max_n = nb_iter_done
                idx = i
            runs.append({"N": nb_iter_done, "all_cells": all_cells, "all_stats": all_stats})
        best = runs[idx]["all_cells"][:, :, 0, 0, ...]

        print(time.time() - t0, idx, max_n)

        run_parralel = partial(
            lenia_core.run_parralel, max_run_iter=max_run_iter, update_fn=update_fn, compute_stats_fn=compute_stats_fn
        )
        vrun = jax.vmap(run_parralel, 0, 0)
        # vrun = jax.jit(jax.vmap(run_parralel, 0, 0))
        t0 = time.time()
        # v_all_cells: ndarray with shape [N_init, N_iter + 1, C, N=1, c=1, H, W]
        # v_all_stats: list of N_iter + 1 stat object with N_init values
        v_all_cells, _, _, v_all_stats = vrun(cells)

        all_should_continue = []
        should_continue = jnp.ones(v_all_cells.shape[0])
        init_mass = v_all_cells[:, 0].sum(axis=tuple(range(1, len(v_all_cells.shape) - 1)))
        mass = init_mass.copy()
        sign = jnp.ones_like(init_mass)
        counters = {
            'nb_slow_mass_step': jnp.zeros_like(init_mass),
            'nb_monotone_step': jnp.zeros_like(init_mass),
            'nb_too_much_percent_step': jnp.zeros_like(init_mass),
        }
        for i in range(len(v_all_stats)):
            stats = v_all_stats[i]
            should_continue, mass, sign = lenia_stat.check_heuristics(
                stats, should_continue, init_mass, mass, sign, counters
            )
            all_should_continue.append(should_continue)
        all_should_continue = jnp.array(all_should_continue)
        idx = all_should_continue.sum(axis=0).argmax()
        N_max_iter = int(all_should_continue[:, idx].sum() + 1)
        best_par = v_all_cells[idx, :N_max_iter, :, 0, 0, ...]

        print(time.time() - t0, idx, N_max_iter)

        assert idx == 0
        np.testing.assert_array_equal(best, best_par)

    # def test_weighted_select_average_jit_perf(self):
    #     field1 = jnp.ones([3, 25, 25]) * .5
    #     channel_list1 = jnp.array([[0, 1], [1, 2], [0, 2]])
    #     weights1 = jnp.array([.5, .4, .7])

    #     t0 = time.time()
    #     out1 = lenia_core.weighted_select_average(field1, channel_list1, weights1)
    #     delta_t = time.time() - t0

    #     field2 = jnp.ones([25, 25]) * .2
    #     channel_list2 = jnp.array([[0, 1], [1, 2], [0, 2]])
    #     weights2 = jnp.array([.5, .4, .7])

    #     t0 = time.time()
    #     out2 = lenia_core.weighted_select_average(field2, channel_list2, weights2)
    #     delta_t_compiled = time.time() - t0

    #     assert 0.01 * delta_t > delta_t_compiled

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
