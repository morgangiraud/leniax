import time
import random
import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

from lenia.helpers import get_container, get_mem_optimized_inputs
from lenia import core as lenia_core
from lenia import kernels as lenia_kernel
from lenia import statistics as lenia_stat
from lenia import utils as lenia_utils
from lenia import qd as lenia_qd

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestCore(unittest.TestCase):
    def test_run_scan_mem_optimized_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_base_config-test")
            base_config = get_container(omegaConf)

        seed = base_config['run_params']['seed']
        rng_key = lenia_utils.seed_everything(seed)

        nb_lenia = 4

        lenia_sols1 = []
        for _ in range(nb_lenia):
            rng_key, subkey = jax.random.split(rng_key)
            len = lenia_qd.LeniaIndividual(base_config, subkey)
            len[:] = [random.random(), random.random()]
            lenia_sols1.append(len)

        (
            rng_key,
            (
                all_cells_0_jnp1,
                all_Ks_jnp1,
                all_gfn_params_jnp1,
                all_kernels_weight_per_channel_jnp1,
                max_run_iter,
                update_fn,
                compute_stats_fn
            )
        ) = get_mem_optimized_inputs(base_config, lenia_sols1)

        lenia_sols2 = []
        for _ in range(nb_lenia):
            rng_key, subkey = jax.random.split(rng_key)
            len = lenia_qd.LeniaIndividual(base_config, subkey)
            len[:] = [random.random(), random.random()]
            lenia_sols2.append(len)

        (rng_key, (all_cells_0_jnp2, all_Ks_jnp2, all_gfn_params_jnp2, all_kernels_weight_per_channel_jnp2, _, _,
                   _)) = get_mem_optimized_inputs(base_config, lenia_sols2)

        # jax.profiler.start_trace("/tmp/tensorboard")

        t0 = time.time()
        Ns1 = lenia_core.run_scan_mem_optimized(
            all_cells_0_jnp1,
            all_Ks_jnp1,
            all_gfn_params_jnp1,
            all_kernels_weight_per_channel_jnp1,
            max_run_iter,
            update_fn,
            compute_stats_fn
        )
        Ns1.block_until_ready()
        delta_t = time.time() - t0

        t0 = time.time()
        Ns2 = lenia_core.run_scan_mem_optimized(
            all_cells_0_jnp2,
            all_Ks_jnp2,
            all_gfn_params_jnp2,
            all_kernels_weight_per_channel_jnp2,
            max_run_iter,
            update_fn,
            compute_stats_fn
        )
        Ns2.block_until_ready()
        delta_t_compiled = time.time() - t0

        # jax.profiler.stop_trace()

        print(delta_t_compiled, delta_t)

        assert delta_t_compiled < 1 / 20 * delta_t

    def test_run_scan_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = get_container(omegaConf)

        max_run_iter = 32
        cells, K, mapping = lenia_core.init(config)
        update_fn = lenia_core.build_update_fn(config['world_params'], K.shape, mapping)
        compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

        cells1 = jnp.ones(cells.shape) * 0.2
        K1 = jnp.ones(K.shape) * 0.3
        gfn_params1 = mapping.get_gfn_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out1, _, _, _ = lenia_core.run_scan(
            cells1, K1, gfn_params1, kernels_weight_per_channel1, max_run_iter, update_fn, compute_stats_fn
        )
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(cells.shape) * 0.5
        K2 = jnp.ones(K.shape) * 0.5
        mapping.cin_growth_fns['m'][0] = .3
        gfn_params2 = mapping.get_gfn_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2, _, _, _ = lenia_core.run_scan(
            cells2, K2, gfn_params2, kernels_weight_per_channel2, max_run_iter, update_fn, compute_stats_fn
        )
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t

    def test_update_fn_jit_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = get_container(omegaConf)

        cells, K, mapping = lenia_core.init(config)
        update_fn = lenia_core.build_update_fn(config['world_params'], K.shape, mapping)

        cells1 = jnp.ones(cells.shape) * 0.2
        K1 = jnp.ones(K.shape) * 0.3
        gfn_params1 = mapping.get_gfn_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out1 = update_fn(cells1, K1, gfn_params1, kernels_weight_per_channel1)
        out1[0].block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(cells.shape) * 0.5
        K2 = jnp.ones(K.shape) * 0.5
        mapping.cin_growth_fns['m'][0] = .3
        gfn_params2 = mapping.get_gfn_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2 = update_fn(cells2, K2, gfn_params2, kernels_weight_per_channel2)
        out2[0].block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t

    def test_get_potential_jit_perf(self):
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

        true_channels = [True, True, True, False]

        get_potential = lenia_core.build_get_potential_fn(K.shape, true_channels)

        t0 = time.time()
        potential = get_potential(cells, K)
        potential.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones([C, 1, 1, H, W])
        K2 = jnp.ones([C, K_o, K_i, K_h, K_w]) * 0.2
        t0 = time.time()
        potential2 = get_potential(cells2, K2)
        potential2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert len(potential.shape) == 3
        assert potential.shape[0] == nb_kernels
        assert potential.shape[1] == H
        assert potential.shape[2] == W

        true_potential = jnp.ones([3, 2, 2])
        true_potential = true_potential.at[0].set(0.09)
        true_potential = true_potential.at[1].set(0.18)
        true_potential = true_potential.at[2].set(0.54)
        np.testing.assert_array_equal(potential, true_potential)

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_get_field_jit_perf(self):
        nb_channels = 2
        nb_kernels = 3
        mapping = lenia_kernel.KernelMapping(nb_channels, nb_kernels)
        mapping.kernels_weight_per_channel = [[.5, .4, .0], [.5, .2, .0]]
        mapping.cin_growth_fns['gf_id'] = [0, 0, 0]
        mapping.cin_growth_fns['m'] = [0.1, 0.2, 0.3]
        mapping.cin_growth_fns['s'] = [0.1, 0.2, 0.3]

        get_field = lenia_core.build_get_field_fn(mapping.cin_growth_fns)

        potential1 = jnp.ones([nb_kernels, 25, 25]) * .5
        gfn_params1 = mapping.get_gfn_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()
        t0 = time.time()
        out1 = get_field(potential1, gfn_params1, kernels_weight_per_channel1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        potential2 = jnp.ones([nb_kernels, 25, 25]) * .5
        mapping.cin_growth_fns['m'] = [0.2, 0.3, 0.4]
        mapping.cin_growth_fns['s'] = [0.8, 0.7, 0.6]
        gfn_params2 = mapping.get_gfn_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2 = get_field(potential2, gfn_params2, kernels_weight_per_channel2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_weighted_select_average_jit_perf(self):
        jit_fn = jax.jit(lenia_core.weighted_select_average)

        nb_kernels = 3
        field1 = jnp.ones([nb_kernels, 25, 25]) * .5
        weights1 = jnp.array([[.5, .4, .0], [.5, .2, .0]])

        t0 = time.time()
        out1 = jit_fn(field1, weights1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        field2 = jnp.ones([nb_kernels, 25, 25]) * .5
        weights2 = jnp.array([[.1, .2, .0], [.5, .2, .0]])

        t0 = time.time()
        out2 = jit_fn(field2, weights2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 100 * delta_t

    def test_update_cells_v1_jit_perf(self):
        jit_fn = jax.jit(lenia_core.update_cells)

        world_shape = [25, 25]

        cells1 = jnp.ones(world_shape) * .5
        field1 = jnp.ones(world_shape) * 3
        dt1 = 1. / 3.

        t0 = time.time()
        out1 = jit_fn(cells1, field1, dt1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(world_shape, ) * .2
        field2 = jnp.ones(world_shape) * 3
        dt2 = 1. / 6.

        t0 = time.time()
        out2 = jit_fn(cells2, field2, dt2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t
        np.testing.assert_array_almost_equal(out1, jnp.ones(world_shape))
        np.testing.assert_array_almost_equal(out2, jnp.ones(world_shape) * .7)

    def test_update_cells_v2_jit_perf(self):
        jit_fn = jax.jit(lenia_core.update_cells_v2)

        world_shape = [25, 25]

        cells1 = jnp.ones(world_shape) * .5
        field1 = jnp.ones(world_shape) * 3
        dt1 = 1. / 3.

        t0 = time.time()
        out1 = jit_fn(cells1, field1, dt1)
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(world_shape, ) * .2
        field2 = jnp.ones(world_shape) * 3
        dt2 = 1. / 6.

        t0 = time.time()
        out2 = jit_fn(cells2, field2, dt2)
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        print(delta_t_compiled, delta_t)

        assert delta_t_compiled < 1 / 20 * delta_t
