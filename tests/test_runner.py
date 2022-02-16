import time
import random
import os
import unittest
import jax
import jax.numpy as jnp
from hydra import compose, initialize

from leniax import runner as leniax_runner
from leniax import statistics as leniax_stat
from leniax import utils as leniax_utils
from leniax.utils import get_container
from leniax.qd import get_dynamic_args
from leniax.helpers import build_update_fn, init
from leniax.lenia import LeniaIndividual
from leniax.kernels import get_kernels_and_mapping

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestRunner(unittest.TestCase):
    def test_run_scan_mem_optimized_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_config-test")
            qd_config = get_container(omegaConf, fixture_dir)

        R = qd_config['world_params']['R']
        max_run_iter = qd_config['run_params']['max_run_iter']
        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        nb_lenia = 4

        leniax_sols1 = []
        for _ in range(nb_lenia):
            rng_key, subkey = jax.random.split(rng_key)
            len = LeniaIndividual(qd_config, subkey, [random.random(), random.random()])
            leniax_sols1.append(len)

        (rng_key1, dynamic_args1) = get_dynamic_args(qd_config, leniax_sols1)

        leniax_sols2 = []
        for _ in range(nb_lenia):
            rng_key, subkey = jax.random.split(rng_key)
            len = LeniaIndividual(qd_config, subkey, [random.random(), random.random()])
            leniax_sols2.append(len)

        (rng_key2, dynamic_args2) = get_dynamic_args(qd_config, leniax_sols2)

        max_run_iter = qd_config['run_params']['max_run_iter']
        world_params = qd_config['world_params']
        get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
        weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
        render_params = qd_config['render_params']
        kernels_params = qd_config['kernels_params']
        K, mapping = get_kernels_and_mapping(
            kernels_params, render_params['world_size'], world_params['nb_channels'], world_params['R']
        )
        update_fn = build_update_fn(K.shape, mapping, get_state_fn_slug, weighted_average)
        compute_stats_fn = leniax_stat.build_compute_stats_fn(world_params, render_params)

        # jax.profiler.start_trace("/tmp/tensorboard")
        t0 = time.time()
        stats1, _ = leniax_runner.run_scan_mem_optimized(
            rng_key1,
            *dynamic_args1,
            max_run_iter,
            R,
            update_fn,
            compute_stats_fn,
        )
        stats1['N'].block_until_ready()
        delta_t = time.time() - t0

        t0 = time.time()
        stats2, _ = leniax_runner.run_scan_mem_optimized(
            rng_key2,
            *dynamic_args2,
            max_run_iter,
            R,
            update_fn,
            compute_stats_fn,
        )
        stats2['N'].block_until_ready()
        delta_t_compiled = time.time() - t0

        # jax.profiler.stop_trace()

        print(delta_t_compiled, delta_t)

        assert delta_t_compiled < 1 / 20 * delta_t

    def test_run_scan_perf(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = get_container(omegaConf, fixture_dir)

        rng_key = jax.random.PRNGKey(0)

        max_run_iter = 32
        cells, K, mapping = init(config)
        world_params = config['world_params']
        R = world_params['R']
        T = jnp.array(world_params['T'])
        get_state_fn_slug = world_params['get_state_fn_slug'] if 'get_state_fn_slug' in world_params else 'v1'
        update_fn = build_update_fn(K.shape, mapping, get_state_fn_slug)
        compute_stats_fn = leniax_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

        cells1 = jnp.ones(cells.shape) * 0.2
        K1 = jnp.ones(K.shape) * 0.3
        gf_params1 = mapping.get_gf_params()
        kernels_weight_per_channel1 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out1, _, _, _ = leniax_runner.run_scan(
            rng_key, cells1, K1, gf_params1, kernels_weight_per_channel1, T, max_run_iter, R, update_fn, compute_stats_fn
        )
        out1.block_until_ready()
        delta_t = time.time() - t0

        cells2 = jnp.ones(cells.shape) * 0.5
        K2 = jnp.ones(K.shape) * 0.5
        mapping.cin_gf_params[0][0][0] = .3
        gf_params2 = mapping.get_gf_params()
        kernels_weight_per_channel2 = mapping.get_kernels_weight_per_channel()

        t0 = time.time()
        out2, _, _, _ = leniax_runner.run_scan(
            rng_key, cells2, K2, gf_params2, kernels_weight_per_channel2, T, max_run_iter, R, update_fn, compute_stats_fn
        )
        out2.block_until_ready()
        delta_t_compiled = time.time() - t0

        assert delta_t_compiled < 1 / 20 * delta_t
