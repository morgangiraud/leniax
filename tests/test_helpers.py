import os
import unittest
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

from leniax import helpers as leniax_helpers
from leniax import utils as leniax_utils
from leniax.lenia import LeniaIndividual

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestHelpers(unittest.TestCase):
    def test_get_mem_optimized_inputs(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_config-test")
            qd_config = leniax_helpers.get_container(omegaConf)
        world_params = qd_config['world_params']
        nb_channels = world_params['nb_channels']
        R = world_params['R']

        nb_kernels = len(qd_config['kernels_params']['k'])

        render_params = qd_config['render_params']
        world_size = render_params['world_size']

        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        lenia_sols = [LeniaIndividual(qd_config, rng_key), LeniaIndividual(qd_config, rng_key)]
        lenia_sols[0][:] = [0.2, 0.02]
        lenia_sols[1][:] = [0.3, 0.03]

        rng_key, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(
            qd_config, lenia_sols, fft=False
        )

        assert len(run_scan_mem_optimized_parameters) == 5
        all_cells_0_jnp = run_scan_mem_optimized_parameters[0]
        all_Ks_jnp = run_scan_mem_optimized_parameters[1]
        all_gfn_params_jnp = run_scan_mem_optimized_parameters[2]
        all_kernels_weight_per_channel_jnp = run_scan_mem_optimized_parameters[3]
        all_Ts_jnp = run_scan_mem_optimized_parameters[4]

        np.testing.assert_array_equal(all_cells_0_jnp.shape, [2, 3, nb_channels] + world_size)
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, nb_channels * nb_kernels, 1, R * 2 - 1, R * 2 - 1])
        np.testing.assert_array_equal(all_gfn_params_jnp.shape, [2, nb_kernels, 2])
        np.testing.assert_array_equal(all_kernels_weight_per_channel_jnp.shape, [2, 1, nb_kernels])
        np.testing.assert_array_equal(all_Ts_jnp.shape, [2])

        rng_key, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(
            qd_config, lenia_sols, fft=True
        )
        all_Ks_jnp = run_scan_mem_optimized_parameters[1]
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, 1, nb_channels, nb_kernels] + world_size)

    def test_update_individuals(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_config-test")
            qd_config = leniax_helpers.get_container(omegaConf)

        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        inds = [LeniaIndividual(qd_config, rng_key), LeniaIndividual(qd_config, rng_key)]
        inds[0][:] = [0.2, 0.02]
        inds[1][:] = [0.3, 0.03]
        stats = {'N': jnp.array([[1, 2, 3], [1, 3, 4]])}
        cells0s = jnp.ones([2, 3] + qd_config["render_params"]["world_size"])

        new_inds = leniax_helpers.update_individuals(inds, stats, cells0s)

        assert len(new_inds) == 2
        assert new_inds[0].fitness == 3
        assert new_inds[1].fitness == 4
