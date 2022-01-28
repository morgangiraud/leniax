import os
import unittest
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
            qd_config = leniax_utils.get_container(omegaConf, fixture_dir)
        world_params = qd_config['world_params']
        nb_channels = world_params['nb_channels']
        R = world_params['R']

        nb_kernels = len(qd_config['kernels_params'])

        render_params = qd_config['render_params']
        world_size = render_params['world_size']

        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        lenia_sols = [
            LeniaIndividual(qd_config, rng_key, [0.2, 0.02]), LeniaIndividual(qd_config, rng_key, [0.3, 0.03])
        ]

        rng_key, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(
            qd_config, lenia_sols, fft=False
        )

        assert len(run_scan_mem_optimized_parameters) == 5
        all_cells_0_jnp = run_scan_mem_optimized_parameters[0]
        all_Ks_jnp = run_scan_mem_optimized_parameters[1]
        all_gf_params_jnp = run_scan_mem_optimized_parameters[2]
        all_kernels_weight_per_channel_jnp = run_scan_mem_optimized_parameters[3]
        all_Ts_jnp = run_scan_mem_optimized_parameters[4]

        np.testing.assert_array_equal(all_cells_0_jnp.shape, [2, 3, nb_channels] + world_size)
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, nb_channels * nb_kernels, 1, R * 2 - 1, R * 2 - 1])
        np.testing.assert_array_equal(all_gf_params_jnp.shape, [2, nb_kernels, 2])
        np.testing.assert_array_equal(all_kernels_weight_per_channel_jnp.shape, [2, 1, nb_kernels])
        np.testing.assert_array_equal(all_Ts_jnp.shape, [2])

        rng_key, run_scan_mem_optimized_parameters = leniax_helpers.get_mem_optimized_inputs(
            qd_config, lenia_sols, fft=True
        )
        all_Ks_jnp = run_scan_mem_optimized_parameters[1]
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, 1, nb_channels, nb_kernels] + world_size)
