import os
import unittest
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

from lenia import helpers as lenia_helpers
from lenia import qd as lenia_qd
from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestHelpers(unittest.TestCase):
    def test_get_mem_optimized_inputs(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_base_config-test")
            base_config = lenia_helpers.get_container(omegaConf)
        world_params = base_config['world_params']
        nb_channels = world_params['nb_channels']
        R = world_params['R']

        nb_kernels = len(base_config['kernels_params']['k'])

        render_params = base_config['render_params']
        world_size = render_params['world_size']

        seed = base_config['run_params']['seed']
        rng_key = lenia_utils.seed_everything(seed)

        lenia_sols = [lenia_qd.LeniaIndividual(base_config, rng_key), lenia_qd.LeniaIndividual(base_config, rng_key)]
        lenia_sols[0][:] = [0.2, 0.02]
        lenia_sols[1][:] = [0.3, 0.03]

        rng_key, run_scan_mem_optimized_parameters = lenia_helpers.get_mem_optimized_inputs(base_config, lenia_sols)

        assert len(run_scan_mem_optimized_parameters) == 7
        all_cells_0_jnp = run_scan_mem_optimized_parameters[0]
        all_Ks_jnp = run_scan_mem_optimized_parameters[1]
        all_gfn_params_jnp = run_scan_mem_optimized_parameters[2]
        all_kernels_weight_per_channel_jnp = run_scan_mem_optimized_parameters[3]

        np.testing.assert_array_equal(all_cells_0_jnp.shape, [2, 3, nb_channels] + world_size)
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, nb_channels * nb_kernels, 1, R * 2 - 1, R * 2 - 1])
        np.testing.assert_array_equal(all_gfn_params_jnp.shape, [2, nb_kernels, 2])
        np.testing.assert_array_equal(all_kernels_weight_per_channel_jnp.shape, [2, 1, nb_kernels])

    def test_update_individuals(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_base_config-test")
            base_config = lenia_helpers.get_container(omegaConf)

        seed = base_config['run_params']['seed']
        rng_key = lenia_utils.seed_everything(seed)

        inds = [lenia_qd.LeniaIndividual(base_config, rng_key), lenia_qd.LeniaIndividual(base_config, rng_key)]
        inds[0][:] = [0.2, 0.02]
        inds[1][:] = [0.3, 0.03]
        Ns = jnp.array([[1, 2, 3], [1, 3, 4]])
        cells0s = jnp.ones([2, 3] + base_config["render_params"]["world_size"])

        new_inds = lenia_helpers.update_individuals(inds, Ns, cells0s)

        assert len(new_inds) == 2
        assert new_inds[0].fitness == 3
        assert new_inds[1].fitness == 4
