import os
import unittest
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize

from leniax import qd as leniax_qd
from leniax.lenia import LeniaIndividual
from leniax import utils as leniax_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestQD(unittest.TestCase):
    def test_get_config(self):
        config = {
            'kernels_params': [{
                'r': 1,
                'b': "1",
                'm': 0.17,
                's': 0.015,
                'h': 1,
                'k_id': 0,
                'gf_id': 0,
                'c_in': 0,
                'c_out': 0,
            }],
            'genotype': [{
                'key': 'kernels_params.0.m', 'domain': [0., .5], 'type': 'float'
            }, {
                'key': 'kernels_params.0.s', 'domain': [0., 2.], 'type': 'float'
            }]
        }
        rng_key = leniax_utils.seed_everything(1)
        ind = LeniaIndividual(config, rng_key, [0.7, 0.3])

        new_config = ind.get_config()
        true_config = {
            'kernels_params': [{
                'r': 1, 'b': '1', 'm': 0.35, 's': 0.6, 'h': 1, 'k_id': 0, 'gf_id': 0, 'c_in': 0, 'c_out': 0
            }],
            'genotype': [{
                'key': 'kernels_params.0.m', 'domain': [0.0, 0.5], 'type': 'float'
            }, {
                'key': 'kernels_params.0.s', 'domain': [0.0, 2.0], 'type': 'float'
            }]
        }

        self.assertDictEqual(true_config, new_config)

    def test_get_dynamic_args(self):
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

        leniax_sols = [
            LeniaIndividual(qd_config, rng_key, [0.2, 0.02]), LeniaIndividual(qd_config, rng_key, [0.3, 0.03])
        ]

        rng_key, dynamic_args = leniax_qd.get_dynamic_args(qd_config, leniax_sols, fft=False)

        assert len(dynamic_args) == 5
        all_cells0_jnp = dynamic_args[0]
        all_Ks_jnp = dynamic_args[1]
        all_gf_params_jnp = dynamic_args[2]
        all_kernels_weight_per_channel_jnp = dynamic_args[3]
        all_Ts_jnp = dynamic_args[4]

        np.testing.assert_array_equal(all_cells0_jnp.shape, [2, 3, nb_channels] + world_size)
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, nb_channels * nb_kernels, 1, R * 2 - 1, R * 2 - 1])
        np.testing.assert_array_equal(all_gf_params_jnp.shape, [2, nb_kernels, 2])
        np.testing.assert_array_equal(all_kernels_weight_per_channel_jnp.shape, [2, 1, nb_kernels])
        np.testing.assert_array_equal(all_Ts_jnp.shape, [2])

        rng_key, dynamic_args = leniax_qd.get_dynamic_args(qd_config, leniax_sols, fft=True)
        all_Ks_jnp = dynamic_args[1]
        np.testing.assert_array_equal(all_Ks_jnp.shape, [2, 1, nb_channels, nb_kernels] + world_size)

    def test_update_individuals(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_config-test")
            qd_config = leniax_utils.get_container(omegaConf, fixture_dir)
            del qd_config['phenotype']

        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        inds = [LeniaIndividual(qd_config, rng_key, [0.2, 0.02]), LeniaIndividual(qd_config, rng_key, [0.3, 0.03])]

        stats = {
            'N': jnp.array([[1, 2, 3], [1, 3, 4]]),
        }

        new_inds = leniax_qd.update_individuals(inds, stats)

        assert len(new_inds) == 2
        assert new_inds[0].fitness == 3
        assert new_inds[1].fitness == 4
