import os
import unittest
import jax.numpy as jnp
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

    def test_update_individuals(self):
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="qd_config-test")
            qd_config = leniax_utils.get_container(omegaConf, fixture_dir)
        seed = qd_config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        inds = [LeniaIndividual(qd_config, rng_key, [0.2, 0.02]), LeniaIndividual(qd_config, rng_key, [0.3, 0.03])]

        stats = {'N': jnp.array([[1, 2, 3], [1, 3, 4]])}

        new_inds = leniax_qd.update_individuals(inds, stats)

        assert len(new_inds) == 2
        assert new_inds[0].fitness == 3
        assert new_inds[1].fitness == 4
