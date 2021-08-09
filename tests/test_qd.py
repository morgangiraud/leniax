import os
import unittest

from lenia.lenia import LeniaIndividual
from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestQD(unittest.TestCase):
    def test_get_config(self):
        config = {
            'kernels_params': {
                'k': [{
                    'r': 1,
                    'b': "1",
                    'm': 0.17,
                    's': 0.015,
                    'h': 1,
                    'k_id': 0,
                    'gf_id': 0,
                    'c_in': 0,
                    'c_out': 0,
                }]
            },
            'genotype': [{
                'key': 'kernels_params.k.0.m', 'domain': [0., .5], 'type': 'float'
            }, {
                'key': 'kernels_params.k.0.s', 'domain': [0., 2.], 'type': 'float'
            }]
        }
        rng_key = lenia_utils.seed_everything(1)
        ind = LeniaIndividual(config, rng_key)
        ind[:] = [0.7, 0.3]

        new_config = ind.get_config()
        true_config = {
            'kernels_params': {
                'k': [{
                    'r': 1, 'b': '1', 'm': 0.35, 's': 0.6, 'h': 1, 'k_id': 0, 'gf_id': 0, 'c_in': 0, 'c_out': 0
                }]
            },
            'genotype': [{
                'key': 'kernels_params.k.0.m', 'domain': [0.0, 0.5], 'type': 'float'
            }, {
                'key': 'kernels_params.k.0.s', 'domain': [0.0, 2.0], 'type': 'float'
            }]
        }

        self.assertDictEqual(true_config, new_config)
