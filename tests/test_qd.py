import os
import unittest
# from qdpy import containers
# from qdpy.base import ParallelismManager

from lenia.qd import LeniaIndividual, update_dict
# from lenia.qd import LeniaEvo, genBaseIndividual, eval_fn
from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_update_dict(self):
        dic = {'test': {'hip': 2.}}

        key_string = 'test.hop'
        val = 1.
        update_dict(dic, key_string, val)

        target_dict = {'test': {'hip': 2., 'hop': 1.}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test.pim.pouf'
        val = 1.
        update_dict(dic, key_string, val)

        target_dict = {'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test_arr.0.pouf'
        val = 1.
        update_dict(dic, key_string, val)

        target_dict = {'test_arr': [{'pouf': 1}], 'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test_arr.2.woo'
        val = 2
        update_dict(dic, key_string, val)

        target_dict = {'test_arr': [{'pouf': 1}, {}, {'woo': 2}], 'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)

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

    # def test_it(self):
    #     print("\n")
    #     config = {
    #         'kernels_params': {
    #             'k': [{
    #                 'r': 1,
    #                 'b': "1",
    #                 'm': 0.17,
    #                 's': 0.015,
    #                 'h': 1,
    #                 'k_id': 0,
    #                 'gf_id': 0,
    #                 'c_in': 0,
    #                 'c_out': 0,
    #             }]
    #         },
    #         'render_params': {
    #             'win_size_power2': 9,
    #             'pixel_size_power2': 2,
    #             'size_power2': 7,
    #             'pixel_border_size': 0,
    #             'world_size': [128, 128],
    #             'pixel_size': 2**2,
    #         },
    #         'world_params': {
    #             'nb_dims': 2,
    #             'nb_channels': 1,
    #             'R': 13,
    #             'T': 10,
    #         },
    #         'run_params': {
    #             'code': 'test',
    #             'seed': 1,
    #             'max_run_iter': 5,
    #             'nb_init_search': 50,
    #         },
    #         'genotype': [{
    #             'key': 'kernels_params.k.0.m', 'domain': [0., 1.], 'type': 'float'
    #         }, {
    #             'key': 'kernels_params.k.0.s', 'domain': [0., 1.], 'type': 'float'
    #         }, {
    #             'key': 'kernels_params.k.0.r', 'domain': [0.7, 1.3], 'type': 'float'
    #         }]
    #     }
    #     generator_builder = genBaseIndividual(config)
    #     lenia_generator = generator_builder()

    #     fitness_domain = [(0, config['run_params']['max_run_iter'])]
    #     features_domain = [(0., 500.), (0., 500.), (0., 500.), (0., 500.)]
    #     grid = containers.Grid(
    #         shape=[2] * len(features_domain),
    #         max_items_per_bin=1,
    #         fitness_domain=fitness_domain,
    #         features_domain=features_domain
    #     )

    #     algo = LeniaEvo(
    #         container=grid,
    #         budget=1,  # Nb of generated individuals
    #         batch_size=1,  # how many to batch together
    #         dimension=3,  # Number of parameters that can be updated, we don't use it
    #         nb_objectives=None,  # With None, use the container fitness domain
    #         optimisation_task="max",
    #         base_ind_gen=lenia_generator,
    #         tell_container_at_init=False,
    #         add_default_logger=False,
    #         name="lenia-evo",
    #     )

    #     with ParallelismManager("none") as pMgr:
    #         _ = algo.optimise(eval_fn, executor=pMgr.executor, batch_mode=False)

    #     # breakpoint()
