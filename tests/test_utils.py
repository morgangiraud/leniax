import os
import unittest
import jax.numpy as jnp
import numpy as np

from leniax import utils as leniax_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_st2fracs2float(self):
        fracs = '1/2,2/3'
        out = leniax_utils.st2fracs2float(fracs)

        assert out[0] == .5
        assert out[1] == 2 / 3

    def test_get_unit_distances(self):
        world_size = [3, 3]

        unit_distances = leniax_utils.get_unit_distances(world_size)

        true_unit_distances = [
            [1.414, 1., 1.414],
            [1., 0., 1.],
            [1.414, 1., 1.414],
        ]

        np.testing.assert_almost_equal(unit_distances, true_unit_distances, decimal=3)

    def test_crop_zero(self):

        kernels = jnp.array([[
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0.],
            [0., 0., 1., 1.],
            [0., 1., 0., 0.],
            [0., 0., 0., 0.],
        ]])
        true_kernels = jnp.array([[
            [1., 0., 0.],
            [1., 1., 0.],
        ], [
            [0., 1., 1.],
            [1., 0., 0.],
        ]])

        kernels_out = leniax_utils.crop_zero(kernels)

        np.testing.assert_array_equal(kernels_out, true_kernels)

    def test_generate_beta_faces(self):
        nb_faces = 3
        denominator = 3

        faces_list = leniax_utils.generate_beta_faces(nb_faces, denominator)

        assert len(faces_list) == nb_faces
        assert len(faces_list[0]) == (denominator + 1)**(nb_faces - 1)
        assert len(faces_list[1]) == (denominator + 1) * denominator
        assert len(faces_list[2]) == denominator**(nb_faces - 1)

        nb_faces = 4
        faces_list = leniax_utils.generate_beta_faces(nb_faces, denominator)

        assert len(faces_list) == nb_faces
        assert len(faces_list[0]) == (denominator + 1)**(nb_faces - 1)
        assert len(faces_list[1]) == (denominator + 1)**(nb_faces - 2) * denominator
        assert len(faces_list[2]) == (denominator + 1) * denominator**(nb_faces - 2)
        assert len(faces_list[3]) == denominator**(nb_faces - 1)

    def test_set_param(self):
        dic = {'test': {'hip': 2.}}

        key_string = 'test.hop'
        val = 1.
        leniax_utils.set_param(dic, key_string, val)

        target_dict = {'test': {'hip': 2., 'hop': 1.}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test.pim.pouf'
        val = 1.
        leniax_utils.set_param(dic, key_string, val)

        target_dict = {'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test_arr.0.pouf'
        val = 1.
        leniax_utils.set_param(dic, key_string, val)

        target_dict = {'test_arr': [{'pouf': 1}], 'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)

        key_string = 'test_arr.2.woo'
        val = 2
        leniax_utils.set_param(dic, key_string, val)

        target_dict = {'test_arr': [{'pouf': 1}, {}, {'woo': 2}], 'test': {'hip': 2., 'hop': 1., 'pim': {'pouf': 1.}}}
        self.assertDictEqual(dic, target_dict)
