import os
import unittest
import yaml
import jax.numpy as jnp
import numpy as np

from leniax import utils as leniax_utils
from leniax.constant import EPSILON, NB_CHARS

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_val2ch_ch2val(self):
        cells_int_l = np.arange(0, 256).astype(int).tolist()

        tmp = [leniax_utils.val2ch(v) for v in cells_int_l]
        out = [leniax_utils.ch2val(c) for c in tmp]

        np.testing.assert_array_equal(cells_int_l, out)

    def test_val2ch_ch2val_deprecated(self):
        cells_int_l = np.arange(0, 256).astype(int).tolist()

        tmp = [leniax_utils.val2char_deprecated(v) for v in cells_int_l]
        out = [leniax_utils.char2val_deprecated(c) for c in tmp]

        np.testing.assert_array_equal(cells_int_l, out)

    def test_make_array_compressible(self):
        shape = [128, 128]
        cells = np.random.rand(*shape)

        compressible_cells = leniax_utils.make_array_compressible(cells)
        cells_out = leniax_utils.decompress_array(leniax_utils.compress_array(compressible_cells), len(shape))

        np.testing.assert_array_almost_equal(compressible_cells, cells_out)

    def test_compress_decompress_compress(self):
        shape = [7, 7]
        cells = np.random.rand(*shape)

        st = leniax_utils.compress_array(cells)
        cells_out = leniax_utils.decompress_array(st, len(shape))

        np.testing.assert_almost_equal(cells, cells_out, decimal=4)

        new_st = leniax_utils.compress_array(cells)

        assert st == new_st

    def test_compress_decompress_compress_yaml(self):
        max_val = NB_CHARS**2 - 1
        cells = jnp.array(np.arange(0, NB_CHARS**2, dtype=np.int32) / max_val + EPSILON)
        yaml_fullpath = os.path.join(fixture_dir, 'compress_yaml.yaml')

        st = leniax_utils.compress_array(cells)
        with open(yaml_fullpath, 'w') as f:
            yaml.dump({'cells': st}, f)
        with open(yaml_fullpath, 'r') as f:
            st_loads = yaml.load(f)['cells']
        os.remove(yaml_fullpath)
        cells_out = leniax_utils.decompress_array(st_loads, 1)

        np.testing.assert_array_almost_equal(cells, cells_out, decimal=4)

        new_st = leniax_utils.compress_array(cells)

        assert st == new_st

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
