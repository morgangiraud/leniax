import os
import unittest
import numpy as np
from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_st2fracs2float(self):
        fracs = '1/2,2/3'
        out = lenia_utils.st2fracs2float(fracs)

        assert out[0] == .5
        assert out[1] == 2 / 3

    def test_val2ch_ch2val(self):
        cells_int_l = np.arange(0, 256).astype(int).tolist()

        tmp = [lenia_utils.val2char(v) for v in cells_int_l]
        out = [lenia_utils.char2val(c) for c in tmp]

        np.testing.assert_array_equal(cells_int_l, out)

    def test_compress_decompress_compress(self):
        shape = [2, 4, 4]
        cells = np.random.rand(*shape)

        st = lenia_utils.compress_array(cells)
        cells_out = lenia_utils.decompress_array(st, len(shape))

        np.testing.assert_almost_equal(cells, cells_out, decimal=2)

        new_st = lenia_utils.compress_array(cells)

        assert st == new_st

    def test_get_unit_distances(self):
        world_size = [3, 3]

        unit_distances = lenia_utils.get_unit_distances(world_size)

        true_unit_distances = [
            [1.414, 1., 1.414],
            [1., 0., 1.],
            [1.414, 1., 1.414],
        ]

        np.testing.assert_almost_equal(unit_distances, true_unit_distances, decimal=3)
