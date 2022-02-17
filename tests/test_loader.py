import os
import unittest
import yaml
import jax.numpy as jnp
import numpy as np

from leniax import loader as leniax_loader
from leniax.constant import NB_CHARS

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_val2ch_ch2val(self):
        cells_int_l = np.arange(0, 256).astype(int).tolist()

        tmp = [leniax_loader.val2ch(v) for v in cells_int_l]
        out = [leniax_loader.ch2val(c) for c in tmp]

        np.testing.assert_array_equal(cells_int_l, out)

    def test_val2ch_ch2val_deprecated(self):
        cells_int_l = np.arange(0, 256).astype(int).tolist()

        tmp = [leniax_loader.val2char_deprecated(v) for v in cells_int_l]
        out = [leniax_loader.char2val_deprecated(c) for c in tmp]

        np.testing.assert_array_equal(cells_int_l, out)

    def test_compress_decompress_compress(self):
        cells = np.arange(0, 1e5, 100) / 1e5

        st = leniax_loader.compress_array(cells)
        cells_out = leniax_loader.decompress_array(st, 1)

        np.testing.assert_array_almost_equal(cells, cells_out, decimal=4)

        new_st = leniax_loader.compress_array(cells_out)
        cells_out = leniax_loader.decompress_array(new_st, 1)

        np.testing.assert_array_almost_equal(cells, cells_out, decimal=4)

    def test_make_array_compressible(self):
        cells = np.arange(0, 1e5, 100) / 1e5

        compressible_cells = leniax_loader.make_array_compressible(cells)
        cells_out = leniax_loader.decompress_array(leniax_loader.compress_array(compressible_cells), 1)

        np.testing.assert_array_almost_equal(compressible_cells, cells_out, decimal=9)

    def test_compress_decompress_compress_yaml_gzip(self):
        max_val = NB_CHARS**2 - 1
        cells = jnp.array(np.arange(0, NB_CHARS**2, 8, dtype=np.int32) / max_val).reshape(32, 49)
        yaml_fullpath = os.path.join(fixture_dir, 'compress_yaml.yaml')

        st = leniax_loader.compress_array(cells)
        with open(yaml_fullpath, 'w') as f:
            yaml.dump({'cells': st}, f)
        with open(yaml_fullpath, 'r') as f:
            st_loads = yaml.safe_load(f)['cells']
        os.remove(yaml_fullpath)
        cells_out = leniax_loader.decompress_array(st_loads)

        np.testing.assert_array_almost_equal(cells, cells_out, decimal=4)

        new_st = leniax_loader.compress_array(cells)

        assert st == new_st
