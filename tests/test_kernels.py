import os
import unittest

from leniax import kernels as leniax_kernels

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestKernels(unittest.TestCase):
    def test_st2fracs2float(self):
        fracs = '1/2,2/3'
        out = leniax_kernels.st2fracs2float(fracs)

        assert out[0] == .5
        assert out[1] == 2 / 3
