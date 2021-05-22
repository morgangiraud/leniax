import os
import unittest

from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_st2fracs2float(self):
        fracs = '1/2,2/3'
        out = lenia_utils.st2fracs2float(fracs)

        assert out[0] == .5
        assert out[1] == 2 / 3
