import os
import unittest

from lenia import utils as lenia_utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_st2fracs(self):
        fracs = '1/2,2/3'
        out = lenia_utils.st2fracs(fracs)

        assert out[0].numerator == 1
        assert out[0].denominator == 2

        assert out[1].numerator == 2
        assert out[1].denominator == 3
