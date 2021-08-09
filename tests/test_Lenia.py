import os
import unittest
from lenia.Lenia import update_dict

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestLenia(unittest.TestCase):
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
