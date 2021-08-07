import os
import unittest
import numpy as np
from hydra import compose, initialize
import pickle

import lenia.helpers as lenia_helpers

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestPipeline(unittest.TestCase):
    # def test_pipeline_perf(self):
    #     pass

    def test_orbium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = lenia_helpers.get_container(omegaConf)

        all_cells, _, _, _ = lenia_helpers.init_and_run(config)
        all_cells = all_cells[:, 0]

        with open(last_frame_fullpath, "rb") as f:
            last_frame = pickle.load(f)

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(last_frame, all_cells[-1], decimal=4)

    def test_orbium_scutium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-scutium-test_last_frame2.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-scutium-test")
            config = lenia_helpers.get_container(omegaConf)

        all_cells, _, _, _ = lenia_helpers.init_and_run(config)
        all_cells = all_cells[:, 0]

        with open(last_frame_fullpath, "rb") as f:
            last_frame = pickle.load(f)

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(last_frame, all_cells[-1], decimal=4)

    def test_orbium_scan(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = lenia_helpers.get_container(omegaConf)

        all_cells, _, _, _ = lenia_helpers.init_and_run(config, True)
        all_cells = all_cells[:, 0]

        with open(last_frame_fullpath, "rb") as f:
            last_frame = pickle.load(f)

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(last_frame, all_cells[-1], decimal=4)
