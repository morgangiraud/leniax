import os
import unittest
import numpy as np
from hydra import compose, initialize
import pickle

import leniax.helpers as leniax_helpers

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestPipeline(unittest.TestCase):
    # def test_pipeline_perf(self):
    #     pass

    def test_orbium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=False)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scan(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=True)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scutium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-scutium-test_last_frame2.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-scutium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=True)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scutium_fft(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-scutium-test_last_frame2.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-scutium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=True, fft=True)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_aquarium(self):
        last_frame_fullpath = f"{fixture_dir}/aquarium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="aquarium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=True)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 32
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_aquarium_fft(self):
        last_frame_fullpath = f"{fixture_dir}/aquarium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="aquarium-test")
            config = leniax_helpers.get_container(omegaConf)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = leniax_helpers.init_and_run(config, with_jit=True, fft=True)
        last_frame = all_cells[-1, 0]

        assert len(all_cells) == 32
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)
