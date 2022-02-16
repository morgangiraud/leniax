import os
import unittest
import numpy as np
from hydra import compose, initialize
import pickle

import leniax.utils as leniax_utils
from leniax.helpers import init_and_run

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestPipeline(unittest.TestCase):
    # def test_pipeline_perf(self):
    #     pass

    def test_orbium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=False)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scan(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=True)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scutium(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-scutium-test_last_frame2.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-scutium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=True)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_orbium_scutium_fft(self):
        last_frame_fullpath = f"{fixture_dir}/orbium-scutium-test_last_frame2.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="orbium-scutium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=True, fft=True)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 128
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=4)

    def test_aquarium(self):
        last_frame_fullpath = f"{fixture_dir}/aquarium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="aquarium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=True)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 32
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=3)

    def test_aquarium_fft(self):
        last_frame_fullpath = f"{fixture_dir}/aquarium-test_last_frame.p"
        with initialize(config_path='fixtures'):
            omegaConf = compose(config_name="aquarium-test")
            config = leniax_utils.get_container(omegaConf, fixture_dir)

        seed = config['run_params']['seed']
        rng_key = leniax_utils.seed_everything(seed)

        with open(last_frame_fullpath, "rb") as f:
            true_last_frame = pickle.load(f)

        all_cells, _, _, _ = init_and_run(rng_key, config, with_jit=True, fft=True)
        all_cells = all_cells[:, 0]
        last_frame = all_cells[-1]

        assert len(all_cells) == 32
        np.testing.assert_array_almost_equal(true_last_frame, last_frame, decimal=3)
