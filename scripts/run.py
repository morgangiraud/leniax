import time
import os
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_path_1c1kv2 = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')
config_path_1c2k = os.path.join(cdir, '..', 'conf', 'species', '1c-2k')
config_path_1c3k = os.path.join(cdir, '..', 'conf', 'species', '1c-3k')
config_path_3c15k = os.path.join(cdir, '..', 'conf', 'species', '3c-15k')
config_path_outputs = os.path.join(cdir, '..', 'outputs', '2021-08-18', '10-50-52', 'c-0000')
config_path_exp = os.path.join(cdir, '..', 'experiments', '007_beta_cube_face1', 'run-b[1.0]', 'c-0002')
config_path_fixtures = os.path.join(cdir, '..', 'tests', 'fixtures')

scale = 1.
stat_trunc = False


@hydra.main(config_path=config_path_1c1k, config_name="orbium")
# @hydra.main(config_path=config_path_1c1k, config_name="vibratium")
# @hydra.main(config_path=config_path_1c2k, config_name="squiggle")
# @hydra.main(config_path=config_path_1c3k, config_name="fish")
# @hydra.main(config_path=config_path_3c15k, config_name="aquarium")
# @hydra.main(config_path=config_path_1c1kv2, config_name="wanderer")
# @hydra.main(config_path=config_path, config_name="orbium-scutium")
# @hydra.main(config_path=config_path_outputs, config_name="config")
# @hydra.main(config_path=config_path_exp, config_name="config")
# @hydra.main(config_path=config_path_fixtures, config_name="aquarium-test.yaml")
def run(omegaConf: DictConfig) -> None:
    config = leniax_helpers.get_container(omegaConf)

    print('Initialiazing and running')
    start_time = time.time()
    all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
        config, scale, with_jit=True, fft=True
    )  # [nb_max_iter, N=1, C, world_dims...]
    if stat_trunc is True:
        all_cells = all_cells[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]
    else:
        all_cells = all_cells[:, 0]  # [nb_iter, C, world_dims...]
    total_time = time.time() - start_time

    nb_iter_done = len(all_cells)
    print(f"{nb_iter_done} frames made in {total_time} seconds: {nb_iter_done / total_time} fps")

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    first_cells = all_cells[0]
    config['run_params']['cells'] = leniax_utils.compress_array(first_cells)
    leniax_utils.save_config(save_dir, config)

    print("Dumping assets")
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    run()
