import time
import os
from absl import logging
from omegaconf import DictConfig
import hydra

import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_path_1c1kv2 = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')


@hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
# @hydra.main(config_path=config_path_1c1kv2, config_name="wanderer")
def launch(omegaConf: DictConfig) -> None:
    config = leniax_helpers.get_container(omegaConf)
    # config['run_params']['nb_init_search'] = 16
    print(config)

    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, best, nb_init_done = leniax_helpers.search_for_init(rng_key, config)
    print(f"Init search done in {time.time() - t0} (nb_inits done: {nb_init_done})")

    all_cells = best['all_cells'][:int(best['N']), 0]
    stats_dict = {k: v.squeeze() for k, v in best['all_stats'].items()}

    print(f"best run length: {best['N']}")

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    first_cells = all_cells[0]
    config['run_params']['cells'] = leniax_utils.compress_array(first_cells)
    leniax_utils.save_config(save_dir, config)

    print("Dumping assets")
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
