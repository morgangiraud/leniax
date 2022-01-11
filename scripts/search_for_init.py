import time
import os
from absl import logging
from omegaconf import DictConfig
import hydra

import leniax.utils as leniax_utils
import leniax.loader as leniax_loader
import leniax.helpers as leniax_helpers

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '2d', '2c-3k')
config_name = "config"


@hydra.main(config_path=config_path, config_name=config_name)
def launch(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)

    config['render_params']['pixel_size_power2'] = 0
    config['render_params']['pixel_size'] = 1
    config['render_params']['size_power2'] = 7
    config['render_params']['world_size'] = [128, 128]
    config['world_params']['scale'] = 1.
    config['run_params']['max_run_iter'] = 4096
    config['run_params']['nb_init_search'] = 256

    leniax_utils.print_config(config)

    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, best, nb_init_done = leniax_helpers.search_for_init(rng_key, config, fft=True)
    print(f"Init search done in {time.time() - t0} (nb_inits done: {nb_init_done})")

    all_cells = best['all_cells'][:int(best['N']), 0]
    stats_dict = {k: v.squeeze() for k, v in best['all_stats'].items()}

    print(f"best run length: {best['N']}")

    save_dir = os.getcwd()  # changed by hydra
    leniax_utils.check_dir(save_dir)

    config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
    config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
    leniax_utils.save_config(save_dir, config)

    print("Dumping assets")
    leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
