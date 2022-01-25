import time
import copy
import os
from absl import logging
import json
from omegaconf import DictConfig
import hydra

import leniax.utils as leniax_utils
import leniax.loader as leniax_loader
import leniax.helpers as leniax_helpers

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
collection_dir = os.path.join(cdir, '..', 'outputs', 'collection-01')
metadata_fullpath = os.path.join(collection_dir, 'official_all_metadata.json')

config_path = os.path.join(cdir, '..', 'outputs', 'collection-01', '00-genesis', '0000')
config_name = "config"


@hydra.main(config_path=config_path, config_name=config_name)
def launch(omegaConf: DictConfig) -> None:
    base_config = leniax_utils.get_container(omegaConf, config_path)
    base_config['kernels_params']['k'] = []

    base_save_dir = os.getcwd()

    with open(metadata_fullpath, 'r') as f:
        metadata = json.load(f)

    nb_lenia = len(metadata)
    for i in range(nb_lenia):
        for j in range(i + 1, nb_lenia):
            lenia_one_kernels_params = metadata[0]['config']['kernels_params'][0]
            lenia_two_kernels_params = metadata[1]['config']['kernels_params'][0]
            # The only variables which could be modified here are:
            # - `h` values
            # - `r` values

            tmp_config = copy.deepcopy(base_config)
            tmp_config['kernels_params']['k'].append(lenia_one_kernels_params)
            tmp_config['kernels_params']['k'].append(lenia_two_kernels_params)

            rng_key = leniax_utils.seed_everything(tmp_config['run_params']['seed'])

            t0 = time.time()
            _, best, nb_init_done = leniax_helpers.search_for_init(rng_key, tmp_config, fft=True)
            print(f"Init search done in {time.time() - t0} (nb_inits done: {nb_init_done})")

            all_cells = best['all_cells'][:int(best['N']), 0]
            stats_dict = {k: v.squeeze() for k, v in best['all_stats'].items()}

            print(f"best run length: {best['N']}")

            # changed by hydra
            save_dir = os.path.join(base_save_dir, f"{str(i).zfill(3)}-{str(j).zfill(3)}")
            leniax_utils.check_dir(save_dir)

            tmp_config['run_params']['init_cells'] = leniax_loader.compress_array(all_cells[0])
            tmp_config['run_params']['cells'] = leniax_loader.compress_array(
                leniax_utils.center_and_crop_cells(all_cells[-1])
            )
            leniax_utils.save_config(save_dir, tmp_config)

            print("Dumping assets")
            leniax_helpers.dump_assets(save_dir, tmp_config, all_cells, stats_dict)


if __name__ == '__main__':
    launch()
