import time
import os
import logging
from absl import logging as absl_logging
import jax
from omegaconf import DictConfig
import hydra

from leniax.lenia import LeniaIndividual
import leniax.utils as leniax_utils
import leniax.loader as leniax_loader
import leniax.helpers as leniax_helpers
import leniax.qd as leniax_qd

absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_name = "config_init_search"
# config_name = "config"


@hydra.main(config_path=config_path, config_name=config_name)
def launch(omegaConf: DictConfig) -> None:
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)

    # config['run_params']['nb_init_search'] = 16
    # config['run_params']['max_run_iter'] = 512

    leniax_utils.print_config(config)

    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    # We are looking at multiple inits for one configuration
    nb_sols = 1
    rng_key, *subkeys = jax.random.split(rng_key, 1 + nb_sols)
    leniax_sols = [LeniaIndividual(config, subkey, []) for subkey in subkeys]

    t0 = time.time()
    results = leniax_qd.build_eval_lenia_config_mem_optimized_fn(config)(leniax_sols)
    logging.info(f"Init search done in {time.time() - t0}")

    for id_best, best in enumerate(results):
        logging.info(f"best run length: {best.fitness}")
        config = best.get_config()

        all_cells, _, _, stats_dict = leniax_helpers.init_and_run(config, with_jit=True, stat_trunc=True)
        all_cells = all_cells[:, 0]

        save_dir = os.path.join(os.getcwd(), f"{str(id_best).zfill(4)}")  # changed by hydra
        leniax_utils.check_dir(save_dir)

        config['run_params']['init_cells'] = leniax_loader.compress_array(
            leniax_utils.center_and_crop_cells(all_cells[0])
        )
        config['run_params']['cells'] = leniax_loader.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
        leniax_utils.save_config(save_dir, config)

        logging.info("Dumping assets")
        leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
