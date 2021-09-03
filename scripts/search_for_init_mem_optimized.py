import time
import os
from absl import logging
import jax
from omegaconf import DictConfig
import hydra

from leniax.lenia import LeniaIndividual
import leniax.utils as leniax_utils
import leniax.helpers as leniax_helpers
import leniax.qd as leniax_qd

logging.set_verbosity(logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
config_path_1c1k = os.path.join(cdir, '..', 'conf', 'species', '1c-1k')
config_path_1c1kv2 = os.path.join(cdir, '..', 'conf', 'species', '1c-1k-v2')


# @hydra.main(config_path=config_path, config_name="config_init_search")
# @hydra.main(config_path=config_path_1c1k, config_name="prototype")
@hydra.main(config_path=config_path_1c1kv2, config_name="wanderer")
def launch(omegaConf: DictConfig) -> None:
    config = leniax_helpers.get_container(omegaConf)
    # config['run_params']['nb_init_search'] = 16
    # config['run_params']['max_run_iter'] = 512
    print(config)

    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])
    lenia_sols = []
    for _ in range(1):
        rng_key, subkey = jax.random.split(rng_key)
        lenia_sols.append(LeniaIndividual(config, subkey))

    eval_fn = leniax_qd.build_eval_lenia_config_mem_optimized_fn(config)

    t0 = time.time()
    results = eval_fn(lenia_sols)
    print(f"Init search done in {time.time() - t0}")

    for id_best, best in enumerate(results):
        print(f"best run length: {best.fitness}")
        config = best.get_config()

        all_cells, _, _, stats_dict = leniax_helpers.init_and_run(config, with_jit=True)
        all_cells = all_cells[:int(stats_dict['N']), 0]

        save_dir = os.path.join(os.getcwd(), f"{str(id_best).zfill(4)}")  # changed by hydra
        leniax_utils.check_dir(save_dir)

        config['run_params']['init_cells'] = leniax_utils.compress_array(
            leniax_utils.center_and_crop_cells(all_cells[0])
        )
        config['run_params']['cells'] = leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
        leniax_utils.save_config(save_dir, config)

        print("Dumping assets")
        leniax_helpers.dump_assets(save_dir, config, all_cells, stats_dict)


if __name__ == '__main__':
    launch()
    # for i in range(49):
    #     config_path_qd = os.path.join('..', 'outputs', 'good-search-2', str(i))
    #     decorator = hydra.main(config_path=config_path_qd, config_name="config")
    #     decorator(launch)()
