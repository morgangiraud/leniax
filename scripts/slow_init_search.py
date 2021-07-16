import time
import os
from omegaconf import DictConfig
import hydra

from lenia.helpers import slow_init_search, get_container
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_init_search")
def launch(omegaConf: DictConfig) -> None:
    config = get_container(omegaConf)
    print(config)

    rng_key = lenia_utils.seed_everything(config['run_params']['seed'])

    t0 = time.time()
    _, runs = slow_init_search(rng_key, config, with_stats=True)
    # _, runs = search_for_init_parallel(rng_key, config)
    print(f"Init search done in {time.time() - t0}")


if __name__ == '__main__':
    launch()
