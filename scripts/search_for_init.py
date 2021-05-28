# import time
import os
from omegaconf import DictConfig
import hydra

from lenia.api import search_for_init

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')


@hydra.main(config_path=config_path, config_name="config_init_search")
def launch(omegaConf: DictConfig) -> None:
    search_for_init(omegaConf)


if __name__ == '__main__':
    launch()
