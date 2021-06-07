import logging
import uuid
from typing import Dict
from omegaconf import DictConfig, OmegaConf

from lenia import utils as lenia_utils
from .core import run_init_search


def search_for_init(config: Dict) -> None:
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    run_params = config['run_params']
    # render_params = config['render_params']

    rng_key = lenia_utils.seed_everything(run_params['seed'])
    rng_key, runs = run_init_search(rng_key, config)

    if len(runs) > 0:
        runs.sort(key=lambda run: run["N"], reverse=True)

    return rng_key, runs


def get_container(omegaConf: DictConfig):
    # TODO: Need to check if missing first
    omegaConf.run_params.code = str(uuid.uuid4())

    world_params = omegaConf.world_params
    render_params = omegaConf.render_params
    pixel_size = 2**render_params.pixel_size_power2
    world_size = [2**render_params['size_power2']] * world_params['nb_dims']
    omegaConf.render_params.world_size = world_size
    omegaConf.render_params.pixel_size = pixel_size

    # When we are here, the omegaConf has already been checked by OmegaConf
    # so we can extract primitives to use with other libs
    config = OmegaConf.to_container(omegaConf)
    assert isinstance(config, dict)

    return config
