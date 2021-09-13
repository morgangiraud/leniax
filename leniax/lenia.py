import copy
import jax.numpy as jnp
from typing import Dict, List, Tuple

from leniax import utils as leniax_utils


class LeniaIndividual(list):
    # Individual inherits from list
    # The raw parameters can then be set with ind[:] = [param1,  ...]
    #
    # The philosophy of the lib is to have parameters sampled from the same domain
    # And then scaled by custom functions before being used in the evaluation function
    # To sum up:
    #   - All parameters are generated in the sampling_domain
    #   - the dimension parameter is the number of parameter
    #   - in the eval function:
    #       1. You scale those parameters
    #       2. You create the configuration from those parameters
    #       3. You evaluate the configuration
    #       4. you set fitness and features

    fitness: float
    features: List[float]

    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        super().__init__()

        self.qd_config = copy.deepcopy(config)

        self.rng_key = rng_key

    def set_cells(self, cells: str):
        self.qd_config['run_params']['cells'] = cells

    def set_init_cells(self, init_cells: str):
        self.qd_config['run_params']['init_cells'] = init_cells

    def get_config(self) -> Dict:
        if 'genotype' in self.qd_config:
            p_and_ds = self.get_genotype()
            raw_values = list(self)
            assert len(raw_values) == len(p_and_ds)

            to_update = get_update_config(p_and_ds, raw_values)
            config = update_config(self.qd_config, to_update)
        else:
            config = self.qd_config

        return config

    def get_genotype(self):
        return self.qd_config['genotype']


def update_config(old_config, to_update):
    new_config = copy.deepcopy(old_config)
    if 'kernels_params' in to_update:
        for i, kernel in enumerate(to_update['kernels_params']['k']):
            new_config['kernels_params']['k'][i].update(kernel)
    if 'world_params' in to_update:
        new_config['world_params'].update(to_update['world_params'])

    return new_config


def get_update_config(genotype: Dict, raw_values: list) -> Dict:
    to_update: Dict = {}
    for p_and_d, raw_val in zip(genotype, raw_values):
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        val_type = p_and_d['type']
        if val_type == 'float':
            val = float(linear_scale(raw_val, domain))
        elif val_type == 'int':
            d = (domain[0], domain[1] + 1)
            val = int(linear_scale(raw_val, d) - 0.5)
        elif val_type == 'choice':
            d = (0, len(domain))
            val = domain[int(linear_scale(raw_val, d) - 0.5)]
        else:
            raise ValueError(f"type {val_type} unknown")

        leniax_utils.set_param(to_update, key_str, val)

    return to_update


def linear_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] + (domain[1] - domain[0]) * raw_value

    return val


def log_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] * (domain[1] / domain[0])**raw_value

    return val
