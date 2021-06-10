import random
import copy
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from qdpy.phenotype import Individual

from .api import search_for_init
from . import utils as lenia_utils
# from .statistics import stats_list_to_dict


class genBaseIndividual(object):
    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        self.config = config
        self.rng_key = rng_key

    def __iter__(self):
        return self

    def __next__(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)

        return LeniaIndividual(self.config, subkey)

    def __call__(self):
        while (True):
            yield self.__next__()


class LeniaIndividual(Individual):
    # Individual inherits from list
    # The raw parameters can then be set with ind[:] = [param1,  ...]
    #
    # The philosophy of the lib is to have parameters sampled from the same domain
    # And then scaled by custom functions before being used in the evaluation function
    # To sum up:
    #   - All parameters are generated in the ind_domain
    #   - the dimension parameter is the number of parameter
    #   - in the eval function:
    #       1. You scale those parameters
    #       2. You create the configuration form those parameters
    #       3. You evaluate the configuration
    #       4. you set fitness and features
    def __init__(self, config: Dict, rng_key: jnp.ndarray):
        super().__init__(name="lenia-generator")

        self.base_config = config

        self.rng_key = rng_key

    def set_init_cells(self, init_cells: jnp.ndarray):
        self.base_config['run_params']['cells'] = init_cells

    def get_config(self) -> Dict:
        p_and_ds = self.get_params_and_domains()
        raw_values = list(self)
        assert len(raw_values) == len(p_and_ds)

        to_update = get_update_config(p_and_ds, raw_values)
        config = update_config(self.base_config, to_update)

        return config

    def get_params_and_domains(self):
        return self.base_config['params_and_domains']


def eval_fn(ind: LeniaIndividual, neg_fitness=False) -> LeniaIndividual:
    config = ind.get_config()

    # We have a problem here for reproducibility
    # This function can be called in an other computer and os the for numpy and random
    # Whould be set again here.
    # We use the first element of the jax rng key to do so
    np.random.seed(ind.rng_key[0])
    random.seed(ind.rng_key[0])

    _, runs = search_for_init(ind.rng_key, config, with_stats=False)

    best = runs[0]
    nb_steps = best['N']
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    # all_stats = best['all_stats']

    ind.set_init_cells(lenia_utils.compress_array(init_cells))
    if neg_fitness is True:
        ind.fitness.values = [-nb_steps]
    else:
        ind.fitness.values = [nb_steps]
    ind.features.values = [
        config['kernels_params']['k'][0]['m'],
        config['kernels_params']['k'][0]['s'],
    ]
    ind.features.values = [get_param(config, key_string) for key_string in ind.base_config['phenotype']]
    # stats_dict = stats_list_to_dict(all_stats)
    # all_keys = list(stats_dict.keys())
    # ind.features.values = [jnp.mean(stats_dict[k]) for k in all_keys]

    # print(ind.fitness.values, ind.features.values)

    return ind


def get_param(dic: Dict, key_string: str) -> Any:
    keys = key_string.split('.')
    for i, key in enumerate(keys):
        if key.isdigit():
            dic = dic[int(key)]
        else:
            dic = dic[key]

    return dic


def linear_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] + (domain[1] - domain[0]) * raw_value

    return val


def log_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    val = domain[0] * (domain[1] / domain[0])**raw_value

    return val


def update_dict(dic: Dict, key_string: str, value: Any):
    keys = key_string.split('.')
    for i, key in enumerate(keys[:-1]):
        if key.isdigit():
            if isinstance(dic, list):
                idx = int(key)
                if len(dic) < idx + 1:
                    for _ in range(idx + 1 - len(dic)):
                        dic.append({})
                dic = dic[idx]
            else:
                raise Exception('This key should be an array')
        else:
            if keys[i + 1].isdigit():
                dic = dic.setdefault(key, [{}])
            else:
                dic = dic.setdefault(key, {})

    dic[keys[-1]] = value


def get_update_config(params_and_domains: Dict, raw_values: list) -> Dict:
    to_update: Dict = {}
    for p_and_d, raw_val in zip(params_and_domains, raw_values):
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        type = p_and_d['type']
        if type == 'float':
            val = linear_scale(raw_val, domain)
        elif type == 'int':
            d = (domain[0], domain[1] + 1)
            val = int(linear_scale(raw_val, d) - 0.5)
        elif type == 'choice':
            d = (0, len(domain))
            val = domain[int(linear_scale(raw_val, d) - 0.5)]
        else:
            raise ValueError(f"type {type} unknown")

        update_dict(to_update, key_str, val)

    return to_update


def update_config(old_config, to_update):
    new_config = copy.deepcopy(old_config)
    if 'kernels_params' in to_update:
        for i, kernel in enumerate(to_update['kernels_params']['k']):
            new_config['kernels_params']['k'][i].update(kernel)
    if 'world_params' in to_update:
        new_config['world_params'].update(to_update['world_params'])

    return new_config
