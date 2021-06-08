import random
import copy
from functools import partial
from typing import Dict, Any
import jax.numpy as jnp

from qdpy.base import DomainLike
from qdpy.phenotype import Individual
from qdpy.containers import Container
from qdpy.algorithms import Evolution, QDAlgorithm
from qdpy import tools

from .api import search_for_init
from . import utils as lenia_utils
# from .statistics import stats_list_to_dict


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


class genBaseIndividual(object):
    def __init__(self, config: Dict):
        self.config = config

    def __iter__(self):
        return self

    def __next__(self):
        return LeniaIndividual(self.config)

    def __call__(self):
        while (True):
            yield self.__next__()


class LeniaIndividual(Individual):
    def __init__(self, config: Dict):
        super().__init__(name="lenia-generator")

        self.config = config

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and self.config == other.config)

    def get_config(self) -> Dict:
        return self.config

    def set_init_cells(self, init_cells: jnp.ndarray):
        self.config['run_params']['cells'] = init_cells

    def get_params_and_domains(self):
        return self.config['params_and_domains']

    def set_params_and_domains(self, paramsAndDomains):
        self.config['params_and_domains'] = paramsAndDomains


def get_update_config(params_and_domains: Dict) -> Dict:
    to_update: Dict = {}
    for p_and_d in params_and_domains:
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        type = p_and_d['type']
        if type == 'float':
            val = random.uniform(domain[0], domain[1])
        elif type == 'int':
            val = random.randint(domain[0], domain[1])
        elif type == 'choice':
            val = random.choice(domain)
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


class LeniaRandomUniform(QDAlgorithm):
    def __init__(self, container: Container, budget: int, **kwargs):
        super().__init__(container, budget, **kwargs)

    def _internal_ask(self, base_ind: LeniaIndividual) -> LeniaIndividual:
        lenia_ind = random_ind(base_ind)

        return lenia_ind


def random_ind(base_ind: LeniaIndividual) -> LeniaIndividual:
    params_and_domains = base_ind.get_params_and_domains()
    to_update = get_update_config(params_and_domains)

    new_config = update_config(base_ind.get_config(), to_update)
    lenia_ind = LeniaIndividual(new_config)

    return lenia_ind


def vary_ind(ind: LeniaIndividual) -> LeniaIndividual:
    params_and_domains = ind.get_params_and_domains()

    to_update: Dict = {}
    for p_and_d in params_and_domains:
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        type = p_and_d['type']
        if type == 'float':
            val = tools.mut_polynomial_bounded(ind, low=domain[0], up=domain[1], eta=ind.eta, mut_pb=ind.mut_pb)
        else:
            raise ValueError(f"type {type} unknown")

        update_dict(to_update, key_str, val)

    new_config = update_config(ind.get_config(), to_update)
    lenia_ind = LeniaIndividual(new_config)

    return lenia_ind


class LeniaEvo(Evolution):
    ind_domain: DomainLike
    sel_pb: float
    init_pb: float
    mut_pb: float
    eta: float

    def __init__(
        self,
        container: Container,
        budget: int,
        dimension: int,
        ind_domain: DomainLike = (0., 1.),
        sel_pb: float = 0.5,
        init_pb: float = 0.5,
        mut_pb: float = 0.2,
        eta: float = 20.,
        **kwargs
    ):
        self.ind_domain = ind_domain
        self.sel_pb = sel_pb
        self.init_pb = init_pb
        self.mut_pb = mut_pb
        self.eta = eta

        select_or_initialise = partial(
            tools.sel_or_init, sel_fn=tools.sel_random, sel_pb=sel_pb, init_fn=random_ind, init_pb=init_pb
        )

        super().__init__(
            container, budget, dimension=dimension, select_or_initialise=select_or_initialise, vary=vary_ind, **kwargs
        )


def eval_fn(ind: LeniaIndividual) -> LeniaIndividual:
    conf = ind.get_config()

    _, runs = search_for_init(conf, with_stats=False)

    best = runs[0]
    nb_steps = best['N']
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    # all_stats = best['all_stats']

    ind.set_init_cells(lenia_utils.compress_array(init_cells))

    ind.fitness.values = [nb_steps]

    ind.features.values = [
        conf['kernels_params']['k'][0]['m'],
        conf['kernels_params']['k'][0]['s'],
    ]
    # stats_dict = stats_list_to_dict(all_stats)
    # all_keys = list(stats_dict.keys())
    # ind.features.values = [jnp.mean(stats_dict[k]) for k in all_keys]

    print(ind.fitness.values, ind.features.values)

    return ind
