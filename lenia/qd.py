import random
import copy
from functools import partial
from typing import Dict, Any, Sequence
import jax
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
    def __init__(self, config: Dict, subkey: jnp.ndarray):
        super().__init__(name="lenia-generator")

        self.config = config
        self.subkey = subkey

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and self.config == other.config)

    def get_config(self) -> Dict:
        return self.config

    def set_init_cells(self, init_cells: jnp.ndarray):
        self.config['run_params']['cells'] = init_cells

    def get_param(self, key_string: str) -> Any:
        keys = key_string.split('.')
        dic = self.config
        for i, key in enumerate(keys):
            if key.isdigit():
                dic = dic[int(key)]
            else:
                dic = dic[key]

        return dic

    def get_params_and_domains(self):
        return self.config['params_and_domains']


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


def get_varied_config(ind: LeniaIndividual, params_and_domains: Dict, mut_pb: float, eta: float) -> Dict:
    to_update: Dict = {}
    for p_and_d in params_and_domains:
        key_str = p_and_d['key']
        domain = p_and_d['domain']
        type = p_and_d['type']
        current_val = ind.get_param(key_str)
        if type == 'float':
            val = tools.mut_polynomial_bounded([current_val], low=domain[0], up=domain[1], eta=eta, mut_pb=mut_pb)
            val = val[0]
        elif type == 'int':
            if random.random() < mut_pb:
                suggestion = random.choice([-1, 1])
                if current_val + suggestion < domain[0]:
                    val = domain[1]
                elif current_val + suggestion > domain[1]:
                    val = domain[0]
                else:
                    val = current_val + suggestion
            else:
                val = current_val
        elif type == 'choice':
            if random.random() < mut_pb:
                assert len(domain) > 1
                domain_idx = domain.index(current_val)
                val_idx = domain_idx
                while val_idx == domain_idx:
                    val = random.choice(domain)
                    val_idx = domain.index(val)
            else:
                val = current_val
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

    _, subkey = jax.random.split(base_ind.subkey)
    lenia_ind = LeniaIndividual(new_config, subkey)

    return lenia_ind


def vary_ind(ind: LeniaIndividual, mut_pb: float, eta: float) -> LeniaIndividual:
    params_and_domains = ind.get_params_and_domains()
    to_update = get_varied_config(ind, params_and_domains, mut_pb, eta)
    new_config = update_config(ind.get_config(), to_update)

    _, subkey = jax.random.split(ind.subkey)
    lenia_ind = LeniaIndividual(new_config, subkey)

    return lenia_ind


def cross_ind(inds: Sequence[LeniaIndividual]) -> Sequence[LeniaIndividual]:
    raise NotImplementedError()


class LeniaEvolution(Evolution):
    ind_domain: DomainLike
    sel_pb: float
    init_pb: float
    mut_pb: float
    eta: float

    def __init__(
        self,
        container: Container,
        budget: int,
        sel_pb: float = 0.5,
        init_pb: float = 0.5,
        mut_pb: float = 0.2,
        eta: float = 20.,
        **kwargs
    ):
        self.sel_pb = sel_pb
        self.init_pb = init_pb
        self.mut_pb = mut_pb
        self.eta = eta

        select_or_initialise = partial(
            tools.sel_or_init, sel_fn=tools.sel_random, sel_pb=sel_pb, init_fn=random_ind, init_pb=init_pb
        )
        partial_vary_fn = partial(vary_ind, mut_pb=mut_pb, eta=eta)

        super().__init__(container, budget, select_or_initialise=select_or_initialise, vary=partial_vary_fn, **kwargs)


def eval_fn(ind: LeniaIndividual) -> LeniaIndividual:
    config = ind.get_config()

    _, runs = search_for_init(ind.subkey, config, with_stats=False)

    best = runs[0]
    nb_steps = best['N']
    init_cells = best['all_cells'][0][:, 0, 0, ...]
    # all_stats = best['all_stats']

    ind.set_init_cells(lenia_utils.compress_array(init_cells))

    ind.fitness.values = [nb_steps]

    ind.features.values = [
        config['kernels_params']['k'][0]['m'],
        config['kernels_params']['k'][0]['s'],
    ]
    # stats_dict = stats_list_to_dict(all_stats)
    # all_keys = list(stats_dict.keys())
    # ind.features.values = [jnp.mean(stats_dict[k]) for k in all_keys]

    # print(ind.fitness.values, ind.features.values)

    return ind
