import copy
from typing import Dict, List, Tuple
import jax

from . import utils as leniax_utils


class LeniaIndividual(object):
    """A Lenia individual used by QD algorithms

    .. note::
        The philosophy of the lib is to have parameters sampled from the same domain
        And then scaled by custom functions before being used in the evaluation function
        To sum up:
            - All parameters are generated in the sampling_domain
            - the dimension parameter is the number of parameter
            - in the eval function:
                1. You scale those parameters
                2. You create the configuration from those parameters
                3. You evaluate the configuration
                4. you set fitness and features

    Attributes:
        fitness: QD fitness value
        features: List of QD behaviour values
        qd_config: QD configuration
        rng_key: JAX PRNG key
        params: A list of parameters to be updated by QD
    """

    fitness: float
    features: List[float]
    qd_config: Dict
    rng_key: jax.random.KeyArray
    params: List

    def __init__(self, config: Dict, rng_key, params: List = []):
        super().__init__()

        self.qd_config = copy.deepcopy(config)
        self.rng_key = rng_key
        self.params = params

        # We assert that the genotype only update existing values
        if 'genotype' in self.qd_config:
            genotype = self.get_genotype()
            for gene in genotype:
                leniax_utils.get_param(self.qd_config, gene['key'])

    def set_init_props(self, rng_key: jax.random.KeyArray, best_init_idxs: List[int]):
        self.qd_config['algo']['init_rng_key'] = rng_key.tolist()  # type: ignore
        self.qd_config['algo']['best_init_idxs'] = best_init_idxs

    def set_cells(self, cells: str):
        """Set the cells in the QD configuration"""
        self.qd_config['run_params']['cells'] = cells

    def set_init_cells(self, init_cells: str):
        """Set the initial cells in the QD configuration"""
        self.qd_config['run_params']['init_cells'] = init_cells

    def get_config(self) -> Dict:
        if 'genotype' in self.qd_config:
            genotype = self.get_genotype()
            # Removing too much precision ensure that the one found are quite stable
            raw_values = [round(raw_val, 8) for raw_val in self.params]
            assert len(raw_values) == len(genotype)

            to_update = get_update_config(genotype, raw_values)
            config = update_config(self.qd_config, to_update)
        else:
            config = self.qd_config

        return config

    def get_genotype(self):
        return self.qd_config['genotype']


def update_config(config, to_update):
    """Update a Leniax configuration with new values

    Args:
        config: Leniax configuration
        to_update: An update dictionnary

    Returns:
        The udpated Leniax configuration
    """
    new_config = copy.deepcopy(config)

    if 'kernels_params' in to_update:
        for i, kernel in enumerate(to_update['kernels_params']):
            new_config['kernels_params'][i].update(kernel)
    if 'world_params' in to_update:
        new_config['world_params'].update(to_update['world_params'])

    return new_config


def get_update_config(genotype: Dict, raw_values: List) -> Dict:
    """Update the QD configuration using raw_values

    Args:
        genotype: A dictionnary of genotype value to be updated
        raw_values: Raw values for the update

    Returns:
        A dictionnary mapping keys with updated values
    """
    to_update: Dict = {}
    for gene, raw_val in zip(genotype, raw_values):
        key_str = gene['key']
        domain = gene['domain']
        val_type = gene['type']
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
    """Scale linearly raw_value in domain

    Args:
        raw_value: a value in ``[0, 1]``
        domain: Domain bounding the final value

    Returns:
        The scaled value
    """
    val = domain[0] + (domain[1] - domain[0]) * raw_value

    return val


def log_scale(raw_value: float, domain: Tuple[float, float]) -> float:
    """Scale logarithmically raw_value in domain

    Args:
        raw_value: a value in ``[0, 1]``
        domain: Domain bounding the final value

    Returns:
        The scaled value
    """

    val = domain[0] * (domain[1] / domain[0])**raw_value

    return val
