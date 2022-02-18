#!/usr/bin/env python3

import os
import copy
import random
import itertools
import logging
from absl import logging as absl_logging
from omegaconf import DictConfig
import hydra
import numpy as np
import click

from backends import (__backends__ as setup_functions, BackendNotSupported)
from utilities import (Timer, estimate_repetitions, format_output, compute_statistics, check_consistency, get_task)

from leniax import utils as leniax_utils

# Disable JAX logging https://abseil.io/docs/python/guides/logging
absl_logging.set_verbosity(absl_logging.ERROR)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir)
config_name = "base"


@hydra.main(config_path=config_path, config_name=config_name)
def main(omegaConf: DictConfig) -> None:
    """Leniax benchmark

    Usage:

        $ python run.py bench.task='run' bench.device='gpu'

    Examples:

        $ taskset -c 0 python run.py bench.task='run' bench.device='gpu'

        $ python run.py benchmarks/equation_of_state bench.task='run' bench.device='gpu' run_params.nb_init_search=64 world_params.nb_channels=16
    """
    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    task = config['bench']['task']
    device = config['bench']['device']
    burnin = config['bench']['burnin']
    multipliers = config['bench']['multipliers']
    repetitions = config['bench']['repetitions']

    try:
        task_module, task_identifier = get_task(task)
    except ImportError as e:
        logging.info(f"Error while loading benchmark {task}: {e!s}", err=True)
        exit(1)

    try:
        with setup_functions['jax'](device=device) as bmod:
            logging.info(f"Using jax version {bmod.__version__}")
    except BackendNotSupported as e:
        logging.info(f'Setup for backend "jax" failed (skipping), reason: {e!s}', err=True)
        exit(1)

    runs = sorted(itertools.product(['jax'], multipliers))
    if len(runs) == 0:
        logging.info("Nothing to do")
        return

    timings = {run: [] for run in runs}

    if repetitions is None:
        logging.info("Estimating repetitions...")
        repetitions = {}

        for b, s in runs:
            # use end-to-end runtime for repetition estimation
            def run_func():
                run = task_module.make_run_fn(rng_key, copy.deepcopy(config), s)
                with setup_functions[b](device=device):
                    run()

            repetitions[(b, s)] = estimate_repetitions(run_func)
    else:
        repetitions = {(b, s): repetitions for b, s in runs}

    all_runs = list(itertools.chain.from_iterable([run] * (repetitions[run] + burnin) for run in runs))
    random.shuffle(all_runs)

    results = {}
    checked = {r: False for r in runs}

    pbar = click.progressbar(label=f"Running {len(all_runs)} benchmarks...", length=len(runs))

    try:
        with pbar:
            for size in all_runs:
                with setup_functions[b](device=device):
                    run = task_module.make_run_fn(rng_key, copy.deepcopy(config), s)
                    with Timer() as t:
                        res = run()

                # YOWO (you only warn once)
                if not checked[(b, size)]:
                    if size in results:
                        is_consistent = check_consistency(results[size], np.asarray(res))
                        if not is_consistent:
                            logging.info(
                                f"\nWarning: inconsistent results for size {size}",
                                err=True,
                            )
                    else:
                        results[size] = np.asarray(res)
                    checked[(b, size)] = True

                timings[(b, size)].append(t.elapsed)
                pbar.update(1.0 / (repetitions[(b, size)] + burnin))

            # push pbar to 100%
            pbar.update(1.0)

        for run in runs:
            assert len(timings[run]) == repetitions[run] + burnin

    finally:
        stats = compute_statistics(timings)
        logging.info(format_output(stats, task_identifier, device=device))


if __name__ == "__main__":
    main()
