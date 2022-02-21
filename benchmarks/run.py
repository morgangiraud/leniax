#!/usr/bin/env python3

import os
import shutil
import copy
import random
import itertools
import logging
from omegaconf import DictConfig
import hydra
import numpy as np
import click

from utilities import (Timer, estimate_repetitions, format_output, compute_statistics, check_consistency, get_task, setup_jax)

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir)
config_name = "base"


@hydra.main(config_path=config_path, config_name=config_name)
def main(omegaConf: DictConfig) -> None:
    """Leniax benchmark

    Usage:

        $ python run.py bench.task='single_run' bench.device='gpu'

    Examples:

        $ taskset -c 0 python run.py bench.task='single_run' bench.device='gpu'

        $ python run.py bench.task='single_run' bench.device='gpu' run_params.nb_init_search=64 world_params.nb_channels=16
    """
    device = omegaConf.bench.device
    jax = setup_jax(device)
    
    from leniax import utils as leniax_utils

    config = leniax_utils.get_container(omegaConf, config_path)
    leniax_utils.set_log_level(config)

    save_dir = os.getcwd()  # Hydra change automatically the working directory for each run.
    leniax_utils.check_dir(save_dir)
    logging.info(f"Output directory: {save_dir}")

    # We seed the whole python environment.
    rng_key = leniax_utils.seed_everything(config['run_params']['seed'])

    task = config['bench']['task']
    burnin = config['bench']['burnin']
    multipliers = config['bench']['multipliers']
    repetitions = config['bench']['repetitions']

    try:
        task_module, task_identifier = get_task(task)
    except ImportError as e:
        logging.info(f"Error while loading benchmark {task}: {e!s}", err=True)
        exit(1)

    runs = sorted(itertools.product(['jax'], multipliers))

    for i, run in enumerate(runs):
        rng_key, subkey = jax.random.split(rng_key)
        run_func = task_module.make_run_fn(subkey, copy.deepcopy(config), run[1])
        runs[i] = (run[0], run[1], run_func)

    if len(runs) == 0:
        logging.info("Nothing to do")
        return

    timings = {(b, s): [] for b, s, run_func in runs}

    if repetitions is None:
        logging.info("Estimating repetitions...")
        repetitions = {}

        for b, s, run_func in runs:
            # use end-to-end runtime for repetition estimation
            repetitions[(b, s)] = estimate_repetitions(run_func)
    else:
        repetitions = {(b, s): repetitions for b, s, run_func in runs}

    all_runs = list(
        itertools.chain.from_iterable([(b, s, run_func)] * (repetitions[(b, s)] + burnin) for b, s, run_func in runs)
    )
    random.shuffle(all_runs)

    results = {}
    checked = {(b, s): False for b, s, run_func in runs}

    pbar = click.progressbar(label=f"Running {len(all_runs)} benchmarks...", length=len(runs))

    try:
        with pbar:
            for b, size, run_func in all_runs:
                with Timer() as t:
                    res = run_func()

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

        for b, s, _ in runs:
            assert len(timings[(b, s)]) == repetitions[(b, s)] + burnin

    finally:
        stats = compute_statistics(timings)
        logging.info(format_output(stats, task_identifier, device=device))

        shutil.rmtree(save_dir)


if __name__ == "__main__":
    main()
