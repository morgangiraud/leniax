#!/usr/bin/env python3

import os
import shutil
import copy
import random
import itertools
import logging
from omegaconf import DictConfig
import hydra
from hydra.experimental.callback import Callback
import numpy as np
import click
from utilities import (
    Timer,
    estimate_repetitions,
    format_output,
    compute_statistics,
    update_results,
    check_consistency,
    get_task,
    setup_jax
)

from leniax import utils as leniax_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir)
config_name = "base"


class RunCB(Callback):
    def on_job_end(self, config: DictConfig, job_return) -> None:
        ret = job_return.return_value
        job_id = job_return.hydra_cfg.hydra.job.config_name
        date_val = job_return.working_dir.split('/')[1]
        timings = ret['timings']
        device = ret['device']
        save_dir = ret['save_dir']

        stats_df = compute_statistics(timings)

        stats_df['job_id'] = job_id
        stats_df['day'] = date_val
        stats_df = stats_df.sort_values(['job_id', 'day', 'size', 'delta']).set_index(['job_id', 'day', 'size', 'task'])
        logging.info(format_output(stats_df, job_id, device=device))

        shutil.rmtree(save_dir)

        results_dir = os.path.join(cdir, 'results')
        leniax_utils.check_dir(results_dir)

        results_fullpath = os.path.join(results_dir, 'results.json')

        all_results_df = update_results(results_fullpath, stats_df)
        max_val = all_results_df['mean'].max() + all_results_df['stdev'].max()
        for indexes, sub_results_df in all_results_df.groupby(level=(0, 1)):
            means = sub_results_df.loc[indexes]['mean']
            stds = sub_results_df.loc[indexes]['stdev']
            ax = means.unstack().plot.bar(
                yerr=stds.unstack(),
                title=job_id,
                ylabel='Mean duration',
                ylim=[0, max_val],
            )
            fig = ax.get_figure()
            fig.savefig(os.path.join(results_dir, f'{indexes[0]}-{indexes[1]}.png'))


@hydra.main(config_path=config_path, config_name=config_name)
def bench(omegaConf: DictConfig) -> None:
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

    tasks = config['bench']['tasks']
    if type(tasks) != list:
        tasks = [tasks]
    burnin = config['bench']['burnin']
    multipliers = config['bench']['multipliers']
    repetitions = config['bench']['repetitions']

    all_tasks = {}
    for task in tasks:
        task_module, task_identifier = get_task(task)
        all_tasks[task_identifier] = {'tm': task_module}

    runs = sorted(itertools.product(all_tasks.keys(), multipliers))

    for run in runs:
        rng_key, subkey = jax.random.split(rng_key)
        current_task = all_tasks[run[0]]
        run_fn = current_task['tm'].make_run_fn(subkey, copy.deepcopy(config), run[1])
        current_task['fn'] = run_fn

    if len(runs) == 0:
        logging.info("Nothing to do")
        return

    timings = {(run_id, mul): [] for run_id, mul in runs}

    if repetitions is None:
        logging.info("Estimating repetitions...")
        repetitions = {}

        for run_id, mul in runs:
            # use end-to-end runtime for repetition estimation
            repetitions[(run_id, mul)] = estimate_repetitions(all_tasks[run_id]['fn'])
    else:
        repetitions = {(run_id, mul): repetitions for run_id, mul in runs}

    all_runs = list(
        itertools.chain.from_iterable([(run_id, mul)] * (repetitions[(run_id, mul)] + burnin) for run_id, mul in runs)
    )
    random.shuffle(all_runs)

    results = {}
    checked = {(run_id, mul): False for run_id, mul in runs}

    pbar = click.progressbar(label=f"Running {len(all_runs)} benchmarks...", length=len(runs))
    with pbar:
        for run_id, mul in all_runs:
            with Timer() as t:
                res = all_tasks[run_id]['fn']()

            # YOWO (you only warn once)
            if not checked[(run_id, mul)]:
                if (run_id, mul) in results:
                    is_consistent = check_consistency(results[(run_id, mul)], np.asarray(res))
                    if not is_consistent:
                        logging.info(
                            f"\nWarning: inconsistent results for multiplier {mul}",
                            err=True,
                        )
                else:
                    results[(run_id, mul)] = np.asarray(res)
                checked[(run_id, mul)] = True

            timings[(run_id, mul)].append(t.elapsed)
            pbar.update(1.0 / (repetitions[(run_id, mul)] + burnin))

        # push pbar to 100%
        pbar.update(1.0)

    for run_id, mul in runs:
        assert len(timings[(run_id, mul)]) == repetitions[(run_id, mul)] + burnin

    return {
        'timings': timings,
        'device': device,
        'save_dir': save_dir,
    }


if __name__ == "__main__":
    bench()
