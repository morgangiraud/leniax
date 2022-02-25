###
# File taken from https://github.com/dionhaefner/pyhpc-benchmarks
###

import os
import time
import math
import json
import importlib

import numpy as np
import pandas as pd


class Timer:
    def __init__(self):
        self.elapsed = float("nan")

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if value is None:
            self.elapsed = time.perf_counter() - self._start


def estimate_repetitions(func, args=(), target_time=10, powers_of=10):
    # call function once for warm-up
    func(*args)

    # some backends need an extra nudge (looking at you, PyTorch)
    func(*args)

    # call again and measure time
    with Timer() as t:
        func(*args)

    time_per_rep = t.elapsed
    exponent = math.log(target_time / time_per_rep, powers_of)
    num_reps = int(powers_of**round(exponent))
    return max(powers_of, num_reps)


def compute_statistics(timings, burnin=1):
    stats = []

    for (task, size), t in timings.items():
        t = t[burnin:]
        repetitions = len(t)

        if repetitions:
            mean = np.mean(t)
            stdev = np.std(t)
            percentiles = np.percentile(t, [0, 25, 50, 75, 100])
        else:
            mean = stdev = float("nan")
            percentiles = [float("nan")] * 5

        stats.append((size, task, repetitions, mean, stdev, *percentiles, float("nan")))

    stats = np.array(
        stats,
        dtype=[
            ("size", "i8"),
            ("task", object),
            ("calls", "i8"),
            ("mean", "f4"),
            ("stdev", "f4"),
            ("min", "f4"),
            ("25%", "f4"),
            ("median", "f4"),
            ("75%", "f4"),
            ("max", "f4"),
            ("Δ", "f4"),
        ],
    )

    # add deltas
    sizes = np.unique(stats["size"])
    for s in sizes:
        mask = stats["size"] == s
        reference_time = np.nanmax(stats["mean"][mask])
        stats["Δ"][mask] = reference_time / stats["mean"][mask]

    stats = np.sort(stats, axis=0, order=["size", "mean", "max", "median"])
    stats_df = pd.DataFrame(stats)
    
    return stats_df


def format_output(stats_df, benchmark_title, device="cpu"):
    out = [
        "",
        benchmark_title,
        "=" * len(benchmark_title),
        f"Running on {device.upper()}",
    ]
    out.append("-" * len(out[-1]))
    out.append(stats_df.to_string())
    out.extend([
        "",
        "(time in wall seconds, less is better; delta is computed in the other direction: 1 is the slowest)",
    ])

    return "\n".join(out)


def check_consistency(res1, res2):
    if isinstance(res1, (tuple, list)):
        if not len(res1) == len(res2):
            return False

        return all(check_consistency(r1, r2) for r1, r2 in zip(res1, res2))

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    return np.allclose(res1, res2)


def get_task(task_id):
    task_module = importlib.import_module(f".{task_id}", 'tasks')

    return task_module, task_id


def update_results(results_fullpath, stats_df):
    if os.path.isfile(results_fullpath):
        results_df = pd.read_json(results_fullpath)
        results_df = results_df.set_index(['job_id', 'day', 'size', 'task']).sort_index()

        results_df = stats_df.combine_first(results_df)
    else:
        results_df = stats_df

    results_df.reset_index().to_json(results_fullpath)

    return results_df


class BackendNotSupported(Exception):
    pass


def setup_jax(device):
    if device not in ["cpu", "gpu", "tpu"]:
        raise BackendNotSupported(f"Device {device} not supported.")

    os.environ.update(
        XLA_FLAGS=(
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1 "
        ),
    )

    import jax
    from jax.config import config

    if device in ["cpu", "gpu"]:
        config.FLAGS.jax_platforms = device

    if device == "tpu":
        config.FLAGS.jax_platforms = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

    return jax
