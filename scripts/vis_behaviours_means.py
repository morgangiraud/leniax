import os
import pickle
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List

from ribs.archives import ArchiveBase, GridArchive

from lenia import qd as lenia_qd
from lenia import utils as lenia_utils
from lenia import helpers as lenia_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')
exp_dir = os.path.join(cdir, '..', 'experiments')
final_filename = 'final.p'


def run() -> None:
    os.chdir(exp_dir)

    all_mean_stats: Dict[str, List] = {}
    all_std_stats: Dict[str, List] = {}
    grid_shape = [10, 10]
    # grid_shape = [10, 10, 10]
    mass_density_domain = [0, 1.]
    mass_speed_domain = [0, 1.]
    # mass_volume_domain = [0, 20.]
    features_domain = [mass_density_domain, mass_speed_domain]
    # features_domain = [mass_density_domain, mass_volume_domain, mass_speed_domain]
    fitness_domain = [0, 1024]
    subdirs = [x[0] for x in os.walk(exp_dir)]
    for subdir in subdirs:
        file_path = os.path.join(subdir, final_filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                grid = pickle.load(f)

            if not isinstance(grid, ArchiveBase):
                continue
            if not hasattr(grid, '_metadata'):
                continue
        else:
            continue

        try:
            qd_config = grid.qd_config
        except Exception:
            # backward compatibility
            qd_config = grid.base_config

        seed = qd_config['run_params']['seed']
        rng_key = lenia_utils.seed_everything(seed)
        generator_builder = lenia_qd.genBaseIndividual(qd_config, rng_key)
        lenia_generator = generator_builder()

        real_bests = []
        max_val = qd_config['run_params']['max_run_iter']
        for idx in grid._occupied_indices:
            if abs(grid._objective_values[idx]) >= max_val:
                lenia = next(lenia_generator)
                lenia.qd_config = grid._metadata[idx]
                lenia[:] = grid._solutions[idx]
                real_bests.append(lenia)

        print(f"Found {len(real_bests)} beast in {file_path}")

        behaviour_archive = GridArchive(grid_shape, features_domain)
        solution_dim = 2  # genome
        behaviour_archive.initialize(solution_dim)
        behaviour_archive.qd_config = qd_config

        for lenia in real_bests:
            config = lenia.get_config()
            _, _, _, stats_dict = lenia_helpers.init_and_run(config, True)
            for k, v in stats_dict.items():
                if k == 'N':
                    continue
                if k not in all_mean_stats:
                    all_mean_stats[k] = []
                    all_std_stats[k] = []

                all_mean_stats[k].append(stats_dict[k].squeeze()[-128:].mean())
                all_std_stats[k].append(stats_dict[k].squeeze()[-128:].std())

            behaviour = [
                all_mean_stats['mass'][-1], all_mean_stats['mass_volume'][-1], all_mean_stats['mass_speed'][-1]
            ]
            behaviour_archive.add(lenia, 1024, behaviour, config)

        print(len(behaviour_archive._occupied_indices))
        lenia_qd.save_heatmap(behaviour_archive, fitness_domain, f"{subdir}/behaviour_archive_heatmap.png")
        with open(f"{subdir}/behaviour_final.p", 'wb') as handle:
            pickle.dump(behaviour_archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_keys = list(all_mean_stats.keys())
    fig, axs = plt.subplots(len(all_keys))
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        axs[i].set_title(k.capitalize() + '_mean')
        axs[i].plot(all_mean_stats[k], jnp.zeros_like(jnp.array(all_mean_stats[k])), 'x')
    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, 'all_mean_stats.png'))
    plt.close(fig)

    fig, axs = plt.subplots(len(all_keys))
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        axs[i].set_title(k.capitalize() + '_std')
        axs[i].plot(all_std_stats[k], jnp.zeros_like(jnp.array(all_mean_stats[k])), 'x')
    plt.tight_layout()
    fig.savefig(os.path.join(exp_dir, 'all_std_stats.png'))
    plt.close(fig)


if __name__ == '__main__':
    run()
