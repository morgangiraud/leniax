import os
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List
from hydra import compose, initialize

from leniax import utils as leniax_utils
from leniax import helpers as leniax_helpers
# from leniax import colormaps as leniax_colormaps

cdir = os.path.dirname(os.path.realpath(__file__))

collection_dir = os.path.join(cdir, '..', 'outputs', 'collection-01')
collection_dir_relative = os.path.join('..', 'outputs', 'collection-01')
config_filename = 'config.yaml'


def run() -> None:
    all_mean_stats: Dict[str, List] = {}
    all_std_stats: Dict[str, List] = {}

    token_id = 0
    for (subdir, dirs, _) in os.walk(collection_dir):
        dirs.sort()
        if 'ko' in subdir:
            continue
        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        family_dir_name = subdir.split('/')[-2]
        family_name = family_dir_name.split('-')[-1]

        config_path = os.path.join(collection_dir_relative, family_dir_name, subdir.split('/')[-1])
        with initialize(config_path=config_path):
            omegaConf = compose(config_name=config_filename.split('.')[0])
            config = leniax_helpers.get_container(omegaConf, config_path)
            config['render_params']['pixel_size_power2'] = 0
            config['render_params']['pixel_size'] = 1
            config['render_params']['size_power2'] = 7
            config['render_params']['world_size'] = [128, 128]
            config['world_params']['scale'] = 1.
            config['run_params']['max_run_iter'] = 2048
            use_init_cells = True
            # config = leniax_utils.update_config_to_hd(config)

            leniax_utils.print_config(config)

            print('Initialiazing and running', subdir)
            all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
                config, with_jit=True, fft=True, use_init_cells=use_init_cells
            )  # [nb_max_iter, N=1, C, world_dims...]
            all_cells = all_cells[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]
            print(stats_dict['N'])
            colormaps_name_mapping = {
                'plasma': 'ms-dos',
                'turbo': 'x-ray',
                'viridis': 'phantom',
            }
            if family_name == 'firium':
                colormap = plt.get_cmap('plasma')
            elif family_name == 'phantomium':
                colormap = plt.get_cmap('viridis')
            else:
                colormap = plt.get_cmap('turbo')
            # colormaps.append(leniax_colormaps.LeniaTemporalColormap('earth'))
            leniax_helpers.dump_assets(subdir, config, all_cells, stats_dict, [colormap])

            # Prepare metadata
            viz_data_fullpath = os.path.join(subdir, 'viz_data.json')
            with open(viz_data_fullpath, 'r') as f:
                viz_data = json.load(f)
            stats = viz_data['stats']

            for k, v in stats.items():
                if 'mean' not in k:
                    continue
                if k not in all_mean_stats:
                    all_mean_stats[k] = []
                    all_std_stats[k] = []

                all_mean_stats[k].append(round(v, 5))
                all_std_stats[k].append(round(v, 5))

            metadata_fullpath = os.path.join(subdir, 'metadata.json')
            metadata = {
                'name':
                f"Lenia #{token_id}",
                'tokenID':
                token_id,
                'description':
                'Lorem ipsum dolor sit amet, \
                    consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
                'external_link':
                'https://lenia.stockmouton.com',
                'image':
                f'lenia-{token_id}.gif',
                'animation_url':
                'on_chain_url',
                'attributes': [{
                    "value": colormaps_name_mapping[colormap.name], "trait_type": "Colormap"
                }, {
                    "value": family_name, "trait_type": 'Family'
                }],
                'config': {
                    'kernels_params': config['kernels_params']['k'],
                    'world_params': config['world_params'],
                    'cells': leniax_utils.compress_array(leniax_utils.center_and_crop_cells(all_cells[-1]))
                }
            }
            with open(metadata_fullpath, 'w') as f:
                json.dump(metadata, f)

            token_id += 1

    # Compute Mean and Std of the whole collection to define the five categories
    collection_stats = {}
    all_keys = list(all_mean_stats.keys())
    fig, axs = plt.subplots(len(all_keys))
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        if 'mean' not in k:
            continue
        if k not in collection_stats:
            collection_stats[k] = {
                'percentile-0.1': float(jnp.quantile(jnp.array(all_mean_stats[k]), 0.1)),
                'percentile-0.3': float(jnp.quantile(jnp.array(all_mean_stats[k]), 0.3)),
                'percentile-0.7': float(jnp.quantile(jnp.array(all_mean_stats[k]), 0.7)),
                'percentile-0.9': float(jnp.quantile(jnp.array(all_mean_stats[k]), 0.9)),
            }
        axs[i].set_title(k.capitalize())
        axs[i].plot(all_mean_stats[k], jnp.zeros_like(jnp.array(all_mean_stats[k])), 'x')
    plt.tight_layout()
    fig.savefig(os.path.join(collection_dir, 'collection_stats.png'))
    plt.close(fig)

    # Update metadata attributes
    for (subdir, dirs, _) in os.walk(collection_dir):
        dirs.sort()
        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        metadata_fullpath = os.path.join(subdir, 'metadata.json')
        with open(metadata_fullpath, 'r') as f:
            current_metadata = json.load(f)

        viz_data_fullpath = os.path.join(subdir, 'viz_data.json')
        with open(viz_data_fullpath, 'r') as f:
            viz_data = json.load(f)
        viz_data_stats = viz_data['stats']

        all_keys = list(collection_stats.keys())
        attributes_map = {
            'mass_mean': 'Weight',
            'mass_volume_mean': 'Spread',
            'mass_speed_mean': 'Velocity',
            'growth_mean': 'Ki',
            'growth_volume_mean': 'Aura',
            'mass_density_mean': 'Robustness',
            'mass_growth_dist_mean': 'Avoidance',
        }
        attributes_names = {
            'mass_mean': ['fly', 'feather', 'welter', 'Cruiser', 'Heavy'],
            'mass_volume_mean': ['Demie', 'Standard', 'Magnum', 'Joeroboam', 'Balthazar'],
            'mass_speed_mean': ['immovable', 'unrushed', 'swift', 'turbo', 'flash'],
            'growth_mean': ['kiai', 'kiroku', 'kiroku', 'kihaku',
                            'hibiki'],  # Because there are only 4 variations, I double the second one
            'growth_volume_mean': ['etheric', 'mental', 'astral', 'celestial', 'spiritual'],
            'mass_density_mean': ['Aluminium', 'iron', 'steel', 'tungsten', 'vibranium'],
            'mass_growth_dist_mean': ['Kawarimi', 'Shunshin', 'Raiton', 'Hiraishin', 'Kamui'],
            # 'mass_angle_speed_mean': ['Very low', 'Low', 'Average', 'High', 'Very high'],
            # 'inertia_mean': ['Very low', 'Low', 'Average', 'High', 'Very high'],
            # 'growth_density_mean': ['Very low', 'Low', 'Average', 'High', 'Very high'],
        }
        for k in all_keys:
            if k not in attributes_names:
                continue
            val = viz_data_stats[k]
            col_stat = collection_stats[k]

            if val < col_stat['percentile-0.1']:
                attribute_val = attributes_names[k][0]
            elif val < col_stat['percentile-0.3']:
                attribute_val = attributes_names[k][1]
            elif val < col_stat['percentile-0.7']:
                attribute_val = attributes_names[k][2]
            elif val < col_stat['percentile-0.9']:
                attribute_val = attributes_names[k][3]
            else:
                attribute_val = attributes_names[k][4]
            current_metadata['attributes'].append({"value": attribute_val, "trait_type": attributes_map[k]})

        with open(metadata_fullpath, 'w') as f:
            json.dump(current_metadata, f)


if __name__ == '__main__':
    run()
