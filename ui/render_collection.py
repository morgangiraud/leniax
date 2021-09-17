import os
import json
import copy
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List
from hydra import compose, initialize
import scipy

from leniax import core as leniax_core
from leniax import statistics as leniax_stat
from leniax import utils as leniax_utils
from leniax import helpers as leniax_helpers
from leniax import video as leniax_video
# from leniax import colormaps as leniax_colormaps

cdir = os.path.dirname(os.path.realpath(__file__))

collectin_name = 'collection-01'
collection_dir_relative = os.path.join('..', 'outputs', collectin_name)
collection_dir = os.path.join(cdir, collection_dir_relative)
config_filename = 'config.yaml'


def run() -> None:
    all_mean_stats: Dict[str, List] = {}
    all_std_stats: Dict[str, List] = {}
    fft = True

    for (subdir, dirs, _) in os.walk(collection_dir):
        dirs.sort()

        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        family_dir_name = subdir.split('/')[-2]
        family_name = family_dir_name.split('-')[-1]

        # dir_idx = subdir.split('/')[-1]
        # to_check = ['11-kaleidium/0400']
        # if f"{family_dir_name}/{dir_idx}" not in to_check:
        #     continue

        config_path = os.path.join(collection_dir_relative, family_dir_name, subdir.split('/')[-1])
        with initialize(config_path=config_path):
            omegaConf = compose(config_name=config_filename.split('.')[0])
            config = leniax_helpers.get_container(omegaConf, config_path)
            config['render_params']['pixel_size_power2'] = 0
            config['render_params']['pixel_size'] = 1
            config['render_params']['size_power2'] = 7
            config['render_params']['world_size'] = [128, 128]
            config['world_params']['scale'] = 1.
            config['run_params']['max_run_iter'] = 1024

            leniax_utils.print_config(config)

            max_run_iter = config['run_params']['max_run_iter']
            T = jnp.array(config['world_params']['T'])

            tmp_config = copy.deepcopy(config)
            world_params = tmp_config['world_params']
            update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
            weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
            use_init_cells = True
            render_params_1 = tmp_config['render_params']
            cells_1, K_1, mapping_1 = leniax_core.init(tmp_config, fft, use_init_cells)
            gfn_params_1 = mapping_1.get_gfn_params()
            kernels_weight_per_channel_1 = mapping_1.get_kernels_weight_per_channel()
            R_1 = config['world_params']['R']
            update_fn_1 = leniax_core.build_update_fn(K_1.shape, mapping_1, update_fn_version, weighted_average, fft)
            compute_stats_fn_1 = leniax_stat.build_compute_stats_fn(tmp_config['world_params'], render_params_1)

            tmp_config = copy.deepcopy(config)
            tmp_config['render_params']['world_size'] = [256, 256]
            tmp_config['world_params']['scale'] = 2.
            use_init_cells = False
            render_params_2 = tmp_config['render_params']
            _, K_2, mapping_2 = leniax_core.init(tmp_config, fft, use_init_cells)
            gfn_params_2 = mapping_2.get_gfn_params()
            kernels_weight_per_channel_2 = mapping_2.get_kernels_weight_per_channel()
            R_2 = config['world_params']['R']
            update_fn_2 = leniax_core.build_update_fn(K_2.shape, mapping_2, update_fn_version, weighted_average, fft)
            compute_stats_fn_2 = leniax_stat.build_compute_stats_fn(tmp_config['world_params'], render_params_2)

            tmp_config = copy.deepcopy(config)
            tmp_config['render_params']['world_size'] = [512, 512]
            tmp_config['world_params']['scale'] = 4.
            use_init_cells = False
            render_params_4 = tmp_config['render_params']
            _, K_4, mapping_4 = leniax_core.init(tmp_config, fft, use_init_cells)
            gfn_params_4 = mapping_4.get_gfn_params()
            kernels_weight_per_channel_4 = mapping_4.get_kernels_weight_per_channel()
            R_4 = config['world_params']['R']
            update_fn_4 = leniax_core.build_update_fn(K_4.shape, mapping_4, update_fn_version, weighted_average, fft)
            compute_stats_fn_4 = leniax_stat.build_compute_stats_fn(tmp_config['world_params'], render_params_4)

            colormaps_name_mapping = {
                'plasma': 'ms-dos',
                'turbo': 'x-ray',
                'viridis': 'phantom',
            }
            if family_name == 'firium':
                colormap = plt.get_cmap('plasma')
            else:
                colormap = plt.get_cmap('viridis')
    

            print(f'Rendering {subdir}')
            
            all_cells, _, _, stats_dict = leniax_core.run_scan(
                cells_1, K_1, gfn_params_1, kernels_weight_per_channel_1, T, max_run_iter, R_1, update_fn_1, compute_stats_fn_1
            )
            all_cells = all_cells[:, 0] # [nb_iter, C, world_dims...]
            stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}
            print(stats_dict['N'])
            leniax_helpers.dump_assets(subdir, config, all_cells, stats_dict, [colormap])

            # I didn't find a solution to discover which frames are stable to scale
            # so I check a few frames
            for frame_idx in range(31, -1, -1):
                cropped_compressible_cells_1 = leniax_utils.make_array_compressible(
                    leniax_utils.center_and_crop_cells(all_cells[-128 + frame_idx])
                )
                compressible_cells_1 = leniax_utils.merge_cells(
                    jnp.zeros_like(cells_1[0]),
                    cropped_compressible_cells_1
                )
                compressible_cells_2 = jnp.array([scipy.ndimage.zoom(compressible_cells_1[i], 2, order=2) for i in range(1)], dtype=jnp.float32)
                compressible_cells_4 = jnp.array([scipy.ndimage.zoom(compressible_cells_1[i], 4, order=2) for i in range(1)], dtype=jnp.float32)

                compressible_cells_1 = compressible_cells_1[jnp.newaxis]
                compressible_cells_2 = compressible_cells_2[jnp.newaxis]
                compressible_cells_4 = compressible_cells_4[jnp.newaxis]

                all_cells_1, _, _, stats_dict_1 = leniax_core.run_scan(
                    compressible_cells_1, K_1, gfn_params_1, kernels_weight_per_channel_1, T, max_run_iter, R_1, update_fn_1, compute_stats_fn_1
                )  # [nb_max_iter, N=1, C, world_dims...]
                all_cells_1 = all_cells_1[:, 0]  # [nb_iter, C, world_dims...]
                stats_dict_1 = {k: v.squeeze() for k, v in stats_dict_1.items()}
                N_1 = stats_dict_1['N']                

                all_cells_2, _, _, stats_dict_2 = leniax_core.run_scan(
                    compressible_cells_2, K_2, gfn_params_2, kernels_weight_per_channel_2, T, max_run_iter, R_2, update_fn_2, compute_stats_fn_2
                )  # [nb_max_iter, N=1, C, world_dims...]
                all_cells_2 = all_cells_2[:, 0]  # [nb_iter, C, world_dims...]
                stats_dict_2 = {k: v.squeeze() for k, v in stats_dict_2.items()}
                N_2 = stats_dict_2['N']
                
                all_cells_4, _, _, stats_dict_4 = leniax_core.run_scan(
                    compressible_cells_4, K_4, gfn_params_4, kernels_weight_per_channel_4, T, max_run_iter, R_4, update_fn_4, compute_stats_fn_4
                ) 
                all_cells_4 = all_cells_4[:, 0]  # [nb_iter, C, world_dims...]
                stats_dict_4 = {k: v.squeeze() for k, v in stats_dict_4.items()}
                N_4 = stats_dict_4['N']
                print(N_1, N_2, N_4)

                if N_1 + N_2 + N_4 == 3 * config['run_params']['max_run_iter']:
                    leniax_video.dump_video(subdir, all_cells_1, render_params_1, [colormap], 'creature')
                    leniax_video.dump_video(subdir, all_cells_2, render_params_2, [colormap], 'creature')
                    leniax_video.dump_video(subdir, all_cells_4, render_params_4, [colormap], 'creature')
                    break

            metadata_fullpath = os.path.join(subdir, 'metadata.json')
            metadata = {
                'name': "Lenia #",
                'description':
                'Lorem ipsum dolor sit amet, \
consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
                'external_link':
                'https://lenia.stockmouton.com',
                'image': '',
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
                    'cells': leniax_utils.compress_array(cropped_compressible_cells_1)
                }
            }
            with open(metadata_fullpath, 'w') as f:
                json.dump(metadata, f)

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
    token_id = 0
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
            current_metadata['attributes'].append({
                "value": attribute_val,
                "trait_type": attributes_map[k],
                "numerical_value": val,
            })
        current_metadata['name'] = f'Lenia #{token_id}'
        current_metadata['tokenID'] = token_id
        current_metadata['image'] = f'lenia-{token_id}.gif'

        with open(metadata_fullpath, 'w') as f:
            json.dump(current_metadata, f)

        token_id += 1

    print(f'Nb creatures: {token_id}')

if __name__ == '__main__':
    run()
