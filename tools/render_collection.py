import os
import json
import copy
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List
from hydra import compose, initialize
import scipy
import multiprocessing

from leniax import runner as leniax_runner
from leniax import statistics as leniax_stat
from leniax import utils as leniax_utils
from leniax import loader as leniax_loader
from leniax import helpers as leniax_helpers
from leniax import video as leniax_video
from leniax import colormaps as leniax_colormaps

cdir = os.path.dirname(os.path.realpath(__file__))

collection_name = 'collection-test'
collection_dir_relative = os.path.join('..', 'outputs', collection_name)
collection_dir = os.path.join(cdir, collection_dir_relative)
config_filename = 'config.yaml'


def get_parameters_for_scale(scale: float, world_size: List[int], config: Dict, fft: bool = True) -> Dict:
    tmp_config = copy.deepcopy(config)
    tmp_config['render_params']['world_size'] = world_size
    tmp_config['world_params']['scale'] = scale

    world_params = tmp_config['world_params']
    update_fn_version = world_params['update_fn_version'] if 'update_fn_version' in world_params else 'v1'
    weighted_average = world_params['weighted_average'] if 'weighted_average' in world_params else True
    T = world_params['T']

    max_run_iter = config['run_params']['max_run_iter']

    use_init_cells = True
    render_params = tmp_config['render_params']
    cells, K, mapping = leniax_helpers.init(tmp_config, use_init_cells, fft)
    gf_params = mapping.get_gf_params()
    kernels_weight_per_channel = mapping.get_kernels_weight_per_channel()
    R = config['world_params']['R']
    update_fn = leniax_helpers.build_update_fn(K.shape, mapping, update_fn_version, weighted_average, fft)
    compute_stats_fn = leniax_stat.build_compute_stats_fn(tmp_config['world_params'], render_params)

    return {
        'cells0': cells,
        'K': K,
        'gf_params': gf_params,
        'kernels_weight_per_channel': kernels_weight_per_channel,
        'T': T,
        'max_run_iter': max_run_iter,
        'R': R,
        'update_fn': update_fn,
        'compute_stats_fn': compute_stats_fn
    }, tmp_config


def run_at_scale(config, cropped_cells, scale, params_scale):
    cropped_cells = scale_cells(cropped_cells, scale)

    nb_channels = config['world_params']['nb_channels']
    world_size = config['render_params']['world_size']
    zero_background = jnp.zeros([1, nb_channels] + world_size)
    cells = leniax_utils.merge_cells(zero_background, cropped_cells)
    params_scale['cells0'] = cells

    all_cells, _, _, stats_dict = leniax_runner.run_scan(**params_scale)
    all_cells = all_cells[:, 0]  # [nb_iter, C, world_dims...]
    stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}

    return all_cells, stats_dict


def scale_cells(cells: jnp.ndarray, scale: float) -> jnp.ndarray:
    if len(cells.shape) == 4:
        # We remove the batch dimension
        cells = cells[0]

    new_cells = jnp.array(
        [scipy.ndimage.zoom(cells[i], scale, order=0) for i in range(cells.shape[0])],
        dtype=jnp.float32,
    )
    new_cells = new_cells[jnp.newaxis]

    return new_cells


def dump_assets(inputData):
    scale, subdir, all_cells, render_params, colormap, dump_gif = inputData

    leniax_helpers.dump_frame(subdir, f'creature_scale{scale}', all_cells[-1], True, colormap)
    leniax_helpers.dump_frame(
        subdir, f'world_scale{scale}', leniax_utils.auto_center_cells(all_cells[-1]), False, colormap
    )
    all_outputs_fullpath = leniax_video.dump_video(subdir, all_cells, render_params, [colormap], 'creature_scale4')
    if dump_gif:
        leniax_video.dump_gif(all_outputs_fullpath[0])

    return all_outputs_fullpath


def run() -> None:
    all_mean_stats: Dict[str, List] = {}
    all_std_stats: Dict[str, List] = {}
    fft = True

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-f','--family', required=True)
    # args = parser.parse_args()

    for (subdir, dirs, _) in os.walk(collection_dir):
        dirs.sort()

        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        family_dir_name = subdir.split('/')[-2]
        family_name = family_dir_name.split('-')[-1]
        # dir_idx = subdir.split('/')[-1]

        # to_check = ['00-genesis/0000', "05-maelstrom/0084"]
        # if f"{family_dir_name}/{dir_idx}" not in to_check:
        #     continue
        # if family_dir_name != args.family:
        #     continue

        metadata_fullpath = os.path.join(subdir, 'metadata.json')
        if os.path.isfile(metadata_fullpath):
            print(f'{subdir} already rendered')
        else:
            print(f'Rendering {subdir}')
            config_path = os.path.join(collection_dir_relative, family_dir_name, subdir.split('/')[-1])
            with initialize(config_path=config_path):
                omegaConf = compose(config_name=config_filename.split('.')[0])
                config = leniax_utils.get_container(omegaConf, config_path)
                config['render_params']['pixel_size_power2'] = 0
                config['render_params']['pixel_size'] = 1
                config['render_params']['size_power2'] = 7
                config['render_params']['world_size'] = [128, 128]
                config['world_params']['scale'] = 1.
                config['run_params']['max_run_iter'] = 600

                leniax_utils.print_config(config)

                params_scale_init, tmp_config_init = get_parameters_for_scale(1., [128, 128], config, fft)
                params_scale_1, tmp_config_1 = get_parameters_for_scale(1., [128, 128], config, fft)
                params_scale_2, tmp_config_2 = get_parameters_for_scale(2., [256, 256], config, fft)
                params_scale_4, tmp_config_4 = get_parameters_for_scale(4., [512, 512], config, fft)
                params_scale_6, tmp_config_6 = get_parameters_for_scale(6., [768, 768], config, fft)
                params_scale_8, tmp_config_8 = get_parameters_for_scale(8., [1024, 1024], config, fft)

                # Here the goal is to make sure we reach the stable creature state from the initial conditions
                params_scale_init['max_run_iter'] = 600
                all_cells, _, _, stats_dict = leniax_runner.run_scan(**params_scale_init)
                all_cells = all_cells[:, 0]  # [nb_iter, C, world_dims...]
                stats_dict = {k: v.squeeze() for k, v in stats_dict.items()}
                print("init N", stats_dict['N'])
                if stats_dict['N'] < tmp_config_init['run_params']['max_run_iter'] and stats_dict['mass'][-1] == 0:
                    print('/////////////////////////////////////////////')
                    print('Current run finished with no mass, continuing')
                    print('/////////////////////////////////////////////')
                    continue
                if stats_dict['N'] < tmp_config_init['run_params']['max_run_iter'] and stats_dict['mass_volume'][-1
                                                                                                                 ] > 30:
                    print('/////////////////////////////////////////////')
                    print('Current run finished with mass explosion, continuing')
                    print('/////////////////////////////////////////////')
                    continue

                leniax_helpers.dump_viz_data(subdir, config, stats_dict)

                # colormap_names = list(leniax_colormaps.colormaps.keys())
                # colormap_names.remove('rainbow_transparent')
                # color_proba = np.array([12, 3, 12, 8, 4, 8, 8, 4, 6, 2, 12, 12, 6, 3]) / 100
                # assert (len(colormap_names) == len(color_proba))
                # colormap_idx = np.random.multinomial(1, color_proba, size=1)[0].argmax()
                # # colormap_idx = colormap_names.index('msdos')
                # colormap = leniax_colormaps.get(colormap_names[colormap_idx])
                colormap = leniax_colormaps.ExtendedColormap('extended')

                leniax_helpers.dump_frame(subdir, 'init', all_cells[-1], True, colormap)
                leniax_video.dump_video(subdir, all_cells, tmp_config_1['render_params'], [colormap], 'init')

                # I didn't find a solution to discover which frames are stable to scale
                # so I check a few frames
                for frame_idx in range(31, -1, -1):
                    print(f'testing frame_idx: {-32 + frame_idx}')

                    zip_scales = []
                    zip_cells = []
                    zip_render_params = []

                    cropped_compressible_cells_1 = leniax_loader.make_array_compressible(
                        leniax_utils.center_and_crop_cells(all_cells[-32 + frame_idx])
                    )[jnp.newaxis]

                    # It's much more stable to scale by factor less than 4, to reach higher factor one should run the loop a bit and then scale again
                    all_cells_1, stats_dict_1 = run_at_scale(tmp_config_1, cropped_compressible_cells_1, 1., params_scale_1)
                    print("N_1", stats_dict_1["N"])
                    if stats_dict_1["N"] != config['run_params']['max_run_iter']:
                        continue
                    zip_scales.append(1)
                    zip_cells.append(all_cells_1)
                    zip_render_params.append(tmp_config_1['render_params'])

                    all_cells_2, stats_dict_2 = run_at_scale(tmp_config_2, cropped_compressible_cells_1, 2., params_scale_2)
                    print("N_2", stats_dict_2["N"])
                    if stats_dict_2["N"] != config['run_params']['max_run_iter']:
                        continue
                    zip_scales.append(2)
                    zip_cells.append(all_cells_2)
                    zip_render_params.append(tmp_config_2['render_params'])

                    all_cells_4, stats_dict_4 = run_at_scale(tmp_config_4, leniax_utils.center_and_crop_cells(all_cells_2[20]), 2., params_scale_4)
                    print("N_4", stats_dict_4["N"])
                    if stats_dict_4["N"] != config['run_params']['max_run_iter']:
                        continue
                    zip_scales.append(4)
                    zip_cells.append(all_cells_4)
                    zip_render_params.append(tmp_config_4['render_params'])

                    # all_cells_6, stats_dict_6 = run_at_scale(tmp_config_6, leniax_utils.center_and_crop_cells(all_cells_2[20]), 3., params_scale_6)
                    # print("N_6", stats_dict_6["N"])
                    # if stats_dict_6["N"] != config['run_params']['max_run_iter']:
                    #     continue
                    # zip_scales.append(6)
                    # zip_cells.append(all_cells_6)
                    # zip_render_params.append(tmp_config_6['render_params'])

                    # all_cells_8, stats_dict_8 = run_at_scale(tmp_config_8, leniax_utils.center_and_crop_cells(all_cells_4[20]), 2., params_scale_8)
                    # print("N_8", stats_dict_8["N"])
                    # if stats_dict_8["N"] == config['run_params']['max_run_iter']:
                    #     continue
                    # zip_scales.append(8)
                    # zip_cells.append(all_cells_8)
                    # zip_render_params.append(tmp_config_8['render_params'])
                    break

                nb_media = len(zip_scales)
                gif_idx = 2
                assert gif_idx < nb_media
                zip_subdirs = [subdir] * nb_media
                zip_colormap = [colormap] * nb_media
                zip_dump_gif = [False] * nb_media
                zip_dump_gif[gif_idx] = True
                input_data = zip(zip_scales, zip_subdirs, zip_cells, zip_render_params, zip_colormap, zip_dump_gif)
                with multiprocessing.Pool(processes=nb_media) as pool:
                    pool_results = pool.map(dump_assets, input_data)
                all_outputs_fullpath = pool_results[gif_idx][0]

                metadata = {
                    'name': "Lenia #",
                    'description': 'A beautiful mathematical life-form',
                    'external_link': 'https://lenia.world',
                    'image': all_outputs_fullpath[0].split('/')[-1].split('.')[0] + '.gif',
                    'animation_url': all_outputs_fullpath[0].split('/')[-1],
                    'attributes': [{
                        "value": colormap.name, "trait_type": "Colormap"
                    }],
                    'config': {
                        'kernels_params': config['kernels_params'],
                        'world_params': config['world_params'],
                        'cells': leniax_loader.compress_array(cropped_compressible_cells_1[0])
                    }
                }
                with open(metadata_fullpath, 'w') as f:
                    json.dump(metadata, f)
        # Store all stats
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

        family_dir_name = subdir.split('/')[-2]
        family_name = family_dir_name.split('-')[-1]

        current_metadata['name'] = f'Lenia #{token_id}'
        current_metadata['image'] = f'lenia-{token_id}.gif'

        # We reset metadata attributes to the only attribute that should already be set: Colormap
        for attribute in current_metadata['attributes']:
            if attribute['trait_type'] == 'Colormap':
                current_metadata['attributes'] = [attribute]
                break
        current_metadata['attributes'].append({"value": family_name, "trait_type": 'Family'})

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
            'mass_mean': ['Fly', 'Feather', 'Welter', 'Cruiser', 'Heavy'],
            'mass_volume_mean': ['Demie', 'Standard', 'Magnum', 'Jeroboam', 'Balthazar'],
            'mass_speed_mean': ['Immovable', 'Unrushed', 'Swift', 'Turbo', 'Flash'],
            'growth_mean': ['Kiai', 'Kiroku', 'Kiroku', 'Kihaku',
                            'Hibiki'],  # Because there are only 4 variations, I double the second one
            'growth_volume_mean': ['Etheric', 'Mental', 'Astral', 'Celestial', 'Spiritual'],
            'mass_density_mean': ['Aluminium', 'Iron', 'Steel', 'Tungsten', 'Vibranium'],
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

        with open(metadata_fullpath, 'w') as f:
            json.dump(current_metadata, f)

        token_id += 1

    print(f'Nb creatures: {token_id}')


if __name__ == '__main__':
    run()
