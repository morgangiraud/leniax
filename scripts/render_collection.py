import os
import copy
import yaml
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List
from hydra import compose, initialize

from leniax import utils as leniax_utils
from leniax import helpers as leniax_helpers
from leniax import colormaps as leniax_colormaps

cdir = os.path.dirname(os.path.realpath(__file__))

collection_dir = os.path.join(cdir, '..', 'outputs', 'collection-01')
collection_dir_relative = os.path.join('..', 'outputs', 'collection-01')
config_filename = 'config.yaml'

resolutions = [
    # [128, 128],  # Default Lenia world
    [512, 512],  # Default Lenia NFT size
]
scales = [
    # 1,
    4,
]

def run() -> None:    
    for (subdir, _, _) in os.walk(collection_dir):
        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        config_path = os.path.join(collection_dir_relative, subdir.split('/')[-1])
        with initialize(config_path=config_path):
            omegaConf = compose(config_name=config_filename.split('.')[0])
            ori_config = leniax_helpers.get_container(omegaConf)

            leniax_utils.check_dir(subdir)

            for res, scale in zip(resolutions, scales):
                config = copy.deepcopy(ori_config)
                config["render_params"]["world_size"] = res
                config["render_params"]["pixel_size"] = 1
                config['world_params']['scale'] = scale

                print('Initialiazing and running', subdir, res, scale)
                all_cells, _, _, stats_dict = leniax_helpers.init_and_run(
                    config, with_jit=True, fft=True
                )  # [nb_max_iter, N=1, C, world_dims...]
                all_cells = all_cells[:int(stats_dict['N']), 0]  # [nb_iter, C, world_dims...]

                print("Dumping assets")
                colormaps = [plt.get_cmap(name) for name in ['viridis']]#, 'plasma', 'magma', 'cividis', 'turbo', 'ocean']]
                # colormaps.append(leniax_colormaps.LeniaTemporalColormap('earth'))
                leniax_helpers.dump_assets(subdir, config, all_cells, stats_dict, colormaps)

    all_mean_stats: Dict[str, List] = {}
    all_std_stats: Dict[str, List] = {}

    for (subdir, _, _) in os.walk(collection_dir):
        viz_data_fullpath = os.path.join(subdir, 'viz_data.json')
        if not os.path.isfile(viz_data_fullpath):
            continue

        with open(viz_data_fullpath, 'r') as f:
            viz_data = json.load(f)
    
        for k, v in viz_data['stats'].items():
            if k == 'N':
                continue
            if k not in all_mean_stats:
                all_mean_stats[k] = []
                all_std_stats[k] = []

            all_mean_stats[k].append(viz_data['stats'][k])
            all_std_stats[k].append(viz_data['stats'][k])

        metadata_fullpath = os.path.join(subdir, 'metadata.json')
        metadata = {
            'name': '',
            'description': '',
            'external_link': '',
            'attributes': [
                {
                    "value": "plasma",
                    "trait_type": "Colormap"
                },
                {
                    "value": "Static",
                    "trait_type": "Colormap type"
                },
                {
                    "value": "Small",
                    "trait_type": "Volume"
                },
                {
                    "value": "average",
                    "trait_type": "Speed"
                },
                {
                    "value": "Average",
                    "trait_type": "density"
                }
            ]
        }
        with open(metadata_fullpath, 'w') as f:
            json.dump(metadata, f)


    all_keys = list(all_mean_stats.keys())
    fig, axs = plt.subplots(len(all_keys))
    fig.set_size_inches(10, 10)
    for i, k in enumerate(all_keys):
        if 'mean' not in k:
            continue
        axs[i].set_title(k.capitalize())
        axs[i].plot(all_mean_stats[k], jnp.zeros_like(jnp.array(all_mean_stats[k])), 'x')
    plt.tight_layout()
    fig.savefig(os.path.join(collection_dir, 'collection_stats.png'))
    plt.close(fig)



if __name__ == '__main__':
    run()
