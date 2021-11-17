import os
import json
import copy
import numpy as np
import random
import shutil

cdir = os.path.dirname(os.path.realpath(__file__))
output_metadata_dir = os.path.join(cdir, 'metadata')

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

base_gif = [
    'alizarin.gif',
    'black-white.gif',
    'carmine-blue.gif',
    'cinnamon.gif',
    'city.gif',
    'golden.gif',
    'laurel.gif',
    'msdos.gif',
    'pink-beach.gif',
    'rainbow.gif',
    'river-Leaf.gif',
    'salvia.gif',
    'summer.gif',
    'white-black.gif',
]

with open(os.path.join(cdir, 'fake_metadata.json'), 'r') as f:
    base_metadata = json.load(f)

if os.path.exists(output_metadata_dir):
    shutil.rmtree(output_metadata_dir)
os.makedirs(output_metadata_dir)

all_metadata = []
for i in range(202):
    gif_name = np.random.choice(base_gif)

    dst_metadata_path = os.path.join(output_metadata_dir, f"{i}.json")
    dst_video_path = os.path.join(output_metadata_dir, f"{i}.mp4")
    dst_gif_path = os.path.join(output_metadata_dir, f"{i}.gif")

    metadata = copy.deepcopy(base_metadata)
    metadata['name'] = f"Lenia #{i}"
    # metadata['image'] = f"https://lenia.world/metadata/{i}.gif"
    # metadata['animation_url'] = f"https://lenia.world/metadata/{i}.mp4"
    metadata['image'] = f"metadata/{i}.gif"
    metadata['animation_url'] = f"metadata/{i}.mp4"
    metadata['tokenID'] = f"{i}"
    colormap_name = gif_name.split('.')[0]
    if '-' in colormap_name:
        colormap_name = ' '.join([w.capitalize() for w in colormap_name.split('-')])
    metadata['attributes'] = [{"value": colormap_name, "trait_type": "Colormap"}]

    for k, v in attributes_names.items():
        metadata['attributes'].append({
            "value": np.random.choice(attributes_names[k], p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            "trait_type": attributes_map[k],
            "numerical_value": random.random()
        })

    all_metadata.append(metadata)
    with open(dst_metadata_path, 'w') as f:
        json.dump(metadata, f)
    shutil.copyfile(f"{cdir}/{gif_name.split('.')[0]}.mp4", dst_video_path)
    shutil.copyfile(f"{cdir}/{gif_name.split('.')[0]}.gif", dst_gif_path)

with open(os.path.join(output_metadata_dir, "all_metadata.json"), 'w') as f:
    json.dump(all_metadata, f)
