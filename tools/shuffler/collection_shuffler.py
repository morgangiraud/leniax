###
# Goals:
# - List all files
# - Extract reserved element from those files
# - shuffle remaining files
# - set reserved elements as first elements
# - add reaminig files to the list
# - copy all 3 elements to a folder (metadata.json, 512_video.mp4, 512_gif.gif)

import os
import random
import json
import shutil

cdir = os.path.dirname(os.path.realpath(__file__))

collection_dir = os.path.join(cdir, '..', 'outputs', 'collection-01')
metadata_filename = 'metadata.json'
media_prefix = 'creature_scale4'
video_suffix = '1024_1024.mp4'
gif_suffix = '512_512.gif'

output_metadata_dir = os.path.join(cdir, 'col_shuffled')

creatures = []

reserved_list = [
    "00-genesis/0000",  # DAO
    "05-maelstrom/0084",  # Bert
    "00-genesis/0121",  # Morgan
    "01-aquarium/00086",  # Morgan
    "07-pulsium/0186",  # Morgan
    "04-ignis/0007",  # Alex
    "01-aquarium/0031",  # Alex
]

creatures_reserved = {}

for (subdir, dirs, _) in os.walk(collection_dir):
    dirs.sort()

    print(subdir)
    metadata_fullpath = os.path.join(subdir, metadata_filename)
    if not os.path.isfile(metadata_fullpath):
        continue

    with open(metadata_fullpath, 'r') as f:
        metadata = json.load(f)

    colormap_name = ""
    for attr in metadata["attributes"]:
        if attr["trait_type"] == 'Colormap':
            colormap_name = attr["value"].lower()
            break

    if colormap_name == "":
        print("No colormap?")
        continue

    video_fullpath = os.path.join(subdir, f"{media_prefix}_{colormap_name}_{video_suffix}")
    gif_fullpath = os.path.join(subdir, f"{media_prefix}_{colormap_name}_{gif_suffix}")
    if not os.path.isfile(video_fullpath) or not os.path.isfile(gif_fullpath):
        print('Missing medias')
        continue

    data = {
        'metadata': metadata_fullpath,
        'video': video_fullpath,
        'gif': gif_fullpath,
    }
    is_reserved = False
    for idx, reserved_path in enumerate(reserved_list):
        if reserved_path in subdir:
            is_reserved = True
            break

    if is_reserved:
        creatures_reserved[str(idx)] = data
    else:
        creatures.append(data)

random.shuffle(creatures)

final_set = [creatures_reserved[str(i)] for i in range(len(reserved_list))] + creatures

assert (len(list(creatures_reserved.values())) == len(reserved_list))
assert (len(final_set) == 202)

if os.path.exists(output_metadata_dir):
    shutil.rmtree(output_metadata_dir)
os.makedirs(output_metadata_dir)

all_metadata = []
for i, datum in enumerate(final_set):
    datum = final_set[i]

    dst_metadata_path = os.path.join(output_metadata_dir, f"{i}.json")
    dst_video_path = os.path.join(output_metadata_dir, f"{i}.mp4")
    dst_gif_path = os.path.join(output_metadata_dir, f"{i}.gif")

    with open(datum['metadata'], 'r') as f:
        metadata = json.load(f)
        metadata['name'] = f"Lenia #{i}"
        metadata['image'] = f"https://lenia.world/metadata/{i}.gif"
        metadata['animation_url'] = f"https://lenia.world/metadata/{i}.mp4"
        metadata['tokenID'] = f"{i}"
        for attribute in metadata['attributes']:
            if attribute['trait_type'] == 'Colormap':
                if '-' in attribute['value']:
                    attribute['value'] = ' '.join([w.capitalize() for w in attribute['value'].split('-')])
                break

        all_metadata.append(metadata)

        with open(dst_metadata_path, 'w') as f:
            json.dump(metadata, f)
    shutil.copyfile(datum['video'], dst_video_path)
    shutil.copyfile(datum['gif'], dst_gif_path)

with open(os.path.join(output_metadata_dir, "all_metadata.json"), 'w') as f:
    json.dump(all_metadata, f)
