import os
import json
from typing import List, Dict
from shutil import copyfile

from leniax import utils as leniax_utils

cdir = os.path.dirname(os.path.realpath(__file__))
experiments_dir = os.path.join(cdir, '..', 'experiments')
exp_dir = os.path.join(experiments_dir, '007_extended_6')
exp_viz_dir = os.path.join(exp_dir, 'viz_data')

collection_name = 'collection-02'


def gather_viz_data(exp_dir: str):

    leniax_utils.check_dir(exp_viz_dir)

    ui_viz_link = os.path.join(cdir, 'viz_data')
    if os.path.islink(ui_viz_link):
        os.remove(ui_viz_link)
    os.symlink(exp_viz_dir, ui_viz_link)

    all_viz_data: List[Dict] = []
    i = 0
    for (subdir, dirs, _) in os.walk(exp_dir):
        dirs.sort()

        viz_data_filename = os.path.join(subdir, 'viz_data.json')
        if not os.path.isfile(viz_data_filename):
            continue

        with open(viz_data_filename, 'r') as f:
            current_viz_data = json.load(f)

        creature_idx = subdir.split('/')[-1]
        exp_run_name = subdir.split('/')[-2]
        folder_link = f'{exp_run_name}--{creature_idx}'
        link_dst = os.path.join(exp_viz_dir, folder_link)
        if os.path.islink(link_dst):
            os.remove(link_dst)
        os.symlink(subdir, link_dst)

        current_viz_data['relative_url'] = folder_link
        all_viz_data.append(current_viz_data)

        i += 1

    with open(os.path.join(exp_viz_dir, 'all_viz_data.json'), 'w') as f:
        json.dump(all_viz_data, f)


def create_categories_folder(exp_dir: str):
    exp_viz_dir = os.path.join(exp_dir, 'viz_data')

    for entry in os.scandir(exp_viz_dir):
        name = os.path.basename(entry.path).split('.')[0]
        if name == 'all_viz_data':
            continue
        # Look for categories stored as json
        if entry.path.endswith(".json") and entry.is_file():
            with open(entry.path, 'r') as f:
                data = json.load(f)

            # Create subfolder for the category
            entry_dir = os.path.join(exp_viz_dir, name)
            leniax_utils.check_dir(entry_dir)

            category_viz_data = []
            for idx in data:
                # Get the symlink folder
                creature_symlink_dir = os.path.join(exp_viz_dir, str(idx))
                # Retrieve the real folder path
                real_creature_dir = os.path.realpath(creature_symlink_dir)

                new_category_creature_link_dst = os.path.join(entry_dir, idx)
                # Add a symlink from the category subfolder to the creature folder
                if os.path.islink(new_category_creature_link_dst):
                    os.remove(new_category_creature_link_dst)
                os.symlink(real_creature_dir, new_category_creature_link_dst)

                creature_viz_data_filename = os.path.join(real_creature_dir, 'viz_data.json')
                with open(creature_viz_data_filename, 'r') as f:
                    creature_viz_data = json.load(f)

                creature_viz_data['relative_url'] = idx
                category_viz_data.append(creature_viz_data)

            all_category_viz_data_fullpath = os.path.join(entry_dir, 'all_viz_data.json')
            with open(all_category_viz_data_fullpath, 'w') as f:
                json.dump(category_viz_data, f)


def make_collection(exp_dir, collection_name):
    exp_viz_dir = os.path.join(exp_dir, 'viz_data')
    originals_dir = os.path.join(exp_viz_dir, 'originals')
    collection_dir = os.path.join(cdir, '..', 'outputs', collection_name)

    if os.path.exists(collection_dir):
        raise Exception(f"Collection directory {collection_dir} already exist")

    for (subdir, _, _) in os.walk(originals_dir, followlinks=True):
        # First subdir is final_dir
        config_filename = os.path.join(subdir, 'config.yaml')
        if not os.path.isfile(config_filename):
            continue

        target_folder_name = subdir.split('/')[-1].zfill(5)
        target_folder_fullpath = os.path.join(collection_dir, '000-no_family_yet', target_folder_name)
        leniax_utils.check_dir(target_folder_fullpath)

        target_config_fullpath = os.path.join(target_folder_fullpath, 'config.yaml')
        copyfile(config_filename, target_config_fullpath)


if __name__ == "__main__":
    # In between those calls, you should use viz.html to fill categories json
    if not os.path.isdir(exp_viz_dir):
        gather_viz_data(exp_dir)
    else:
        create_categories_folder(exp_dir)
        make_collection(exp_dir, collection_name)
