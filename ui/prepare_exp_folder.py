import os
import json
from typing import List, Dict

from leniax import utils as leniax_utils

cdir = os.path.dirname(os.path.realpath(__file__))
experiments_dir = os.path.join(cdir, '..', 'experiments')


def gather_viz_data(exp_dir: str):
    exp_viz_dir = os.path.join(exp_dir, 'viz_data')
    leniax_utils.check_dir(exp_viz_dir)

    ui_viz_link = os.path.join(cdir, 'viz_data')
    if os.path.islink(ui_viz_link):
        os.remove(ui_viz_link)
    os.symlink(exp_viz_dir, ui_viz_link)

    all_viz_data: List[Dict] = []
    i = 0
    for (subdir, _, _) in os.walk(exp_dir):
        viz_data_filename = os.path.join(subdir, 'viz_data.json')
        if not os.path.isfile(viz_data_filename):
            continue

        with open(viz_data_filename, 'r') as f:
            current_viz_data = json.load(f)

        folder_link = str(i)
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
        if name == 'all_viz_data.json':
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
                os.symlink(real_creature_dir, new_category_creature_link_dst)

                creature_viz_data_filename = os.path.join(real_creature_dir, 'viz_data.json')
                with open(creature_viz_data_filename, 'r') as f:
                    creature_viz_data = json.load(f)

                creature_viz_data['relative_url'] = idx
                category_viz_data.append(creature_viz_data)

            all_category_viz_data_fullpath = os.path.join(entry_dir, 'all_viz_data.json')
            with open(all_category_viz_data_fullpath, 'w') as f:
                json.dump(category_viz_data, f)


if __name__ == "__main__":
    exp_dir = os.path.join(experiments_dir, '012_orbium_k2_beta_cube_4')

    gather_viz_data(exp_dir)
    # In between those calls, you should use viz.html to fill categories json
    # create_categories_folder(exp_dir)
