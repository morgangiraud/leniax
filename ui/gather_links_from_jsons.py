import os
import json

from leniax import utils as leniax_utils

cdir = os.path.dirname(os.path.realpath(__file__))
viz_dir = os.path.join(cdir, 'viz_data')


for entry in os.scandir(cdir):
    if entry.path.endswith(".json") and entry.is_file():
        name = os.path.basename(entry.path).split('.')[0]
        
        entry_dir = os.path.join(viz_dir, name)
        leniax_utils.check_dir(entry_dir)
        entry_viz_data = []
        # CHANGE THIS VALUE
        with open(entry.path, 'r') as f:
            data = json.load(f)
        
        for idx in data:
            current_entry_dir = os.path.join(viz_dir, str(idx))
            real_entry_dir = os.path.realpath(current_entry_dir)
            link_dst = os.path.join(entry_dir, idx)
            os.symlink(real_entry_dir, link_dst)

            entry_viz_data_filename = os.path.join(real_entry_dir, 'viz_data.json')
            with open(entry_viz_data_filename, 'r') as f:
                current_viz_data = json.load(f)

            current_viz_data['relative_url'] = idx
            entry_viz_data.append(current_viz_data)

        with open(os.path.join(entry_dir, 'all_viz_data.json'), 'w') as f:
            json.dump(entry_viz_data, f)