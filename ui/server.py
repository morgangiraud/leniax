from flask import Flask, send_from_directory, Response

import os
import json
import time

cdir = os.path.dirname(os.path.realpath(__file__))

exp_dir = os.path.join(cdir, '..', 'experiments', '007_beta_cube_4')
viz_data_dir = os.path.join(exp_dir, 'viz_data', '007_beta_cube_4')

def load_creature_list(creature_list_fullpath):
    if os.path.isfile(creature_list_fullpath):
        with open(creature_list_fullpath, 'r') as f:
            creatures_l = json.load(f)
    else:
        creatures_l = []    

    return creatures_l

junks_fullpath = os.path.join(cdir, 'junks.json')
potentials_fullpath = os.path.join(cdir, 'potentials.json')
variations_fullpath = os.path.join(cdir, 'variations.json')
originals_fullpath = os.path.join(cdir, 'originals.json')
all_creature_viz_data_fullpath = os.path.join(cdir, 'viz_data', 'all_viz_data.json')

classified_creatures_ids = {
    'junks': load_creature_list(junks_fullpath),
    'potentials': load_creature_list(potentials_fullpath),
    'variations': load_creature_list(variations_fullpath),
    'originals': load_creature_list(originals_fullpath),
}

all_creature_viz_data = load_creature_list(all_creature_viz_data_fullpath)

app = Flask(__name__)

@app.route('/get_all_creatures_viz_data', methods = ['GET'])
def get_all_creatures_viz_data():
    return Response(json.dumps(all_creature_viz_data),  mimetype='application/json')

@app.route('/get_classified_creatures_ids', methods = ['GET'])
def get_classified_creatures_ids():
    return classified_creatures_ids

@app.route('/add/<creature_type>/<creature_id>', methods = ['GET'])
def add_creature(creature_type, creature_id):
    if creature_type in classified_creatures_ids.keys():
        if creature_id not in classified_creatures_ids[creature_type]:
            classified_creatures_ids[creature_type].append(creature_id)

            with open(os.path.join(cdir, f"{creature_type}.json"), 'w') as f:
                json.dump(classified_creatures_ids[creature_type], f)

    return ('', 200)
        

@app.route('/remove/<creature_type>/<creature_id>', methods = ['GET'])
def remove_creature(creature_type, creature_id):
    if creature_type in classified_creatures_ids.keys():
        if creature_id in classified_creatures_ids[creature_type]:
            idx = classified_creatures_ids[creature_type].index(creature_id)
            del classified_creatures_ids[creature_type][idx]

            with open(os.path.join(cdir, f"{creature_type}.json"), 'w') as f:
                json.dump(classified_creatures_ids[creature_type], f)

    return ('', 200)

@app.route('/metadata/<creature_id>', methods = ['GET'])
def get_metadata(creature_id):
    metadata_fullpath = os.path.join(cdir, 'viz_data', str(creature_id),  "metadata.json")
    if os.path.isfile(metadata_fullpath):
        with open(metadata_fullpath) as f:
            metadata = json.load(f)

    else:
        metadata = {}

    return metadata

@app.route("/<path:path>")
def static_dir(path):
    return send_from_directory(cdir, path)

app.run()
