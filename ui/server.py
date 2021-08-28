from flask import Flask, send_from_directory
import os
import json
import time

cdir = os.path.dirname(os.path.realpath(__file__))

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

all_creatures = {
    'junks': load_creature_list(junks_fullpath),
    'potentials': load_creature_list(potentials_fullpath),
    'variations': load_creature_list(variations_fullpath),
    'originals': load_creature_list(originals_fullpath),
}
VIZ_DATA_DIR = ''
app = Flask(__name__)

@app.route('/get_creatures', methods = ['GET'])
def get_creatures():
    return all_creatures

@app.route('/add/<creature_type>/<creature_id>', methods = ['GET'])
def add_creature(creature_type, creature_id):
    print(creature_type, creature_id)
    if creature_type in all_creatures.keys():
        if creature_id not in all_creatures[creature_type]:
            all_creatures[creature_type].append(creature_id)

            with open(os.path.join(cdir, f"{creature_type}.json"), 'w') as f:
                json.dump(all_creatures[creature_type], f)

    return ('', 200)
        

@app.route('/remove/<creature_type>/<creature_id>', methods = ['GET'])
def remove_creature(creature_type, creature_id):
    if creature_type in all_creatures.keys():
        if creature_id in all_creatures[creature_type]:
            idx = all_creatures[creature_type].index(creature_id)
            del all_creatures[creature_type][idx]

            with open(os.path.join(cdir, f"{creature_type}.json"), 'w') as f:
                json.dump(all_creatures[creature_type], f)

    return ('', 200)


@app.route("/<path:path>")
def static_dir(path):
    return send_from_directory(cdir, path)

app.run()
