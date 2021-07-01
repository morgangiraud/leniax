import os
import pickle

from lenia import helpers as lenia_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# final_dir = os.path.join(cdir, '..', 'outputs', '2021-06-22', '15-51-01')
final_dir = os.path.join(cdir, '..', 'outputs', 'data')
final_filename = os.path.join(final_dir, 'final.p')


def run() -> None:
    with open(final_filename, "rb") as f:
        data = pickle.load(f)

    if 'container' in data:
        grid = data['container']
    elif 'algorithms' in data:
        grid = data['algorithms'][0].container
    else:
        breakpoint()
    max_val = grid[0].base_config['run_params']['max_run_iter'] // 2

    os.chdir(final_dir)
    lenia_helpers.dump_best(grid, max_val)


if __name__ == '__main__':
    run()
