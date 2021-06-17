import os
import pickle

from lenia import helpers as lenia_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

final_dir = os.path.join(cdir, '..', 'outputs', '2021-06-17', '14-16-02')
final_filename = os.path.join(final_dir, 'final.p')


def run() -> None:
    with open(final_filename, "rb") as f:
        data = pickle.load(f)

    if 'container' in data:
        grid = data['container']
    else:
        breakpoint()
    max_val = grid[0].base_config['run_params']['max_run_iter']

    os.chdir(final_dir)
    lenia_helpers.dump_best(grid, max_val)


if __name__ == '__main__':
    run()
