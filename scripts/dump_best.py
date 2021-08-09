import os
import pickle

from ribs.archives import ArchiveBase

from lenia import qd as lenia_qd

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# final_dir = os.path.join(cdir, '..', 'outputs', '2021-08-08', '08-51-38')
final_dir = os.path.join(cdir, '..', 'outputs', '08-44-26')
# final_dir = os.path.join(cdir, '..', 'outputs', 'good-search')
final_filename = os.path.join(final_dir, 'final.p')


def run() -> None:
    with open(final_filename, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, ArchiveBase):
            grid = data
            base_config = grid.base_config
        else:
            if 'container' in data:
                grid = data['container']
            elif 'algorithms' in data:
                grid = data['algorithms'][0].container
            else:
                breakpoint()
            base_config = grid[0].base_config
    max_val = base_config['run_params']['max_run_iter']

    os.chdir(final_dir)
    lenia_qd.dump_best(grid, max_val)


if __name__ == '__main__':
    run()
