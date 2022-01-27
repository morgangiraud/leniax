import os

from leniax import qd as leniax_qd

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

final_dir = os.path.join(cdir, '..', 'experiments', '000_debug')


def run() -> None:
    for (subdir, dirs, _) in os.walk(final_dir):
        dirs.sort()

        # First subdir is final_dir
        final_fullpath = os.path.join(subdir, 'final.p')
        if not os.path.isfile(final_fullpath):
            continue
        grid, qd_config = leniax_qd.load_qd_grid_and_config(final_fullpath)

        os.chdir(subdir)
        max_val = qd_config['run_params']['max_run_iter']
        leniax_qd.dump_best(grid, max_val)


if __name__ == '__main__':
    run()
