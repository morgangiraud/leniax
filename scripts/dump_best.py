import os
import pickle

from ribs.archives import ArchiveBase

from leniax import qd as leniax_qd

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# final_dir = os.path.join(cdir, '..', 'experiments', '007_extended_6')
# final_dir = os.path.join(cdir, '..', 'experiments', '014_2channels', 'run-b[1.0]')
# final_dir = os.path.join(cdir, '..', 'experiments', '014_2channels', 'run-orbium-b[1.0]')
final_dir = os.path.join(cdir, '..', 'experiments', '015_3channels', 'run-b[1.0]_from_0_and_134')
# final_dir = os.path.join(cdir, '..', 'experiments', '999_debug')


def run() -> None:
    for (subdir, dirs, _) in os.walk(final_dir):
        dirs.sort()

        # First subdir is final_dir
        final_filename = os.path.join(subdir, 'final.p')
        if not os.path.isfile(final_filename):
            continue

        with open(final_filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, ArchiveBase):
                grid = data
                try:
                    qd_config = grid.qd_config
                except Exception:
                    # backward compatibility
                    qd_config = grid.base_config
            else:
                if 'container' in data:
                    grid = data['container']
                elif 'algorithms' in data:
                    grid = data['algorithms'][0].container
                else:
                    raise ValueError('Unknown data')
                qd_config = grid[0].qd_config
        max_val = qd_config['run_params']['max_run_iter']

        os.chdir(subdir)
        leniax_qd.dump_best(grid, max_val)


if __name__ == '__main__':
    run()
