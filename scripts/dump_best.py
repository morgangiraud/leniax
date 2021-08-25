import os
import pickle

from ribs.archives import ArchiveBase

from leniax import qd as leniax_qd
from leniax import helpers as leniax_helpers

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# final_dir = os.path.join(cdir, '..', 'outputs', '2021-08-08', '08-51-38')
# final_dir = os.path.join(cdir, '..', 'outputs', 'test')
# final_dir = os.path.join(cdir, '..', 'experiments', '003_v1_1c1k_b0.5,1_m_s')
# final_dir = os.path.join(cdir, '..', 'experiments', '005_behaviours')
# final_dir = os.path.join(cdir, '..', 'experiments', '006_v1_3c6k')
final_dir = os.path.join(cdir, '..', 'experiments', '007_beta_cube_4')


def run() -> None:
    for (subdir, _, _) in os.walk(final_dir):
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
                    breakpoint()
                qd_config = grid[0].qd_config
        max_val = qd_config['run_params']['max_run_iter']

        print(subdir)
        os.chdir(subdir)
        leniax_qd.dump_best(grid, max_val)

    leniax_helpers.gather_viz_data(final_dir)


if __name__ == '__main__':
    run()
