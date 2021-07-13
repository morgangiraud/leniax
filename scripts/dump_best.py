import os
import pickle

from ribs.archives import ArchiveBase

from lenia import helpers as lenia_helpers
from lenia import qd as lenia_qd
from lenia import utils as lenia_utils

cdir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cdir, '..', 'conf')

# final_dir = os.path.join(cdir, '..', 'outputs', '2021-07-05', '17-12-21')
final_dir = os.path.join(cdir, '..', 'outputs', '2021-07-09', '10-52-51')
final_filename = os.path.join(final_dir, 'final.p')


def run() -> None:
    with open(final_filename, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, ArchiveBase):
            grid = data
            base_config = grid.base_config
            seed = base_config['run_params']['seed']
            rng_key = lenia_utils.seed_everything(seed)
            generator_builder = lenia_qd.genBaseIndividual(base_config, rng_key)
            lenia_generator = generator_builder()
        else:
            if 'container' in data:
                grid = data['container']
            elif 'algorithms' in data:
                grid = data['algorithms'][0].container
            else:
                breakpoint()
            base_config = grid[0].base_config
            lenia_generator = None
    max_val = base_config['run_params']['max_run_iter'] // 2

    os.chdir(final_dir)
    lenia_helpers.dump_best(grid, max_val, lenia_generator)


if __name__ == '__main__':
    run()
