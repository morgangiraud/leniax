import os
import pytest
from hydra import compose, initialize

from lenia.api import get_container
from lenia import core as lenia_core
from lenia import statistics as lenia_stat

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


@pytest.mark.benchmark(
    max_time=0.3,
    min_rounds=5,
    disable_gc=True,
    warmup=True  # Warm up the jit
)
def test_run(benchmark):
    with initialize(config_path='fixtures'):
        omegaConf = compose(config_name="orbium-test")
        config = get_container(omegaConf)

    cells, K, mapping = lenia_core.init(config)
    update_fn = lenia_core.build_update_fn(config['world_params'], K, mapping)
    compute_stats_fn = lenia_stat.build_compute_stats_fn(config['world_params'], config['render_params'])

    all_cells, _, _, _ = benchmark(
        lenia_core.run,
        cells,
        config['run_params']['max_run_iter'],
        update_fn,
        compute_stats_fn
    )

    assert len(all_cells) == 32
