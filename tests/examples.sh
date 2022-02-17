#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   pip install .[all] .[examples]
#   bash tests/examples.sh
CURRENT_DIR=$(dirname "$(realpath -s "$0")")
FIXTURES_DIR=$CURRENT_DIR/fixtures
TMPDIR=$CURRENT_DIR/hydra_tmp

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

python examples/cgol.py  render_params.pixel_size=1 render_params.world_size='[64, 64]' run_params.max_run_iter=20 ++other.log_level=30
python examples/run.py -cp $FIXTURES_DIR -cn orbium-test render_params.pixel_size=1 render_params.size_power2=6 run_params.max_run_iter=20 ++other.log_level=30
python examples/search_for_init.py -cp $FIXTURES_DIR -cn orbium-test render_params.pixel_size=1 render_params.size_power2=6 run_params.max_run_iter=40 ++run_params.nb_init_search=3 ++other.log_level=30
python examples/search_for_init_mem_optimized.py -cp $FIXTURES_DIR -cn orbium-test render_params.pixel_size=1 run_params.max_run_iter=40 ++run_params.nb_init_search=3 ++other.log_level=30
python examples/search_for_mutation.py -cp $FIXTURES_DIR -cn orbium-mut render_params.pixel_size=1 run_params.max_run_iter=40 ++run_params.nb_mut_search=3 ++other.log_level=30
python examples/qd_cmame.py -cp $FIXTURES_DIR -cn qd_config-test render_params.pixel_size=1 render_params.size_power2=6 run_params.max_run_iter=20 algo.budget=16 algo.batch_size=2 ++other.log_level=30 ++other.render_bests=False
python examples/optimization.py -cp $FIXTURES_DIR -cn orbium-test render_params.pixel_size=1 run_params.max_run_iter=64 ++run_params.nb_init_search=2 ++other.log_level=30
python examples/neural_ca.py -cp $FIXTURES_DIR -cn nca-test render_params.pixel_size=1 run_params.max_run_iter=96 ++run_params.nb_init_search=1 ++run_params.nb_epochs=1 ++other.log_level=30
python examples/neural_lenia.py -cp $FIXTURES_DIR -cn nca-test render_params.pixel_size=1 run_params.max_run_iter=96 ++run_params.nb_init_search=1 ++run_params.nb_epochs=1 ++other.log_level=30

# Cleanup.
set +x 
rm -rf $TMPDIR
echo "Tests done in $SECONDS seconds"