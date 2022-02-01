#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   pip install .[all] .[examples]
#   bash tests/examples.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

CURRENT_DIR=$(dirname "$(realpath -s "$0")")
FIXTURES_DIR=$CURRENT_DIR/fixtures
TMPDIR=$CURRENT_DIR/hydra_tmp

python examples/run.py -cp $FIXTURES_DIR -cn orbium-test render_params.pixel_size=1 run_params.max_run_iter=20 ++other.dry_run=True ++other.log_level=30

# Cleanup.
rm -rf $TMPDIR
echo "Success in $SECONDS seconds"