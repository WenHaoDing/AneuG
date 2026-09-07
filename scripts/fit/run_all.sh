#!/bin/bash
# Full-corpus fitting for ImperialNHS + AneuX, CLONE FIRST.
#
# Every case with an old fitted mesh runs the `clone` config; the rest fall
# back to `default`. Clone jobs are queued ahead of fallback jobs on every GPU.
# Results: runtime/<config>/<dataset>/<case>/
#
#   bash scripts/fit/run_all.sh                  # launch on cuda:0,1,2
#   bash scripts/fit/run_all.sh --dry-run        # show the plan only
#   bash scripts/fit/run_all.sh --gpus cuda:0    # single GPU
#   bash scripts/fit/run_all.sh --fallback-config arap_init
set -uo pipefail
cd "$(dirname "$0")/../.."
exec /home/yaplab2/miniconda3/envs/new/bin/python scripts/fit/run_all.py \
    --datasets ImperialNHS AneuX \
    --gpus cuda:0 cuda:1 cuda:2 \
    --priority-config clone \
    --fallback-config default \
    "$@"
