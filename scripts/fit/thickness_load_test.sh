#!/bin/bash
# Does the mesh_thickness_loss crash come back UNDER LOAD?
#
# SNF00000535_cut2 died in the batch with "unspecified launch failure" at
# mesh_thickness_loss.py:58, then completed cleanly on its own. The loss runs on
# fixed-topology canonical geometry with argsort-bounded indices, so a genuine
# out-of-range index should be impossible -- which points at GPU contention
# rather than arithmetic. This runs the failed cases CONCURRENTLY on one card to
# see whether the failure is reproducible only when the GPU is saturated.
#
#   bash scripts/fit/thickness_load_test.sh <n_concurrent> <gpu>
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry"
N="${1:-6}"; GPU="${2:-cuda:0}"
OUT="runtime/debug_thickness/loadtest_${N}workers"
LIST="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/f6a22b1a-2ba6-4892-882d-fd99d68dd2f0/scratchpad/failed10.txt"
mkdir -p "$OUT"

i=0
while read -r DS CASE; do
    [ -z "$CASE" ] && continue
    SRC=$([ "$DS" = "AneuX" ] && echo mesh || echo nrrd)
    ( $PY ghd/fitting/run_case.py --case-dir "$GEO/$DS/$CASE" \
          --save-root "$OUT/$DS" --dome-source "$SRC" --device "$GPU" \
          > "$OUT/${DS}__${CASE}.log" 2>&1
      if grep -q "chamfer_best" "$OUT/${DS}__${CASE}.log"; then
          echo "OK     $DS/$CASE"
      else
          echo "FAILED $DS/$CASE  :: $(grep -oE '^[A-Za-z.]*Error.*' "$OUT/${DS}__${CASE}.log" | head -1 | cut -c1-70)"
      fi ) &
    i=$((i+1))
    [ "$i" -ge "$N" ] && break
done < "$LIST"
wait
echo "=== load test done ($N concurrent on $GPU) ==="
