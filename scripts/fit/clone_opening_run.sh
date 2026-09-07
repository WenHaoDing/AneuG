#!/bin/bash
# Clone-fit arm: lambda_rigid_stage_a=1.0 PLUS a Stage-A opening chamfer.
# Usage: clone_opening_run.sh [rigid_weight] [opening_weight] [device]
# Same 10 cases as clone_lambda_sweep.sh, so it drops straight into that
# comparison as a fourth arm.
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
OLD="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/cloned_lambda_sweep"
LAM_RIGID="${1:-2.0}"    # Stage-A rigid weight
LAM_OPEN="${2:-0.5}"     # Stage-A opening-chamfer weight
DEV="${3:-cuda:1}"
TAG="lambda_${LAM_RIGID}_open${LAM_OPEN}"
LOG="$OUT/${TAG}.log"

CASES=(
    0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1
    2uw5ITsPNj_aneurysm1 3AZ6NRxph7_aneurysm1 4lNuZOEFjC_aneurysm1
    4N00Dar9XN_aneurysm1 5M9lbzxsQn_aneurysm1 5ywqV955hJ_aneurysm1
    6GpXbvnIHl_aneurysm1
)

cd "$ROOT"
echo "=== rigid=$LAM_RIGID  opening_chamfer=$LAM_OPEN  on $DEV ===" > "$LOG"
for CASE in "${CASES[@]}"; do
    echo "--- $CASE ---" >> "$LOG"
    "$PY" "$ROOT/ghd/fitting/clone_fit.py" \
        --old-case-dir "$OLD/$CASE" \
        --case-dir     "$GEO/$CASE" \
        --out-dir      "$OUT/$TAG/ImperialNHS/$CASE" \
        --lambda-rigid-stage-a "$LAM_RIGID" \
        --lambda-opening-chamfer-stage-a "$LAM_OPEN" \
        --device "$DEV" >> "$LOG" 2>&1 \
    && echo "  OK $CASE" >> "$LOG" || echo "  FAILED $CASE" >> "$LOG"
done
echo "=== $TAG done ===" >> "$LOG"
