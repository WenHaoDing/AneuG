#!/bin/bash
# One clone-fit arm over the standard 10-case set.
# Usage: clone_arm_run.sh [rigid] [opening] [stageB_lr] [device]
# The output tag encodes every non-default knob so arms never collide.
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
OLD="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/cloned_lambda_sweep"

LAM_RIGID="${1:-2.0}"
LAM_OPEN="${2:-0.5}"
LR_B="${3:-1e-3}"
DEV="${4:-cuda:0}"

TAG="lambda_${LAM_RIGID}"
[ "$LAM_OPEN" != "0" ] && [ "$LAM_OPEN" != "0.0" ] && TAG="${TAG}_open${LAM_OPEN}"
[ "$LR_B" != "1e-3" ] && TAG="${TAG}_lr${LR_B}"
LOG="$OUT/${TAG}.log"

CASES=(
    0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1
    2uw5ITsPNj_aneurysm1 3AZ6NRxph7_aneurysm1 4lNuZOEFjC_aneurysm1
    4N00Dar9XN_aneurysm1 5M9lbzxsQn_aneurysm1 5ywqV955hJ_aneurysm1
    6GpXbvnIHl_aneurysm1
)

cd "$ROOT"
echo "=== rigid=$LAM_RIGID  opening=$LAM_OPEN  stageB_lr=$LR_B  on $DEV -> $TAG ===" > "$LOG"
for CASE in "${CASES[@]}"; do
    echo "--- $CASE ---" >> "$LOG"
    "$PY" "$ROOT/ghd/fitting/clone_fit.py" \
        --old-case-dir "$OLD/$CASE" \
        --case-dir     "$GEO/$CASE" \
        --out-dir      "$OUT/$TAG/ImperialNHS/$CASE" \
        --lambda-rigid-stage-a "$LAM_RIGID" \
        --lambda-opening-chamfer-stage-a "$LAM_OPEN" \
        --lr "$LR_B" \
        --device "$DEV" >> "$LOG" 2>&1 \
    && echo "  OK $CASE" >> "$LOG" || echo "  FAILED $CASE" >> "$LOG"
done
echo "=== $TAG done ===" >> "$LOG"
