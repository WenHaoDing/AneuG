#!/bin/bash
# Stage-A rigid-weight sweep for clone-fitting, POST frame-inheritance fix.
# One lambda per GPU, 10 cases each, run sequentially within a GPU.
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
OLD="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/cloned_lambda_sweep"

CASES=(
    0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1
    2uw5ITsPNj_aneurysm1 3AZ6NRxph7_aneurysm1 4lNuZOEFjC_aneurysm1
    4N00Dar9XN_aneurysm1 5M9lbzxsQn_aneurysm1 5ywqV955hJ_aneurysm1
    6GpXbvnIHl_aneurysm1
)

run_lambda () {
    local LAM=$1 DEV=$2
    local LOG="$OUT/lambda_${LAM}.log"
    echo "=== lambda_rigid_stage_a=$LAM on $DEV ===" > "$LOG"
    for CASE in "${CASES[@]}"; do
        echo "--- $CASE ---" >> "$LOG"
        "$PY" "$ROOT/ghd/fitting/clone_fit.py" \
            --old-case-dir "$OLD/$CASE" \
            --case-dir     "$GEO/$CASE" \
            --out-dir      "$OUT/lambda_${LAM}/ImperialNHS/$CASE" \
            --lambda-rigid-stage-a "$LAM" \
            --device "$DEV" >> "$LOG" 2>&1 \
        && echo "  OK $CASE" >> "$LOG" || echo "  FAILED $CASE" >> "$LOG"
    done
    echo "=== lambda_$LAM done ===" >> "$LOG"
}

cd "$ROOT"
run_lambda 1.0 cuda:0 &
run_lambda 1.5 cuda:1 &
run_lambda 2.0 cuda:2 &
wait
echo "sweep complete"
