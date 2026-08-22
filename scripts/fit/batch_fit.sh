#!/bin/bash
# batch_fit.sh -- runs the combined alignment + GHD-fitting pipeline
# (ghd/fitting/run_case.py) over a fixed list of cases, one at a time.
# Same style as the old almaha/Aneu_GHD/scripts/batch_fit_gpu2_V2.sh, but
# run_case.py does BOTH stages (alignment then fitting) per case in one call,
# so there's no separate --target/--target-lm wiring here -- just --case-dir.

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
SCRIPT="ghd/fitting/run_case.py"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
SAVE_ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/ghd_fit_test"
DEVICE="cuda:0"

cd "/media/yaplab2/HDD Storage/wenhao/AneuG"

CASES=(
    0ZafRQZ51s_aneurysm1
    0HRrQPIFOr_aneurysm1
    1KPkS2Uocy_aneurysm1
)

for CASE in "${CASES[@]}"; do
    echo "========================================================"
    echo "  [$DEVICE] $CASE"
    echo "========================================================"
    $PYTHON "$SCRIPT" \
        --case-dir  "$GEOMETRY_DIR/${CASE}" \
        --save-root "$SAVE_ROOT" \
        --device    "$DEVICE" \
        --n-iter    8000 \
        --eta-min   1e-4

    if [ $? -eq 0 ]; then
        echo "  Done: $CASE"
    else
        echo "  FAILED: $CASE"
    fi
done

echo "All cases complete."

# cd "/media/yaplab2/HDD Storage/wenhao/AneuG" && bash scripts/fit/batch_fit.sh
