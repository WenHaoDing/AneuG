#!/bin/bash
# clone_fit Stage-A MESH-HEALTH ratio sweep. STAGE A ONLY.
#
# QUESTION: with chamfer and occupancy OFF, what MSE : mesh-health ratio
# actually morphs the shape without wrecking mesh quality?
#
# The node-MSE weight is held at 1.0 and the WHOLE mesh-health group (rigid,
# laplacian, consistency, edge, thickness) is scaled by one factor -- the same
# knob arap_mesh_health_sweep.sh varies, so the two are directly comparable.
#
#   ratio  scale    rigid
#   10:1    0.05     0.1
#   20:1    0.025    0.05
#   50:1    0.01     0.02
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
OLD="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/mse_weight_sweep"

RATIOS=(10 20 50); SCALES=(0.05 0.025 0.01)
CASES=(0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1)
GPUS=(cuda:0 cuda:1 cuda:2)
mkdir -p "$OUT"; cd "$ROOT"

run_gpu () {
    local gi=$1 DEV=${GPUS[$1]} LOG="$OUT/gpu${1}.log"; : > "$LOG"; local i=0
    for k in "${!SCALES[@]}"; do
        S=${SCALES[$k]}; R=${RATIOS[$k]}
        for CASE in "${CASES[@]}"; do
            if [ $((i % ${#GPUS[@]})) -eq "$gi" ]; then
                echo "--- ratio=$R (scale=$S) $CASE ---" >> "$LOG"
                "$PY" "$ROOT/ghd/fitting/clone_fit.py" \
                    --old-case-dir "$OLD/$CASE" --case-dir "$GEO/$CASE" \
                    --out-dir "$OUT/r${R}/ImperialNHS/$CASE" \
                    --mesh-health-scale-stage-a "$S" \
                    --lambda-opening-chamfer-stage-a 1.0 \
                    --lambda-chamfer-stage-a 0.0 --lambda-occupancy-stage-a 0.0 \
                    --lambda-node-mse 1.0 \
                    --skip-stage-b --device "$DEV" >> "$LOG" 2>&1 \
                && echo "  OK ratio=$R $CASE" >> "$LOG" || echo "  FAILED ratio=$R $CASE" >> "$LOG"
            fi
            i=$((i+1))
        done
    done
    echo "=== gpu$gi done ===" >> "$LOG"
}
for g in 0 1 2; do run_gpu $g & done; wait; echo "clone sweep complete"
