#!/bin/bash
# ARAP Stage-A MESH-HEALTH ratio sweep. STAGE A ONLY.
#
# fit_with_arap_init.py's Stage A is ALREADY MSE-only (exact per-vertex
# correspondence to the ARAP-warped mesh, branch-proximity weighted) with no
# chamfer and no occupancy. Its MSE weight is implicitly 1.0, so the ratio is
# set entirely by the mesh-health weights, scaled here by one factor.
#
#   ratio  scale    rigid
#   10:1    2        0.1
#   20:1    1        0.05
#   50:1    0.4      0.02
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/arap_mesh_health_sweep"

RATIOS=(10 20 50); SCALES=(2 1 0.4)
CASES=(0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1)
GPUS=(cuda:0 cuda:1 cuda:2)
B_RIGID=0.05; B_LAPL=0.001; B_CONS=0.03; B_EDGE=0.01
mkdir -p "$OUT"; cd "$ROOT"

run_gpu () {
    local gi=$1 DEV=${GPUS[$1]} LOG="$OUT/gpu${1}.log"; : > "$LOG"; local i=0
    for k in "${!SCALES[@]}"; do
        S=${SCALES[$k]}; R=${RATIOS[$k]}
        RG=$($PY -c "print($B_RIGID*$S)"); LP=$($PY -c "print($B_LAPL*$S)")
        CS=$($PY -c "print($B_CONS*$S)"); ED=$($PY -c "print($B_EDGE*$S)")
        for CASE in "${CASES[@]}"; do
            if [ $((i % ${#GPUS[@]})) -eq "$gi" ]; then
                echo "--- ratio=$R (scale=$S rigid=$RG) $CASE ---" >> "$LOG"
                "$PY" "$ROOT/ghd/registration/fit_with_arap_init.py" \
                    --case-dir "$GEO/$CASE" \
                    --out-dir "$OUT/r${R}/ImperialNHS/$CASE" \
                    --lambda-rigid-stage-a "$RG" --lambda-laplacian-stage-a "$LP" \
                    --lambda-consistency-stage-a "$CS" --lambda-edge-stage-a "$ED" \
                    --use-ring-normal --ring-normal-blend 0.5 \
                    --skip-stage-b --device "$DEV" >> "$LOG" 2>&1 \
                && echo "  OK ratio=$R $CASE" >> "$LOG" || echo "  FAILED ratio=$R $CASE" >> "$LOG"
            fi
            i=$((i+1))
        done
    done
    echo "=== gpu$gi done ===" >> "$LOG"
}
for g in 0 1 2; do run_gpu $g & done; wait; echo "arap sweep complete"
