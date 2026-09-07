#!/bin/bash
# One clone-fit arm, with Stage-A data-term ablation.
# Supersedes clone_arm.sh (a running bash script can't be edited in place,
# so new knobs go in a new file).
#
# Usage: clone_arm2.sh [rigid] [opening] [stageB_lr] [stageB_iters] \
#                      [stageA_chamfer] [stageA_occupancy] [device]
#
# Stage-A loss groups:
#   MESH HEALTH : rigid, thickness, edge, laplacian, normal-consistency
#                 -- always on, they constrain mesh quality not target fit
#   DATA        : chamfer, occupancy, node-MSE, opening-chamfer, volume
#                 -- chamfer and occupancy are switchable to 0 here
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG"
OLD="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS"
GEO="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUT="$ROOT/runtime/fitting/cloned_lambda_sweep"

LAM_RIGID="${1:-2.0}"
LAM_OPEN="${2:-1.0}"
LR_B="${3:-3e-3}"
NITER="${4:-7500}"
CHAM_A="${5:-1.0}"
OCC_A="${6:-1.0}"
DEV="${7:-cuda:0}"

TAG="lambda_${LAM_RIGID}"
[ "$LAM_OPEN" != "0" ] && [ "$LAM_OPEN" != "0.0" ] && TAG="${TAG}_open${LAM_OPEN}"
[ "$LR_B"  != "1e-3" ] && TAG="${TAG}_lr${LR_B}"
[ "$NITER" != "7500" ] && TAG="${TAG}_it${NITER}"
[ "$CHAM_A" != "1.0" ] && TAG="${TAG}_chamA${CHAM_A}"
[ "$OCC_A"  != "1.0" ] && TAG="${TAG}_occA${OCC_A}"
LOG="$OUT/${TAG}.log"

CASES=(
    0ZafRQZ51s_aneurysm1 0ZafRQZ51s_aneurysm2 184D0jB3dS_aneurysm1
    2uw5ITsPNj_aneurysm1 3AZ6NRxph7_aneurysm1 4lNuZOEFjC_aneurysm1
    4N00Dar9XN_aneurysm1 5M9lbzxsQn_aneurysm1 5ywqV955hJ_aneurysm1
    6GpXbvnIHl_aneurysm1
)

cd "$ROOT"
echo "=== rigid=$LAM_RIGID open=$LAM_OPEN lr=$LR_B iters=$NITER stageA_chamfer=$CHAM_A stageA_occ=$OCC_A on $DEV -> $TAG ===" > "$LOG"
for CASE in "${CASES[@]}"; do
    echo "--- $CASE ---" >> "$LOG"
    "$PY" "$ROOT/ghd/fitting/clone_fit.py" \
        --old-case-dir "$OLD/$CASE" \
        --case-dir     "$GEO/$CASE" \
        --out-dir      "$OUT/$TAG/ImperialNHS/$CASE" \
        --lambda-rigid-stage-a "$LAM_RIGID" \
        --lambda-opening-chamfer-stage-a "$LAM_OPEN" \
        --lambda-chamfer-stage-a "$CHAM_A" \
        --lambda-occupancy-stage-a "$OCC_A" \
        --lr "$LR_B" --n-iter "$NITER" \
        --device "$DEV" >> "$LOG" 2>&1 \
    && echo "  OK $CASE" >> "$LOG" || echo "  FAILED $CASE" >> "$LOG"
done
echo "=== $TAG done ===" >> "$LOG"
