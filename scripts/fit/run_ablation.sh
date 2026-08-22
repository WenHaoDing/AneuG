#!/usr/bin/env bash
# Ablation: no-opening-loss baseline (opening-chamfer/roundness/normal-
# alignment/geodesic all OFF), across 10 cases (4 bifurcated, 3+3
# sidewall), 7500 Stage-2 iterations each -- 10 runs total, round-robined
# across the given GPUs.
#
# RERUN (2nd time) with the new dome-region alignment term in Stage 1
# (alignment.py's w_dome, now default-on at weight 1.0 -- confirmed good
# by eyeball before this rerun). The "with opening loss" arm from the
# first ablation run is dropped entirely per explicit instruction (already
# evaluated, not working well) -- this script now only has the one arm.
# Occupancy stays OFF too, unchanged from the first run. Volume control
# uses the cosine ramp-up scheduler (0 -> lambda_volume over
# --volume-ramp-frac, target = 100% of true volume, no undershoot), the
# ghd_fit.py default -- no extra flags needed for it here.
#
# Usage:
#   bash scripts/fit/run_ablation.sh [gpu0 gpu1 ...]
# Default GPUs: cuda:0 cuda:1 cuda:2 -- check `nvidia-smi` first, other jobs
# may already be using them.

set -uo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -gt 0 ]; then GPUS=("$@"); else GPUS=(cuda:0 cuda:1 cuda:2); fi

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
SAVE_ROOT_NO="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/ablation_test/no_opening_loss"
N_ITER=7500

# 4 bifurcated (type 0), 3 sidewall (type 1), 3 sidewall (type 2)
CASES=(
  0ZafRQZ51s_aneurysm1
  1NndJJrTjS_aneurysm1
  3AZ6NRxph7_aneurysm1
  5M9lbzxsQn_aneurysm1
  0HRrQPIFOr_aneurysm1
  0bjVkRg2pM_aneurysm1
  33aRAgScjC_aneurysm1
  1KPkS2Uocy_aneurysm1
  6jilh4kjKC_aneurysm1
  8dVIkrLOF3_aneurysm1
)

# Build the flat job list: just the "no" arm now.
JOBS=()
for CASE in "${CASES[@]}"; do
  JOBS+=("$CASE no")
done

N_GPUS=${#GPUS[@]}
declare -a BUCKET
for g in $(seq 0 $((N_GPUS - 1))); do BUCKET[$g]=""; done
for i in "${!JOBS[@]}"; do
  g=$((i % N_GPUS))
  BUCKET[$g]="${BUCKET[$g]}${BUCKET[$g]:+;}${JOBS[$i]}"
done

CONDA_INIT='source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new'

for g in "${!GPUS[@]}"; do
  GPU="${GPUS[$g]}"
  SESSION="ghd_ablation_$(echo "$GPU" | tr ':' '_')"
  tmux kill-session -t "$SESSION" 2>/dev/null
  CMDS="$CONDA_INIT"
  IFS=';' read -ra QUEUE <<< "${BUCKET[$g]}"
  for job in "${QUEUE[@]}"; do
    read -r CASE ARM <<< "$job"
    CMDS="$CMDS && echo '=== [$GPU] $CASE ===' && $PYTHON ghd/fitting/run_case.py --case-dir \"$GEOMETRY_DIR/$CASE\" --save-root \"$SAVE_ROOT_NO\" --device $GPU --n-iter $N_ITER --eta-min 1e-4"
  done
  CMDS="$CMDS && echo ALL_DONE_$SESSION"
  echo "Session $SESSION on $GPU: ${#QUEUE[@]} job(s) queued"
  tmux new-session -d -s "$SESSION" "$CMDS"
  LOG="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/33a50dbd-39a5-4965-b81b-d39aaa4dcfaf/scratchpad/pipeline_test_logs/ablation_${SESSION}.log"
  tmux pipe-pane -t "$SESSION" -o "cat >> '$LOG'"
done

echo "Launched ablation (10 jobs) across ${N_GPUS} GPU session(s). Check with: tmux ls | grep ghd_ablation"
