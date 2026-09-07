#!/usr/bin/env bash
# Parallel companion to run_ablation.sh: SAME 10-case no-opening-loss
# baseline, dome-alignment Stage 1, but with DVS occupancy loss ON
# (lambda=1.0, ghd_fit.py's own default -- passed explicitly here for
# clarity/reproducibility. Was 2.0, the old reference pipeline's weight,
# until 1.0 was adopted as the standard everywhere). Meant to run CONCURRENTLY with
# run_ablation.sh's own tmux sessions (distinct session/log names, separate
# save root) so both can be compared once done -- occupancy vs. no
# occupancy, everything else identical.
#
# Usage:
#   bash scripts/fit/run_ablation_occupancy.sh [gpu0 gpu1 ...]
# Default GPUs: cuda:0 cuda:1 cuda:2 -- these are SHARED with
# run_ablation.sh's own sessions if run concurrently (co-located on the
# same 3 GPUs), which will slow both batches down somewhat but is fine
# given the ample VRAM headroom on these 3090 Tis.

set -uo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -gt 0 ]; then GPUS=("$@"); else GPUS=(cuda:0 cuda:1 cuda:2); fi

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
SAVE_ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/ablation_test/no_opening_loss_with_occupancy"
N_ITER=7500

# Same 10 cases as run_ablation.sh
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

N_GPUS=${#GPUS[@]}
declare -a BUCKET
for g in $(seq 0 $((N_GPUS - 1))); do BUCKET[$g]=""; done
for i in "${!CASES[@]}"; do
  g=$((i % N_GPUS))
  BUCKET[$g]="${BUCKET[$g]}${BUCKET[$g]:+;}${CASES[$i]}"
done

CONDA_INIT='source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new'

for g in "${!GPUS[@]}"; do
  GPU="${GPUS[$g]}"
  SESSION="ghd_ablation_occ_$(echo "$GPU" | tr ':' '_')"
  tmux kill-session -t "$SESSION" 2>/dev/null
  CMDS="$CONDA_INIT"
  IFS=';' read -ra QUEUE <<< "${BUCKET[$g]}"
  for CASE in "${QUEUE[@]}"; do
    CMDS="$CMDS && echo '=== [$GPU] $CASE ===' && $PYTHON ghd/fitting/run_case.py --case-dir \"$GEOMETRY_DIR/$CASE\" --save-root \"$SAVE_ROOT\" --device $GPU --n-iter $N_ITER --eta-min 1e-4 --lambda-occupancy 1.0"
  done
  CMDS="$CMDS && echo ALL_DONE_$SESSION"
  echo "Session $SESSION on $GPU: ${#QUEUE[@]} job(s) queued"
  tmux new-session -d -s "$SESSION" "$CMDS"
  LOG="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/33a50dbd-39a5-4965-b81b-d39aaa4dcfaf/scratchpad/pipeline_test_logs/ablation_${SESSION}.log"
  tmux pipe-pane -t "$SESSION" -o "cat >> '$LOG'"
done

echo "Launched occupancy ablation (10 jobs) across ${N_GPUS} GPU session(s). Check with: tmux ls | grep ghd_ablation_occ"
