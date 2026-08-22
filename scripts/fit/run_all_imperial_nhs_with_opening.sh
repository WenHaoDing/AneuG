#!/usr/bin/env bash
# Same as run_all_imperial_nhs.sh, but with the "with_opening" config
# (scripts/fit/fit_configs.py: opening-chamfer + roundness + normal-
# alignment + geodesic, all at weight 0.5) applied to EVERY case in
# ImperialNHS, not just a curated redo subset -- the user wants this config
# adopted for the whole dataset. Saves to a SEPARATE root
# (runtime/ImperialNHS_with_opening) so it never overwrites the existing
# default-config fits in runtime/ImperialNHS -- both remain on disk side by
# side, same convention manage.py's requeue now also follows for redos.
#
# Same fault-tolerance/resumability conventions as run_all_imperial_nhs.sh:
# one failing case does not stop the rest of its worker's queue, failures
# are appended to $SAVE_ROOT/_failed_worker_N.log, already-finished cases
# are skipped (safe to re-run).
#
# Once this finishes (or partially finishes), track it with:
#   python scripts/fit/manage.py scan --dataset ImperialNHS_with_opening \
#       --dataset-root "runtime/ImperialNHS_with_opening" \
#       --geometry-root "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
#
# Usage:
#   bash scripts/fit/run_all_imperial_nhs_with_opening.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
SAVE_ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/ImperialNHS_with_opening"
N_ITER=7500
LOG_DIR="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/33a50dbd-39a5-4965-b81b-d39aaa4dcfaf/scratchpad/pipeline_test_logs"
WITH_OPENING_FLAGS="--lambda-opening-chamfer 0.5 --lambda-roundness 0.5 --lambda-normal-alignment 0.5 --lambda-geodesic 0.5"

mkdir -p "$SAVE_ROOT" "$LOG_DIR"

# GPU 2 is already busy with the AneuX_stable batch when this was written --
# 2 light workers on GPUs 0/1 instead. Bump this up once other batches free up.
WORKER_GPUS=(cuda:0 cuda:1)
N_WORKERS=${#WORKER_GPUS[@]}

mapfile -t CASES < <(find "$GEOMETRY_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
echo "Found ${#CASES[@]} cases under $GEOMETRY_DIR"

declare -a BUCKET
for w in $(seq 0 $((N_WORKERS - 1))); do BUCKET[$w]=""; done
for i in "${!CASES[@]}"; do
  w=$((i % N_WORKERS))
  BUCKET[$w]="${BUCKET[$w]}${BUCKET[$w]:+;}${CASES[$i]}"
done

CONDA_INIT='source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new'

for w in "${!WORKER_GPUS[@]}"; do
  GPU="${WORKER_GPUS[$w]}"
  SESSION="ghd_imperial_opening_w${w}"
  tmux kill-session -t "$SESSION" 2>/dev/null
  FAILED_LOG="$SAVE_ROOT/_failed_worker_${w}.log"
  : > "$FAILED_LOG"

  IFS=';' read -ra QUEUE <<< "${BUCKET[$w]}"
  WORKER_SCRIPT="$LOG_DIR/imperial_opening_worker_${w}.sh"
  {
    echo "$CONDA_INIT"
    for CASE in "${QUEUE[@]}"; do
      CASE_DIR="$GEOMETRY_DIR/$CASE"
      OUT_DIR="$SAVE_ROOT/$CASE"
      echo "if [ -f \"$OUT_DIR/ghd_coefficients.npz\" ] && [ -f \"$OUT_DIR/ghd_fitted.obj\" ]; then"
      echo "  echo '[$GPU w$w] SKIP (already done): $CASE'"
      echo "else"
      echo "  echo '[$GPU w$w] === $CASE ==='"
      echo "  $PYTHON ghd/fitting/run_case.py --case-dir \"$CASE_DIR\" --save-root \"$SAVE_ROOT\" --device $GPU --n-iter $N_ITER --eta-min 1e-4 --lambda-occupancy 2.0 $WITH_OPENING_FLAGS"
      echo "  if [ \$? -eq 0 ]; then echo 'with_opening' > \"$OUT_DIR/config_used.txt\"; else echo \"$CASE\" >> \"$FAILED_LOG\"; echo '[$GPU w$w] FAILED: $CASE'; fi"
      echo "fi"
    done
    echo "echo ALL_DONE_$SESSION"
  } > "$WORKER_SCRIPT"

  echo "Worker $w ($SESSION) on $GPU: ${#QUEUE[@]} case(s) queued"
  tmux new-session -d -s "$SESSION" "bash '$WORKER_SCRIPT'"
  tmux pipe-pane -t "$SESSION" -o "cat >> '$LOG_DIR/imperial_opening_${SESSION}.log'"
done

echo "Launched ${N_WORKERS} workers over ${#CASES[@]} cases. Check with: tmux ls | grep ghd_imperial_opening"
echo "Failed-case logs (once workers progress): $SAVE_ROOT/_failed_worker_*.log"
