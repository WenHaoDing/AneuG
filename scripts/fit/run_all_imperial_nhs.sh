#!/usr/bin/env bash
# Production run: the full combined alignment + GHD-fitting pipeline over
# EVERY case in the ImperialNHS geometry directory (369 cases), using the
# config confirmed good from the ablation experiments -- dome-region
# alignment (Stage 1, default-on), no opening-index-dependent losses,
# occupancy loss on (lambda=2.0), rigid weight 2.0->0.001 cosine, volume
# cosine ramp-up to 100% of true volume, edge/consistency at their default
# (no-decay) constant weights. Same as run_ablation_occupancy.sh's per-case
# command, just over the full dataset instead of 10 cases.
#
# Failures are EXPECTED (some meshes are beyond automatic salvage per
# earlier discussion) -- each case runs independently; one failing does
# NOT stop the rest of its worker's queue (uses `;` sequencing, not `&&`).
# Failed case names are appended to runtime/ImperialNHS/_failed_worker_N.log
# for later inspection; nothing here tries to fix them automatically.
#
# Already-finished cases (ghd_coefficients.npz + ghd_fitted.obj present)
# are skipped, so this script is safe to re-run to pick up where it left
# off (e.g. after adding more workers, or after a crash).
#
# Concurrency: GPUs 0 and 1 were empty when this was written (22GB+ free,
# 0% util) -- 3 concurrent workers each. GPU 2 already had another job on
# it (12GB used, 19% util) -- only 1 extra worker there, to stay light on
# it rather than piling on. 7 workers total. Memory is not the bottleneck
# (~3-6GB/job observed); this is a compute-contention tradeoff, not a
# strict VRAM one -- adjust WORKER_GPUS below if GPU availability changes.
#
# Usage:
#   bash scripts/fit/run_all_imperial_nhs.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
SAVE_ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/ImperialNHS"
N_ITER=7500
LOG_DIR="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/33a50dbd-39a5-4965-b81b-d39aaa4dcfaf/scratchpad/pipeline_test_logs"

mkdir -p "$SAVE_ROOT" "$LOG_DIR"

# 7 workers: 3 on cuda:0, 3 on cuda:1, 1 on cuda:2 (see concurrency note above).
WORKER_GPUS=(cuda:0 cuda:0 cuda:0 cuda:1 cuda:1 cuda:1 cuda:2)
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
  SESSION="ghd_imperial_w${w}"
  tmux kill-session -t "$SESSION" 2>/dev/null
  FAILED_LOG="$SAVE_ROOT/_failed_worker_${w}.log"
  : > "$FAILED_LOG"

  IFS=';' read -ra QUEUE <<< "${BUCKET[$w]}"
  WORKER_SCRIPT="$LOG_DIR/imperial_worker_${w}.sh"
  {
    echo "$CONDA_INIT"
    for CASE in "${QUEUE[@]}"; do
      CASE_DIR="$GEOMETRY_DIR/$CASE"
      OUT_DIR="$SAVE_ROOT/$CASE"
      # Skip already-finished cases (same completeness check as run_case.py's
      # own batch-mode case_is_done()) -- makes this script resumable.
      echo "if [ -f \"$OUT_DIR/ghd_coefficients.npz\" ] && [ -f \"$OUT_DIR/ghd_fitted.obj\" ]; then"
      echo "  echo '[$GPU w$w] SKIP (already done): $CASE'"
      echo "else"
      echo "  echo '[$GPU w$w] === $CASE ==='"
      echo "  $PYTHON ghd/fitting/run_case.py --case-dir \"$CASE_DIR\" --save-root \"$SAVE_ROOT\" --device $GPU --n-iter $N_ITER --eta-min 1e-4 --lambda-occupancy 2.0"
      echo "  if [ \$? -ne 0 ]; then echo \"$CASE\" >> \"$FAILED_LOG\"; echo '[$GPU w$w] FAILED: $CASE'; fi"
      echo "fi"
    done
    echo "echo ALL_DONE_$SESSION"
  } > "$WORKER_SCRIPT"

  echo "Worker $w ($SESSION) on $GPU: ${#QUEUE[@]} case(s) queued"
  tmux new-session -d -s "$SESSION" "bash '$WORKER_SCRIPT'"
  tmux pipe-pane -t "$SESSION" -o "cat >> '$LOG_DIR/imperial_${SESSION}.log'"
done

echo "Launched ${N_WORKERS} workers over ${#CASES[@]} cases. Check with: tmux ls | grep ghd_imperial"
echo "Failed-case logs (once workers progress): $SAVE_ROOT/_failed_worker_*.log"
