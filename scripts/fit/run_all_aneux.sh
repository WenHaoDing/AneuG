#!/usr/bin/env bash
# Production run: the full combined alignment + GHD-fitting pipeline over
# EVERY case in the AneuX_stable geometry directory (323 real cases -- a
# stray .vscode/ dir is excluded), using the same confirmed-good config as
# run_all_imperial_nhs.sh (dome-region alignment on, occupancy loss on at
# 2.0, no opening-index-dependent losses), plus this dataset's own
# --dome-source mesh (AneuX_stable has no label.nrrd -- dome location comes
# from each case's own dome_sac.ply instead, see alignment.py's
# load_dome_points_from_mesh).
#
# No separate repair pass needed first: ~143 of these cases (all type-1/2
# sidewall cases) have a fallback (short) clip mesh but were never given a
# matching fallback centerline by AneuX_stable's older preprocessing run.
# load_case_data() auto-detects and repairs this the first time each such
# case is touched (ensure_fallback_centerline -> a one-off unattended
# subprocess into the vmtk_autogen env, writing clipped_centerline_fallback
# .npy + merged_centerline_fallback.npy permanently into that case's folder)
# before alignment starts -- transparent from this script's point of view,
# just slower on a gap case's first run.
#
# Same fault-tolerance conventions as run_all_imperial_nhs.sh: one failing
# case does NOT stop the rest of its worker's queue (`;`-sequenced, not
# `&&`), failures are appended to $SAVE_ROOT/_failed_worker_N.log, already-
# finished cases (ghd_coefficients.npz + ghd_fitted.obj present) are
# skipped, so this script is safe to re-run to pick up where it left off.
#
# Concurrency: check `nvidia-smi` / `tmux ls | grep ghd_imperial` before
# launching -- the ImperialNHS batch may still be running on the same 3
# GPUs. WORKER_GPUS below defaults to one worker per GPU (light footprint);
# bump it up (repeat a GPU entry) once ImperialNHS finishes.
#
# Usage:
#   bash scripts/fit/run_all_aneux.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
GEOMETRY_DIR="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable"
SAVE_ROOT="/media/yaplab2/HDD Storage/wenhao/AneuG/runtime/AneuX_stable"
N_ITER=7500
LOG_DIR="/tmp/claude-1000/-media-yaplab2-HDD-Storage-wenhao-AneuG/33a50dbd-39a5-4965-b81b-d39aaa4dcfaf/scratchpad/pipeline_test_logs"

mkdir -p "$SAVE_ROOT" "$LOG_DIR"

# GPU 2 was idle (0% util) when this was launched, GPUs 0/1 already busy
# with the ImperialNHS batch -- 3 workers on GPU 2, 1 light extra on GPU 0.
# Add more workers (repeat a GPU entry) once ImperialNHS finishes.
WORKER_GPUS=(cuda:2 cuda:2 cuda:2 cuda:0)
N_WORKERS=${#WORKER_GPUS[@]}

mapfile -t CASES < <(find "$GEOMETRY_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -v '^\.vscode$' | sort)
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
  SESSION="ghd_aneux_w${w}"
  tmux kill-session -t "$SESSION" 2>/dev/null
  FAILED_LOG="$SAVE_ROOT/_failed_worker_${w}.log"
  : > "$FAILED_LOG"

  IFS=';' read -ra QUEUE <<< "${BUCKET[$w]}"
  WORKER_SCRIPT="$LOG_DIR/aneux_worker_${w}.sh"
  {
    echo "$CONDA_INIT"
    for CASE in "${QUEUE[@]}"; do
      CASE_DIR="$GEOMETRY_DIR/$CASE"
      OUT_DIR="$SAVE_ROOT/$CASE"
      echo "if [ -f \"$OUT_DIR/ghd_coefficients.npz\" ] && [ -f \"$OUT_DIR/ghd_fitted.obj\" ]; then"
      echo "  echo '[$GPU w$w] SKIP (already done): $CASE'"
      echo "else"
      echo "  echo '[$GPU w$w] === $CASE ==='"
      echo "  $PYTHON ghd/fitting/run_case.py --case-dir \"$CASE_DIR\" --save-root \"$SAVE_ROOT\" --device $GPU --n-iter $N_ITER --eta-min 1e-4 --lambda-occupancy 2.0 --dome-source mesh"
      echo "  if [ \$? -ne 0 ]; then echo \"$CASE\" >> \"$FAILED_LOG\"; echo '[$GPU w$w] FAILED: $CASE'; fi"
      echo "fi"
    done
    echo "echo ALL_DONE_$SESSION"
  } > "$WORKER_SCRIPT"

  echo "Worker $w ($SESSION) on $GPU: ${#QUEUE[@]} case(s) queued"
  tmux new-session -d -s "$SESSION" "bash '$WORKER_SCRIPT'"
  tmux pipe-pane -t "$SESSION" -o "cat >> '$LOG_DIR/aneux_${SESSION}.log'"
done

echo "Launched ${N_WORKERS} workers over ${#CASES[@]} cases. Check with: tmux ls | grep ghd_aneux"
echo "Failed-case logs (once workers progress): $SAVE_ROOT/_failed_worker_*.log"
