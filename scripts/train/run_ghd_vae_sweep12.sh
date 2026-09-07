#!/usr/bin/env bash
# GHD-VAE sweep, 12 configs, trained on the ASSEMBLED corpus
# (runtime_dataset/AneuG_processed, 525 cases).
#
# Full 2x3x2 factorial over the three knobs asked for:
#   hidden_dim in {512, 1024}      (>= 512 per instruction)
#   latent_dim in {16, 32, 64}
#   kl_mult    in {0.5, 2}         (0.5 = the previously-used setting,
#                                   2 = train_ghd_vae.py's own default)
# A factorial rather than a one-axis-at-a-time cross, because the only quality
# signal is human inspection of the generative panel: with no metric to trend,
# an interaction (e.g. wide network rescuing a high-KL latent) is only visible
# if both levels are actually run together.
#
# Checkpoints -> runtime_train/ghd_vae/stage1/ghd_vae_h<H>_z<Z>_kl<K>/
#
# Usage: bash scripts/train/run_ghd_vae_sweep12.sh [gpu ...]
set -uo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -gt 0 ]; then GPUS=("$@"); else GPUS=(cuda:0 cuda:1 cuda:2); fi
SAVE_ROOT="runtime_train/ghd_vae"
PY=/home/yaplab2/miniconda3/envs/new/bin/python
LOGDIR="runtime_train/ghd_vae/_logs"
mkdir -p "$LOGDIR"

CONFIGS=(
  "512 16 0.5"   "512 16 2"
  "512 32 0.5"   "512 32 2"
  "512 64 0.5"   "512 64 2"
  "1024 16 0.5"  "1024 16 2"
  "1024 32 0.5"  "1024 32 2"
  "1024 64 0.5"  "1024 64 2"
)

# Round-robin configs onto GPUs; each GPU runs its share SEQUENTIALLY so two
# runs never contend for the same card.
n_gpu=${#GPUS[@]}
for g in "${!GPUS[@]}"; do
  gpu="${GPUS[$g]}"
  session="ghd_vae_sweep_$(echo "$gpu" | tr ':' '_')"
  tmux kill-session -t "$session" 2>/dev/null
  script="$LOGDIR/$session.sh"
  {
    echo "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new"
    echo "cd '$(pwd)'"
    echo "unset DISPLAY"
    for i in "${!CONFIGS[@]}"; do
      [ $((i % n_gpu)) -eq "$g" ] || continue
      read -r H Z K <<< "${CONFIGS[$i]}"
      name="h${H}_z${Z}_kl${K}"
      echo "echo '[$gpu] === TRAIN $name ==='"
      echo "$PY scripts/train/train_ghd_vae.py --hidden-dim $H --latent-dim $Z --kl-mult $K \\"
      echo "    --device $gpu --save-root '$SAVE_ROOT' && echo '[$gpu] OK $name' || echo '[$gpu] FAILED $name'"
    done
    echo "echo ALL_DONE_$session"
  } > "$script"
  tmux new-session -d -s "$session" "bash '$script'"
  tmux pipe-pane -t "$session" -o "cat >> '$LOGDIR/$session.log'"
  n=$(grep -c '=== TRAIN' "$script")
  echo "$session on $gpu: $n run(s) queued"
done
echo "Launched ${#CONFIGS[@]} run(s) across ${#GPUS[@]} GPU(s) -> $SAVE_ROOT"
