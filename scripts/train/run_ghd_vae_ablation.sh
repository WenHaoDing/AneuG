#!/usr/bin/env bash
# GHD-VAE ablation sweep (stage 1), post type2->type1 merge.
#
# Design (agreed in conversation — NOT a replication of the old
# tr_checkpoints/v2/stage1 grid, which produced no actionable conclusions):
# a "cross" sweep from the known-reasonable baseline h512/z16/kl2, one axis
# at a time (cheaper and easier to eyeball a trend from than a full grid,
# important since the only quality signal available is visual inspection of
# the generative sanity panel — no FID-equivalent metric exists yet):
#
#   - beta sweep (latent_dim=16 fixed):      kl_mult in {0.5, 1, 2, 4}
#   - capacity sweep (kl_mult=2 fixed):      latent_dim in {8, 16, 32}
#   - width x high-beta interaction:         hidden_dim=1024 at kl_mult in {2, 4}
#     (does a wider network recover reconstruction detail lost to strong KL?)
#   - annealing comparison at the baseline:  kl_anneal_epochs=500 vs. 0
#
# 9 distinct configs total. Saved under tr_checkpoints/v2_1 (not v2, so the
# old runs are left untouched) via train_ghd_vae.py's --save-root.
#
# Runs sequentially within each GPU (one tmux session per GPU, configs queued
# with &&) rather than firing all 9 at once, to avoid overloading a single
# GPU with several concurrent hidden_dim=512-1024 VAE runs. Round-robins
# configs across the given GPU list.
#
# Usage:
#   bash scripts/train/run_ghd_vae_ablation.sh [gpu0 gpu1 ...]
# Default GPUs: cuda:0 cuda:1 cuda:2 — check `nvidia-smi` first, other jobs
# may already be using them.

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -gt 0 ]; then GPUS=("$@"); else GPUS=(cuda:0 cuda:1 cuda:2); fi
SAVE_ROOT="tr_checkpoints/v2_1"

# hidden_dim latent_dim kl_mult kl_anneal_epochs
CONFIGS=(
  "512 16 0.5 0"
  "512 16 1 0"
  "512 16 2 0"
  "512 16 4 0"
  "1024 16 2 0"
  "1024 16 4 0"
  "512 8 2 0"
  "512 32 2 0"
  "512 16 2 500"
)

N_GPUS=${#GPUS[@]}
declare -a BUCKET
for g in $(seq 0 $((N_GPUS - 1))); do BUCKET[$g]=""; done
for i in "${!CONFIGS[@]}"; do
  g=$((i % N_GPUS))
  BUCKET[$g]="${BUCKET[$g]}${BUCKET[$g]:+;}${CONFIGS[$i]}"
done

CONDA_INIT='source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new'

for g in "${!GPUS[@]}"; do
  GPU="${GPUS[$g]}"
  SESSION="ghd_vae_ablation_$(echo "$GPU" | tr ':' '_')"
  CMDS="$CONDA_INIT"
  IFS=';' read -ra QUEUE <<< "${BUCKET[$g]}"
  for cfg in "${QUEUE[@]}"; do
    read -r H Z KL ANNEAL <<< "$cfg"
    RUN_LOG="/tmp/ghd_vae_ablation_h${H}_z${Z}_kl${KL}_anneal${ANNEAL}.log"
    CMDS="$CMDS && python scripts/train/train_ghd_vae.py --hidden-dim $H --latent-dim $Z --kl-mult $KL --kl-anneal-epochs $ANNEAL --save-root $SAVE_ROOT --device $GPU 2>&1 | tee $RUN_LOG"
  done
  echo "Session $SESSION on $GPU: ${#QUEUE[@]} config(s) queued"
  tmux new-session -d -s "$SESSION" "$CMDS"
done

echo "Launched ablation across ${N_GPUS} GPU session(s). Check with: tmux ls"
