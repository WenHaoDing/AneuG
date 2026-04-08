#!/bin/bash
# batch_fit.sh — Run GHD fitting for multiple cases.
#
# Type (sidewall/bifurcation) and canonical paths are auto-detected
# from each case's landmarks.npz — no manual flags needed per case.
#
# Usage:
"
   cd "/media/yaplab2/HDD Storage/almaha"
   bash Aneu_GHD/scripts/batch_fit.sh

"
#
# To override a loss weight for a specific case, add it to the
# per-case args below, e.g.: EXTRA_ARGS["hard_case"]="--lambda-centroid 0.3"

set -e  # stop on first error

PYTHON="/home/yaplab2/miniconda3/envs/new/bin/python3"
SCRIPT="Aneu_GHD/scripts/fit_case.py"
ALIGN_DIR="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Checkpoints/Alignment/ImperialNHS_batch1"
OUT_DIR="/media/yaplab2/HDD Storage/almaha/Aneu_GHD/fitting_results/ImperialNHS_batch1/test3/${case_id}"

# ── Cases to run ──────────────────────────────────────────────────────────────
CASES=(

#not rigid enough
# I4vL5p57t4_aneurysm1 #start at rigid 5 
# lFnV0JnxJv_aneurysm1 #start at rigid 5 
# NnimNakgLG_aneurysm1

# python Aneu_GHD/scripts/fit_case.py \
#     --target "Aneu_GHD/Checkpoints/Alignment/ImperialNHS_batch1/lFnV0JnxJv_aneurysm1/final_aligned.obj" \
#     --target-lm "Aneu_GHD/Checkpoints/Alignment/ImperialNHS_batch1/lFnV0JnxJv_aneurysm1" \
#     --out "Aneu_GHD/fitting_results/ImperialNHS_batch1/lFnV0JnxJv_aneurysm1_smooth2" \
#     --device cuda:0 \
#     --n-basis 256 \
#     --lambda-occupancy 2.0 \
#     --dvs-surf-d-min 0.05 \
#     --dvs-surf-d-max 0.20 \
#     --lambda-laplacian 0.05 \
#     --lambda-consistency 1.5 \
#     --lambda-rigid-end 0.10 \
#     --lambda-rigid-start 7.0 \
#     --rigid-decay-frac 0.80


#gt small
#qHBtIl5Ft4_aneurysm2 #difficult shape branch far
 uAFn03Y0Gp_aneurysm1 

# #mesh piching
 o7cb5hR1rV_aneurysm1  #@#start at rigid 5 
 rCHyvghmVr_aneurysm1 #@

# cd "/media/yaplab2/HDD Storage/almaha"
# PYTHON=/home/yaplab2/miniconda3/envs/new/bin/python3
# SCRIPT=Aneu_GHD/scripts/fit_case.py
# ALIGN=Aneu_GHD/Checkpoints/Alignment/ImperialNHS_batch1
# OUT=Aneu_GHD/fitting_results/ImperialNHS_batch1/test2

# for case in lFnV0JnxJv_aneurysm1; do
#     $PYTHON $SCRIPT --target $ALIGN/$case/final_aligned.obj --target-lm $ALIGN/$case \
#         --out $OUT/$case --uncap \
#         --lambda-rigid-start 5.0 --rigid-decay-frac 0.90 \
#         --lambda-thickness 2.0 --thickness-r 0.5 \
#         --lambda-ring-chamfer 1.0 --lambda-centroid 0.3 \
#         --device cuda:0
# done

# for case in NnimNakgLG_aneurysm1; do
#   $PYTHON $SCRIPT --target $ALIGN/$case/final_aligned.obj --target-lm $ALIGN/$case --out $OUT/$case --uncap --lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --lambda-consistency 2.0 --lambda-laplacian 3e-2 --device cuda:0
# done

# for case in I4vL5p57t4_aneurysm1; do
#   $PYTHON $SCRIPT --target $ALIGN/$case/final_aligned.obj --target-lm $ALIGN/$case --out $OUT/$case --uncap --lambda-rigid-start 5.0 --rigid-decay-frac 0.90 \
# --lambda-laplacian 0.1 --lambda-consistency 3.0 \
# --lambda-ring-chamfer 1.0 --lambda-centroid 0.3
# --device cuda:0
# done





)

# ── Per-case overrides (optional) ─────────────────────────────────────────────
declare -A EXTRA_ARGS
# EXTRA_ARGS["ANSYS_UNIGE_35_cut1"]="--device cuda:2"
# EXTRA_ARGS["C0001_cut2"]="--device cuda:2"

# Test B: slower rigid decay (stays rigid longer, folds less)
#EXTRA_ARGS["C0008_cut2"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --lambda-rigid-end 0.5"
#Test C: stronger thickness loss (directly penalises pinching)
#EXTRA_ARGS["C0008_cut2"]="--lambda-thickness 2.0 --thickness-r 0.5 --device cuda:2"
# Test E: slower rigid decay (stays rigid longer, folds less)
 #EXTRA_ARGS["ANSYS_UNIGE_35_cut1"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --lambda-rigid-end 0.05 --lambda-thickness 2.0 --thickness-r 0.5"

#  EXTRA_ARGS["I4vL5p57t4_aneurysm1"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --device cuda:0"

#  EXTRA_ARGS["lFnV0JnxJv_aneurysm1"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --device cuda:0"

#  EXTRA_ARGS["NnimNakgLG_aneurysm1"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.90 --device cuda:0"

# EXTRA_ARGS["qHBtIl5Ft4_aneurysm2"]="--rigid-decay-frac 0.90 --lambda-consistency 1.0 --lambda-laplacian 2e-2 --device cuda:2"

#   EXTRA_ARGS["uAFn03Y0Gp_aneurysm1"]=" --lambda-rigid-start 5.0 --rigid-decay-frac 0.65 --lambda-consistency 2.0 --lambda-laplacian 2e-2 --lambda-thickness 2.0 --thickness-r 0.5 --device cuda:2"

#   EXTRA_ARGS["o7cb5hR1rV_aneurysm1"]="--rigid-decay-frac 0.80 --lambda-consistency 2.0 --lambda-laplacian 2e-2 --lambda-thickness 2.0 --thickness-r 0.5 --device cuda:2"

# # # EXTRA_ARGS["rCHyvghmVr_aneurysm1"]="--rigid-decay-frac 0.90 --lambda-consistency 1.0 --lambda-laplacian 2e-2 --lambda-thickness 1.0 --device cuda:3"

#  EXTRA_ARGS["rCHyvghmVr_aneurysm1"]="--lambda-rigid-start 5.0 --rigid-decay-frac 0.65 --lambda-consistency 1.0 --lambda-laplacian 2e-2 --lambda-thickness 2.0 --device cuda:2"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "/media/yaplab2/HDD Storage/almaha"

for case_id in "${CASES[@]}"; do
    echo "========================================================"
    echo "  Case: $case_id"
    echo "========================================================"

    $PYTHON "$SCRIPT" \
        --target    "$ALIGN_DIR/${case_id}/final_aligned.obj" \
        --target-lm "$ALIGN_DIR/${case_id}" \
        --out       "$OUT_DIR/${case_id}" \
        --uncap \
        ${EXTRA_ARGS[$case_id]:-}

    echo "  Done: $case_id"
    echo
done

echo "All cases complete."
