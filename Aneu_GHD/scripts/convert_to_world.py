#!/usr/bin/env python3
"""
convert_to_world.py — Convert already-fitted GHD meshes to original patient coordinates.

Reads (per case):
  fitting_results/.../ghd_fitted.obj          (normalised space)
  fitting_results/.../ghd_fitted_uncapped.obj (normalised, optional)
  fitting_results/.../metrics.json            → s_can
  Checkpoints/Alignment/.../landmarks.npz    → neck_pt_world, R_frame

Writes:
  fitting_results/.../ghd_fitted_world.obj
  fitting_results/.../ghd_fitted_uncapped_world.obj  (if uncapped exists)

Inverse transform:
  verts_world = (verts_norm * s_can) @ R_frame + neck_pt_world

Usage:
  cd "/media/yaplab2/HDD Storage/almaha"
  python3 Aneu_GHD/scripts/convert_to_world.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Aneu_GHD import load_obj, save_obj

ALIGN_DIR   = "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Checkpoints/Alignment/AnueX"
RESULTS_DIR = "/media/yaplab2/HDD Storage/almaha/AneuG/Aneu_GHD/fitting_results/AnueX"

CASES = [
    "ANSYS_UNIGE_09_cut1",
    "ANSYS_UNIGE_35_cut1",
    "C0001_cut2",
    "C0002_cut2",
    "C0003_cut2",
    "C0005_cut2",
    "C0006_cut2",
    "C0008_cut2",
    "C0011_cut2",
    "C0014_cut2",
    "C0016_cut2",
    "C0024_cut2",
    "C0026_cut2",
    "p043_HAARCREcDAAQDQcbHgANDRQM_cut2",
    "p044_BBMdFxESDBMcEwcVBhMBExQC_cut2",
    "p046_FwQADBEGGwQBGwMdHwAADBAK_RICA_cut2",
]


def convert_case(case_id: str) -> None:
    results_dir = os.path.join(RESULTS_DIR, case_id)
    align_dir   = os.path.join(ALIGN_DIR,   case_id)

    metrics_path = os.path.join(results_dir, "metrics.json")
    lm_path      = os.path.join(align_dir,   "landmarks.npz")

    if not os.path.exists(metrics_path):
        print(f"  SKIP: metrics.json not found in {results_dir}")
        return
    if not os.path.exists(lm_path):
        print(f"  SKIP: landmarks.npz not found in {align_dir}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)
    s_can = float(metrics["s_can"])

    lm      = np.load(lm_path, allow_pickle=True)
    neck_pt = np.array(lm["neck_pt_world"], dtype=np.float64)
    R_frame = np.array(lm["R_frame"],       dtype=np.float64)

    def _to_world(verts_norm: np.ndarray) -> np.ndarray:
        return ((verts_norm.astype(np.float64) * s_can) @ R_frame + neck_pt).astype(np.float32)

    for src_name, dst_name in [
        ("ghd_fitted.obj",          "ghd_fitted_world.obj"),
        ("ghd_fitted_uncapped.obj", "ghd_fitted_uncapped_world.obj"),
    ]:
        src = os.path.join(results_dir, src_name)
        dst = os.path.join(results_dir, dst_name)
        if not os.path.exists(src):
            continue
        V, F = load_obj(src)
        save_obj(dst, _to_world(V), F)
        print(f"  {dst_name}")


def main():
    failed = []
    for case_id in CASES:
        print(f"{'='*55}\n  {case_id}")
        try:
            convert_case(case_id)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((case_id, str(e)))

    print(f"\nDone. {len(CASES) - len(failed)}/{len(CASES)} succeeded.")
    if failed:
        for cid, err in failed:
            print(f"  FAILED {cid}: {err}")


if __name__ == "__main__":
    main()
