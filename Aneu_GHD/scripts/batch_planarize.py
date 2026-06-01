"""
Batch-apply planarize_openings to all clipped reconstruction variants
across every case in ImperialNHS_batch1.

Variant → fusion info mapping:
  clipped_reconstruction_fallback_manual.ply  → forward_fusion_info_fallback.npz
  clipped_reconstruction_fallback.ply         → forward_fusion_info_fallback.npz
  clipped_reconstruction_manual.ply           → forward_fusion_info.npz
  clipped_reconstruction.ply                  → forward_fusion_info.npz

Output: <stem>_patched.ply alongside the source file.
"""

import os
import sys
import pyvista as pv

# Allow imports from the project root (AneuG/Aneu_GHD)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.patching import planarize_openings

BATCH_DIR = "/media/yaplab2/HDD Storage/almaha/cases/ImperialNHS_batch1"

# (mesh filename, fusion info filename, output filename)
VARIANTS = [
    ("clipped_reconstruction_fallback_manual.ply", "forward_fusion_info_fallback.npz",
     "clipped_reconstruction_fallback_manual_patched.ply"),
    ("clipped_reconstruction_fallback.ply",        "forward_fusion_info_fallback.npz",
     "clipped_reconstruction_fallback_patched.ply"),
    ("clipped_reconstruction_manual.ply",          "forward_fusion_info.npz",
     "clipped_reconstruction_manual_patched.ply"),
    ("clipped_reconstruction.ply",                 "forward_fusion_info.npz",
     "clipped_reconstruction_patched.ply"),
]

def process_case(case_dir):
    # Try variants in priority order; stop at the first one found
    for mesh_name, fusion_name, out_name in VARIANTS:
        mesh_path = os.path.join(case_dir, mesh_name)
        if not os.path.exists(mesh_path):
            continue

        fusion_path = os.path.join(case_dir, fusion_name)
        if not os.path.exists(fusion_path):
            return [], [f"  SKIP {mesh_name}: fusion info '{fusion_name}' not found"]

        out_path = os.path.join(case_dir, out_name)
        print(f"  Processing {mesh_name} ...")
        mesh = pv.read(mesh_path)
        mesh_out = planarize_openings(
            mesh,
            post_dir=case_dir,
            r_forward_fusion_info_filename=fusion_name,
            n_iter=10,
            lam=0.5,
        )
        pv.save_meshio(out_path, mesh_out)
        return [f"  -> saved {out_name}"], []

    return [], []  # no variant found


def main():
    case_dirs = sorted([
        os.path.join(BATCH_DIR, d)
        for d in os.listdir(BATCH_DIR)
        if os.path.isdir(os.path.join(BATCH_DIR, d))
    ])

    total_ok = 0
    total_skip = 0

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        print(f"\n[{case_name}]")
        processed, skipped = process_case(case_dir)

        if not processed and not skipped:
            print("  no clipped reconstruction files found")
        for msg in processed:
            print(msg)
        for msg in skipped:
            print(msg)

        total_ok += len(processed)
        total_skip += len(skipped)

    print(f"\nDone. {total_ok} file(s) patched, {total_skip} skipped.")


if __name__ == "__main__":
    main()


"""

cd "/media/yaplab2/HDD Storage/almaha/AneuG/Aneu_GHD"
python scripts/batch_planarize.py

"""