import urllib
import numpy as np
import utils.vessel_reconstruct as vr
import os
import pandas as pd
from tqdm import tqdm

def ask_params_terminal(fields: dict) -> dict:
    """
    Prompt the user to fill in parameters via the terminal (headless-friendly).

    Parameters
    ----------
    fields : dict
        {label: default_value} — each entry becomes a prompted input field
        pre-filled with its default value.

    Returns
    -------
    dict
        {label: value_string} — raw strings as entered by the user.
        Parse (int/float/list) as needed after the call.
    """
    print("=== Parameters ===")
    result = {}
    for label, default in fields.items():
        raw = input(f"  {label} [{default}]: ").strip()
        result[label] = raw if raw else str(default)
    print("==================")
    return result

def _parse_float_list(s: str) -> list:
    return [float(x) for x in s.split() if x.strip() and x.strip().lower() != "none"]

def _ask_propagation_params(defaults: dict) -> dict:
    mcl = defaults["max_cl_length"]
    mcl_default = " ".join(str(v) for v in mcl) if isinstance(mcl, list) else str(mcl)
    p = ask_params_terminal({
        "flaw_opening_min_size":                defaults["flaw_opening_min_size"],
        "init_step":                            defaults["init_step"],
        "max_cl_length (list, space-separated)": mcl_default,
        "ring_downsample_ratio":                defaults["ring_downsample_ratio"],
        "smooth_n_rings":                       defaults["smooth_n_rings"],
        "smooth_n_iter":                        defaults["smooth_n_iter"],
        "extrude_outlets (y/n)":                "y" if defaults.get("extrude_outlets", False) else "n",
        "extrude_inlet (y/n)":                  "y" if defaults.get("extrude_inlet", False) else "n",
        "min_torsion (y/n)":                    "y" if defaults.get("min_torsion", False) else "n",
    })
    vals = _parse_float_list(p["max_cl_length (list, space-separated)"])
    return {
        "flaw_opening_min_size": int(p["flaw_opening_min_size"] or defaults["flaw_opening_min_size"]),
        "init_step":             int(p["init_step"]             or defaults["init_step"]),
        "max_cl_length":         vals[0] if len(vals) == 1 else (vals if vals else defaults["max_cl_length"]),
        "ring_downsample_ratio": int(p["ring_downsample_ratio"] or defaults["ring_downsample_ratio"]),
        "smooth_n_rings":        int(p["smooth_n_rings"]        or defaults["smooth_n_rings"]),
        "smooth_n_iter":         int(p["smooth_n_iter"]         or defaults["smooth_n_iter"]),
        "extrude_outlets":       p["extrude_outlets (y/n)"].strip().lower().startswith("y"),
        "extrude_inlet":         p["extrude_inlet (y/n)"].strip().lower().startswith("y"),
        "min_torsion":           p["min_torsion (y/n)"].strip().lower().startswith("y"),
    }


def save_params(params: dict, case_dir: str, filename="ghd_fusion_params.json"):
    import json
    path = os.path.join(case_dir, filename)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def copy_files(src_dir, dst_dir, files_to_copy: list):
    for filename in files_to_copy:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        if os.path.exists(src):
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {src} to {dst}")
            else:
                print(f"File {dst} already exists, skipping copy.")
        else:
            print(f"Source file {src} does not exist, cannot copy.")


def print_aneurysm_type(post_dir):
    endpoints_path = os.path.join(post_dir, "endpoints_manual.npy")
    if os.path.exists(endpoints_path):
        endpoints_data = np.load(endpoints_path, allow_pickle=True).item()
        aneurysm_type = endpoints_data.get('aneurysm_type', 'Unknown')
        print(f"Aneurysm type: {aneurysm_type}")
    else:
        print(f"Endpoints file not found at {endpoints_path}, cannot determine aneurysm type.")

"""
Journal:
manually fix:

"""

if __name__ == "__main__":
    # Dir where original cliiped meshes are stored.
    post_dir = "/media/yaplab2/HDD Storage/wenhao/AneuSeg/PostTr_debug/ImperialNHS"
    # Dir where the GHD fitting results are stored.
    ghd_dir = "fitting_results/job1"
    # Dir where processed results are stored.
    save_root = "/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/processed_shapes/batch_3"
    extrude_outlets = True
    extrude_inlet = False
    use_clipped_gt_mesh = True

    # output filenames.
    w_merged_mesh_filename = "ghd_merged_reconstruction.obj"
    w_smoothed_mesh_filename = "ghd_smoothed_reconstruction.obj"
    files_to_copy = ["forward_fusion_info.npz", "branch_ranking.npy"]
    redo = False

    cases = os.listdir(post_dir)
    case_manager = pd.read_excel("/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/case_registry.xlsx")

    n_existing = len(os.listdir(save_root)) if os.path.exists(save_root) else 0
    print(f"[{save_root}] contains {n_existing} cases")

    case_to_exclude = [
        # "flXboN0NJD_aneurysm1",  # holes, gotta fix manually
        "8t4XoA0zQ0_aneurysm1", "tjXzxbGi8S_aneurysm1" # gotta fix the segmentation
        
    ]
    case_to_rewrite = [184, 201, 215]

    pbar = tqdm(cases, desc="Cases")
    for case in pbar:
        pbar.set_postfix_str(case)
        if case in case_to_exclude:
            continue
        print(f"[{case}] processing")
        
        matched = case_manager.loc[case_manager['original_name'] == case, 'index']
        force_rewrite = False
        if not matched.empty:
            case_index = int(matched.values[0])
            print(f"[{case}] matched in registry, index={case_index}")
            if case_index in case_to_rewrite:
                force_rewrite = True
                print(f"[{case}] index {case_index} in case_to_rewrite, forcing rewrite")
            else:
                continue
        else:
            preprocess_log = os.path.join(post_dir, case, "preprocess_log.txt")
            if not os.path.exists(preprocess_log):
                print(f"[{case}] not in registry and no preprocess_log at {preprocess_log}, skipping")
                continue
            else:
                aneurysm_type = np.load(os.path.join(post_dir, case, "endpoints_manual.npy"), allow_pickle=True).item()['aneurysm_type']
                if int(aneurysm_type) > 2:
                    continue
            print(f"[{case}] not in registry, but preprocess_log exists -> proceeding")
        save_dir = os.path.join(save_root, case)
        os.makedirs(save_dir, exist_ok=True)
        try:
            copy_files(os.path.join(post_dir, case), save_dir, files_to_copy)
        except Exception as e:
            print(f"Error copying files for case {case}: {e}")
            continue
        output_mesh_path = os.path.join(save_dir, w_smoothed_mesh_filename)
        if os.path.exists(output_mesh_path) and not redo and not force_rewrite:
            print(f"Smoothed mesh already exists for case {case}, skipping. ({output_mesh_path})")
            continue
        params = dict(flaw_opening_min_size=5, init_step=1, max_cl_length=None, ring_downsample_ratio=1, smooth_n_rings=5, smooth_n_iter=10, extrude_outlets=extrude_outlets, extrude_inlet=extrude_inlet, min_torsion=False)
        # use clipped ground truth mesh if use_clipped_gt_mesh
        if not use_clipped_gt_mesh:
            r_ghd_mesh_filename = os.path.join(ghd_dir, case, "ghd_fitted_uncapped_world.obj")
        else:
            r_ghd_mesh_filename = os.path.join(post_dir, case, "clipped_reconstruction.ply")
        print_aneurysm_type(post_dir=os.path.join(post_dir, case))
        while True:
            _ = vr.ghd_forward_mesh_fusion(
                save_dir = save_dir,
                r_ghd_mesh_filename = r_ghd_mesh_filename,
                r_clipped_cl_filename = os.path.join(post_dir, case, "clipped_centerline.npy"),
                r_forward_fusion_info_filename = os.path.join(post_dir, case, "forward_fusion_info.npz"),
                r_branch_ranking_filename=os.path.join(post_dir, case, "branch_ranking.npy"),
                w_ghd_reconstructed_filename = os.path.join(save_dir, "ghd_reconstructed.obj"),
                w_ghd_forward_fusion_info_filename=os.path.join(save_dir, "ghd_forward_fusion_info.npz"),
                w_merged_mesh_filename=os.path.join(save_dir, w_merged_mesh_filename),
                w_smoothed_mesh_filename=os.path.join(save_dir, w_smoothed_mesh_filename),
                **params
            )

            print(f"  Merged mesh:\n    file://{urllib.parse.quote(output_mesh_path, safe='/')}")

            if input("Redo propagation? (y/n, default n): ").lower() != 'y':
                break
            params = _ask_propagation_params(params)
        save_params(params, os.path.join(save_root, case))

"""
cd AneuG/Aneu_GHD
conda activate vmtk_autogen
python ghd_forward_mesh_fusion.py


6CEUwdLVY7_aneurysm1 hole
sdfgnnRzJL_aneurysm1 trim
bPyKGh0SGQ_aneurysm2 spike
CFV81HMHqU_aneurysm1 spike
QsLUrELCDq_aneurysm1 spike
jX927SPGZL_aneurysm1 trim
qXdeQXeju1_aneurysm2 delete and fix before propagating
5y61ssrDxn_aneurysm1 delete and fix before propagating
GtJPG36fHo_aneurysm1 trim
HZTGU6EAEz_aneurysm1 delete and fix before propagating
XxWinm78X4_aneurysm1 spike
iQsrvhNKjH_aneurysm2 trim
I4vL5p57t4_mxf9_aneurysm1 trim
vbfoaHqAT8_aneurysm1 spike
jntttq38tf_aneurysm1 spike
fPl4CB4npr_aneurysm2 delete and check rank
8dVIkrLOF3_aneurysm1 gotta manually extrude
ReCexteoIH_aneurysm1 trim
I4vL5p57t4_z19o_aneurysm1 trim
wiYQO6BmRS_aneurysm1 trim
KZ1T8EkshQ_aneurysm1 trim
JFtZYj7T3t_aneurysm1 gotta manually extrude
oUiXl6WLr1_aneurysm1 delete and fix manually
iQsrvhNKjH_aneurysm1 delete and fix source
iy6PJTLKXk_aneurysm3 delete and check rank
BcXeCx3voF_aneurysm1 trim
sdfgnnRzJL_aneurysm2 trim
bPyKGh0SGQ_aneurysm1 trim
eLNFbBQbJ0_aneurysm1 trim
eM6xqoqoo1_aneurysm1 trim
HUfWo4Tb22_aneurysm1 trim
VbC0oFd7nx_aneurysm1 trim
iy6PJTLKXk_aneurysm2 trim
GQNJfE8a2B_aneurysm2 manually extrude
YzNWocbVXH_aneurysm1 delete and manually fix
AwTkB4fFo5_aneurysm2 trim
qYs10dknc6_aneurysm1 trim
r65LsG8iY4_aneurysm2 trim
jpg6q6ISEs_aneurysm1 trim
aqzkopeLuo_aneurysm1 trim

fPl4CB4npr_aneurysm2 manually extrude
iQsrvhNKjH_aneurysm1 clip and extrude
"""