import utils.vessel_reconstruct as vr
import os

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
    return [float(x) for x in s.split() if x.strip()]

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
    })
    vals = _parse_float_list(p["max_cl_length (list, space-separated)"])
    return {
        "flaw_opening_min_size": int(p["flaw_opening_min_size"] or defaults["flaw_opening_min_size"]),
        "init_step":             int(p["init_step"]             or defaults["init_step"]),
        "max_cl_length":         vals[0] if len(vals) == 1 else (vals if vals else defaults["max_cl_length"]),
        "ring_downsample_ratio": int(p["ring_downsample_ratio"] or defaults["ring_downsample_ratio"]),
        "smooth_n_rings":        int(p["smooth_n_rings"]        or defaults["smooth_n_rings"]),
        "smooth_n_iter":         int(p["smooth_n_iter"]         or defaults["smooth_n_iter"]),
    }


def save_params(params: dict, case_dir: str, filename="ghd_fusion_params.json"):
    import json
    path = os.path.join(case_dir, filename)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    # Dir where original cliiped meshes are stored.
    post_dir = "seg_post/ImperialNHS_batch1"
    # Dir where the GHD fitting results are stored.
    ghd_dir = "fitting_results/ImperialNHS_batch1"
    # output filenames.
    w_merged_mesh_filename = "ghd_merged_reconstruction.ply"
    w_smoothed_mesh_filename = "ghd_smoothed_reconstruction.ply"
    redo = True

    cases = os.listdir(post_dir)

    for case in cases:
        try:
            output_mesh_path = os.path.join(ghd_dir, case, w_smoothed_mesh_filename)
            if os.path.exists(output_mesh_path) and not redo:
                print(f"Smoothed mesh already exists for case {case}, skipping. ({output_mesh_path})")
                continue
            params = dict(flaw_opening_min_size=5, init_step=1, max_cl_length=10, ring_downsample_ratio=1, smooth_n_rings=3, smooth_n_iter=10)
            while True:
                _ = vr.ghd_forward_mesh_fusion(
                    post_dir=os.path.join(post_dir, case),
                    ghd_dir=os.path.join(ghd_dir, case),
                    w_merged_mesh_filename=w_merged_mesh_filename,
                    w_ghd_forward_fusion_info_filename="ghd_forward_fusion_info.npz",
                    **params
                )
                print(f"  Merged mesh:\n    {output_mesh_path}")
                if input("Redo propagation? (y/n, default n): ").lower() != 'y':
                    break
                params = _ask_propagation_params(params)
            save_params(params, os.path.join(ghd_dir, case))
        except Exception as e:
            print(f"Error processing case {case}: {e}")

"""
cd Aneu_GHD
python ghd_forward_mesh_fusion.py

"""