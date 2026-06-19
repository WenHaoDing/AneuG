import urllib
import numpy as np
import utils.vessel_reconstruct as vr
import os
import pandas as pd
import pyvista as pv
import SimpleITK as sitk
from tqdm import tqdm


def fix_mesh_affine(mesh_path, label_path, out_path=None):
    """Re-apply the SITK direction matrix that `forward_marching_cubes` dropped upstream.

    Buggy meshes were saved at `origin + spacing*idx` instead of
    `origin + direction @ (spacing*idx)`. Reverse map per vertex:
        V_correct = origin + direction @ (V_buggy - origin)
    No-op (returns False) when direction == I.
    """
    img = sitk.ReadImage(label_path)
    direction = np.array(img.GetDirection()).reshape(3, 3)
    if np.allclose(direction, np.eye(3)):
        return False
    origin = np.array(img.GetOrigin())
    mesh = pv.read(mesh_path)
    mesh.points = origin + (mesh.points - origin) @ direction.T
    mesh.save(out_path or mesh_path)
    return True

def render_alignment_check(mesh_path, label_path, save_dir,
                           aneurysm_label_value: int = 2,
                           artery_label_value: int = 1):
    """Render mesh + label surfaces together to verify alignment. Saves PNGs into
    `save_dir/alignment_check/`. Three views (overview, dome closeup, side) so
    misalignment is easy to spot visually.
    """
    out_dir = os.path.join(save_dir, 'alignment_check')
    os.makedirs(out_dir, exist_ok=True)

    img = sitk.ReadImage(label_path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    size = img.GetSize()

    def _to_world(surf):
        if surf.n_points == 0 or np.allclose(direction, np.eye(3)):
            return surf
        T = np.eye(4); T[:3, :3] = direction
        T[:3, 3] = origin - direction @ origin
        return surf.transform(T, inplace=False)

    grid = pv.ImageData(dimensions=size, spacing=spacing, origin=tuple(origin))
    grid.point_data['lbl'] = arr.transpose(2, 1, 0).flatten(order='F').astype(np.uint8)

    art_surf = _to_world(grid.contour(isosurfaces=[artery_label_value - 0.5],
                                      scalars='lbl'))
    dome_surf = _to_world(grid.contour(isosurfaces=[aneurysm_label_value - 0.5],
                                       scalars='lbl'))
    mesh = pv.read(mesh_path)

    # Three angles (azimuth around vertical), plus an overview, so misalignment
    # in any axis becomes obvious.
    closeup_angles = [('closeup_front', 0, 15),
                      ('closeup_side',  90, 15),
                      ('closeup_top',   0, 80)]
    views = [('overview', None, None, None)] + [
        (name, dome_surf, az, el) for (name, az, el) in closeup_angles
    ]

    for view_name, focus_surf, azimuth, elevation in views:
        plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))
        plotter.set_background('white')
        if art_surf.n_points > 0:
            plotter.add_mesh(art_surf, color='lightblue', opacity=0.25,
                             label=f'arteries (lbl=={artery_label_value})')
        if dome_surf.n_points > 0:
            plotter.add_mesh(dome_surf, color='yellow', opacity=0.85,
                             smooth_shading=True,
                             label=f'aneurysm dome (lbl=={aneurysm_label_value})')
        plotter.add_mesh(mesh, color='red', style='wireframe', line_width=1.5,
                         opacity=0.9,
                         label=f'mesh (n_cells={mesh.n_cells})')

        if focus_surf is not None and focus_surf.n_points > 0:
            xmin, xmax, ymin, ymax, zmin, zmax = focus_surf.bounds
            cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
            extent = float(max(xmax - xmin, ymax - ymin, zmax - zmin))
            az_r = np.deg2rad(azimuth)
            el_r = np.deg2rad(elevation)
            dist = 10.0 * extent
            cam_x = cx + dist * np.cos(el_r) * np.sin(az_r)
            cam_y = cy + dist * np.cos(el_r) * np.cos(az_r)
            cam_z = cz + dist * np.sin(el_r)
            plotter.camera_position = [(cam_x, cam_y, cam_z),
                                       (cx, cy, cz),
                                       (0, 0, 1)]
            plotter.camera.focal_point = (cx, cy, cz)
        else:
            plotter.view_isometric()

        plotter.add_legend()
        plotter.add_axes()
        plotter.add_title(f'{os.path.basename(save_dir)} | {view_name}',
                          font_size=10)
        plotter.show(screenshot=os.path.join(out_dir, f'{view_name}.png'),
                     auto_close=True)


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
    save_root = "/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/cfd_mesh_reference"
    extrude_outlets = False
    extrude_inlet = False
    use_clipped_gt_mesh = True

    # output filenames.
    w_merged_mesh_filename = "ghd_merged_reconstruction.obj"
    w_smoothed_mesh_filename = "ghd_smoothed_reconstruction.obj"
    files_to_copy = ["forward_fusion_info.npz", "branch_ranking.npy"]
    redo = False

    cases = os.listdir(post_dir)

    n_existing = len(os.listdir(save_root)) if os.path.exists(save_root) else 0
    print(f"[{save_root}] contains {n_existing} cases")

    case_to_exclude = [
        # "flXboN0NJD_aneurysm1",  # holes, gotta fix manually
        "8t4XoA0zQ0_aneurysm1", "tjXzxbGi8S_aneurysm1" # gotta fix the segmentation
        
    ]

    pbar = tqdm(cases, desc="Cases")
    for case in pbar:
        pbar.set_postfix_str(case)
        if case in case_to_exclude:
            continue
        print(f"[{case}] processing")
        
        preprocess_log = os.path.join(post_dir, case, "preprocess_log.txt")
        if not os.path.exists(preprocess_log):
            print(f"[{case}] not in registry and no preprocess_log at {preprocess_log}, skipping")
            continue
        else:
            aneurysm_type = np.load(os.path.join(post_dir, case, "endpoints_manual.npy"), allow_pickle=True).item()['aneurysm_type']
            if int(aneurysm_type) > 2:
                continue
        print(f"[{case}]: preprocess_log exists -> proceeding")
        save_dir = os.path.join(save_root, case)
        os.makedirs(save_dir, exist_ok=True)
        try:
            copy_files(os.path.join(post_dir, case), save_dir, files_to_copy)
        except Exception as e:
            print(f"Error copying files for case {case}: {e}")
            continue
        output_mesh_path = os.path.join(save_dir, w_smoothed_mesh_filename)
        if os.path.exists(output_mesh_path) and not redo:
            print(f"Smoothed mesh already exists for case {case}, skipping. ({output_mesh_path})")
            continue
        params = dict(flaw_opening_min_size=5, init_step=1, max_cl_length=[10, 5, 5], ring_downsample_ratio=1, 
                      smooth_n_rings=5, smooth_n_iter=10, extrude_outlets=False, extrude_inlet=False, 
                      min_torsion=True)
        # use clipped ground truth mesh if use_clipped_gt_mesh
        if not use_clipped_gt_mesh:
            r_ghd_mesh_filename = os.path.join(ghd_dir, case, "ghd_fitted_uncapped_world.obj")
        else:
            r_ghd_mesh_filename = os.path.join(post_dir, case, "clipped_reconstruction.ply")
        print_aneurysm_type(post_dir=os.path.join(post_dir, case))
        _ = vr.ghd_forward_mesh_fusion(
            save_dir = save_dir,
            r_ghd_mesh_filename = r_ghd_mesh_filename,
            r_clipped_cl_filename = os.path.join(post_dir, case, "clipped_centerline.npy"),
            r_forward_fusion_info_filename = os.path.join(post_dir, case, "forward_fusion_info.npz"),
            r_branch_ranking_filename = os.path.join(post_dir, case, "branch_ranking.npy"),
            w_ghd_reconstructed_filename = os.path.join(save_dir, "ghd_reconstructed.obj"),
            w_ghd_forward_fusion_info_filename=os.path.join(save_dir, "ghd_forward_fusion_info.npz"),
            w_merged_mesh_filename=os.path.join(save_dir, w_merged_mesh_filename),
            w_smoothed_mesh_filename=os.path.join(save_dir, w_smoothed_mesh_filename),
            **params
        )

        # Re-apply the SITK direction matrix dropped by regional-growth's
        # marching_cubes step. All inputs to ghd_forward_mesh_fusion live in the same
        # buggy frame, so a single affine on the outputs corrects them in one go.
        label_path = os.path.join(post_dir, case, "label.nrrd")
        if os.path.exists(label_path):
            for out_name in (w_smoothed_mesh_filename, w_merged_mesh_filename,
                             "ghd_reconstructed.obj"):
                p = os.path.join(save_dir, out_name)
                if os.path.exists(p) and fix_mesh_affine(p, label_path):
                    print(f"  Fixed affine on {out_name}")
        else:
            print(f"  [warn] label.nrrd missing for {case}; skipped affine fix")

        # if os.path.exists(label_path) and os.path.exists(output_mesh_path):
        #     try:
        #         render_alignment_check(output_mesh_path, label_path, save_dir)
        #         print(f"  Alignment check saved to {os.path.join(save_dir, 'alignment_check')}")
        #     except Exception as e:
        #         print(f"  [warn] alignment render failed for {case}: {e}")

        print(f"  Merged mesh:\n    file://{urllib.parse.quote(output_mesh_path, safe='/')}")

"""
cd AneuG/Aneu_GHD
conda activate vmtk_autogen
python create_cfd_mesh_reference.py


"""