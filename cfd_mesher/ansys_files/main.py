"""Extract Fluent CFD results via ensightreader: blood (interior volume) + wall zones."""
from __future__ import annotations

import contextlib
import glob
import os
import sys
import time
import traceback
from itertools import combinations

import numpy as np
import torch
import ensightreader



BLOOD_KEYS = [
    "viscosity_lam", "dp_dt", "dp_dz", "dp_dy", "dp_dx",
    "dz_velocity_dy", "dy_velocity_dz", "dz_velocity_dx",
    "dx_velocity_dz", "dy_velocity_dx", "dx_velocity_dy",
    "dz_velocity_dz", "dy_velocity_dy", "dx_velocity_dx",
    "z_velocity", "y_velocity", "x_velocity",
    "dynamic_pressure", "absolute_pressure",
]

WALL_KEYS = [
    "z_wall_shear", "y_wall_shear", "x_wall_shear", "wall_shear",
]

COORD_KEYS = ("x_coordinate", "y_coordinate", "z_coordinate")

WALL_PARTS = [
    ("wall", "surface3"),
    ("inlet", "surface4"),
    ("outlet_0", "surface5"),
    ("outlet_1", "surface6"), # 
]

_METADATA_FILES = [
    "flowsplit_ratio.txt",
    "geomagic_processed.obj",
    "ghd_coefficients.npz",
    "ghd_forward_fusion_info.npz",
    "ghd_merged_reconstruction.obj",
    "ghd_reconstructed.obj",
    "ghd_smoothed_reconstruction.obj",
    "inlet_centroids.csv",
    "landmarks.npz",
    "metrics.json",
    "branch_ranking.npy",
]

class _Tee:
    """Write to the console and the log file at once, flushing every line.

    The job runs unattended on the cluster, so a crash must leave the log on disk, not in
    a buffer: everything is flushed as it is written.
    """

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        self._fh.write(text)
        self._fh.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


@contextlib.contextmanager
def tee_output(log_path):
    """Send everything printed inside the block to `log_path` as well as the console.

    Whatever raised is written to the log before it propagates, so a failed run explains
    itself without the scheduler's own stdout capture.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    with open(log_path, "a") as fh:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, fh), _Tee(err, fh)
        print("=" * 70)
        print("run started %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
        print("python %s | host %s | pid %d"
              % (sys.version.split()[0], os.uname().nodename, os.getpid()))
        print("log: %s" % log_path)
        print("=" * 70)
        try:
            yield log_path
        except BaseException:
            print("run failed after %.1f s:" % (time.time() - started))
            traceback.print_exc()
            raise
        finally:
            print("run ended %s (%.1f s)"
                  % (time.strftime("%Y-%m-%d %H:%M:%S"), time.time() - started))
            sys.stdout, sys.stderr = out, err


def fix_encas_geometry_model(encas_path):
    """Patch ensight .encas: model line + TIME filename start number from actual .scl files."""
    with open(encas_path, "r") as f:
        lines = f.readlines()
    modified = False
    in_time = False
    for i, line in enumerate(lines):
        if line.strip().startswith("model:") and "ensight_files.geo" in line:
            if "model:  2" in line:
                lines[i] = 'model:  1  "ensight_files.geo"\n'
                modified = True
            continue
        if line.strip() == "TIME":
            in_time = True
            continue
        if in_time and line.strip().startswith("filename start number:"):
            scls = sorted(glob.glob(os.path.join(os.path.dirname(encas_path), "ensight_files*.scl*")))
            if scls:
                try:
                    start = int(os.path.basename(scls[0]).split(".")[0].replace("ensight_files", ""))
                    lines[i] = f"filename start number:      {start}\n"
                    modified = True
                except ValueError:
                    pass
            continue
    if modified:
        with open(encas_path, "w") as f:
            f.writelines(lines)

def face_to_edge_index(faces):
    k = faces.shape[1]
    edges = np.concatenate([faces[:, [i, (i + 1) % k]] for i in range(k)], axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).T.astype(np.int64)


def _read_part_geometry(geofile, part):
    """Return (pos [N,3], cells [C,k]) with connectivity converted to 0-indexed."""
    with open(geofile.file_path, "rb") as fp:
        pos = np.asarray(part.read_nodes(fp), dtype=np.float64)
        cells = np.concatenate(
            [np.asarray(b.read_connectivity(fp), dtype=np.int64) - 1 for b in part.element_blocks],
            axis=0,
        )
    return pos, cells


def _cells_to_edge_index(cells):
    """All-pairs edges per cell (tet -> 6 edges), deduped and undirected. -> [2, E]."""
    k = cells.shape[1]
    edges = np.concatenate([cells[:, list(p)] for p in combinations(range(k), 2)], axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).T.astype(np.int64)


def _read_frames(case, part, keys, n_frames):
    """Read each `key` over `n_frames` timesteps -> {key: [T, N, 1]}."""
    out = {}
    for key in keys:
        cols = []
        for t in range(n_frames):
            v = case.get_variable(key, timestep=t)
            with open(v.file_path, "rb") as fp:
                arr = np.asarray(v.read_node_data(fp, part.part_id), dtype=np.float64)
            cols.append(torch.from_numpy(arr).view(-1, 1).float())
        out[key] = torch.stack(cols, dim=0)
    return out


def _read_pos_frame(case, part):
    """Read xyz coord variables at t=0 -> [N, 3]. Sanity check vs static geometry."""
    cols = []
    for key in COORD_KEYS:
        v = case.get_variable(key, timestep=0)
        with open(v.file_path, "rb") as fp:
            cols.append(np.asarray(v.read_node_data(fp, part.part_id), dtype=np.float64))
    return torch.from_numpy(np.stack(cols, axis=1)).float()


def _print_data_shapes(label, data):
    print(f"[{label}]")
    if isinstance(data, dict) and isinstance(next(iter(data.values())), dict):
        for zone_name, zone in data.items():
            print(f"  {zone_name}")
            for k, v in zone.items():
                print(f"    {k:<25s} {tuple(v.shape)}")
    else:
        for k, v in data.items():
            print(f"  {k:<25s} {tuple(v.shape)}")


def extract_blood_data(encas_path, n_frames=80):
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()
    part = geofile.get_part_by_name("blood")
    pos, cells = _read_part_geometry(geofile, part)
    data = {
        "pos": torch.from_numpy(pos).float(),
        "pos_frame": _read_pos_frame(case, part),            # [N, 3] coords from t=0 vars
        "cells": torch.from_numpy(cells),                    # [C, k] volumetric cell conn
    }
    data.update(_read_frames(case, part, BLOOD_KEYS, n_frames))
    _print_data_shapes("blood", data)
    return data


def extract_wall_data(encas_path, n_frames=80):
    """Per-zone dict; only the wall zone carries WSS/coord time series."""
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()
    out = {}
    for zone_name, part_name in WALL_PARTS:
        part = geofile.get_part_by_name(part_name)
        if part is None:
            continue
        pos, faces = _read_part_geometry(geofile, part)
        zone = {
            "pos": torch.from_numpy(pos).float(),
            "pos_frame": _read_pos_frame(case, part),        # [N, 3] coords from t=0 vars
            "faces": torch.from_numpy(faces),                # [F, k]
        }
        if zone_name == "wall":
            zone.update(_read_frames(case, part, WALL_KEYS, n_frames))
        out[zone_name] = zone
    _print_data_shapes("wall", out)
    return out





def copy_case_metadata(case_dir, out_dir):
    """Copy metadata files from <case_dir>/Results/ into out_dir if they exist."""
    import shutil
    results_dir = os.path.join(case_dir, "Results")
    candidates = list(_METADATA_FILES) + [
        os.path.basename(p)
        for p in glob.glob(os.path.join(results_dir, "output*"))
    ]
    copied, missing = [], []
    for fname in candidates:
        src = os.path.join(results_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            copied.append(fname)
        else:
            missing.append(fname)
    print(f"  metadata  copied={len(copied)}  missing={len(missing)}")


def process_case(case_dir, save_root, n_frames=80):
    """Write `<save_root>/<case_name>/blood_data.pt` and `wall_data.pt`."""
    encas = os.path.join(case_dir, "Results", "ensight_files.encas")
    if not os.path.exists(encas):
        print(f"[skip] {os.path.basename(case_dir)}: no .encas")
        return False
    fix_encas_geometry_model(encas)
    out_dir = os.path.join(save_root, os.path.basename(case_dir))
    os.makedirs(out_dir, exist_ok=True)
    blood_path = os.path.join(out_dir, "blood_data.pt")
    wall_path = os.path.join(out_dir, "wall_data.pt")
    if not os.path.exists(blood_path):
        torch.save(extract_blood_data(encas, n_frames), blood_path)
    if not os.path.exists(wall_path):
        torch.save(extract_wall_data(encas, n_frames), wall_path)
    copy_case_metadata(case_dir, out_dir)
    return True


def export_encase_to_pt(case_root, save_root, n_frames=80, case_pattern="case_*",
                        redo=False, sanity_check=True):
    """
    🔥🔥🔥
    Export all cases under `case_root` matching `case_pattern` to `save_root` as .pt files.
    Skips cases where both .pt files already exist unless redo=True.
    If sanity_check=True, runs sanity_check_case after each successful export.
    """
    sanity_dir = os.path.join(save_root, "_sanity")
    os.makedirs(save_root, exist_ok=True)
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        name = os.path.basename(case_dir)
        out_dir = os.path.join(save_root, name)
        if not redo and os.path.exists(os.path.join(out_dir, "blood_data.pt")) \
                    and os.path.exists(os.path.join(out_dir, "wall_data.pt")):
            print(f"[skip] {name}: already done")
            continue
        t0 = time.time()
        try:
            if process_case(case_dir, save_root, n_frames=n_frames):
                print(f"[ok] {name}  ({time.time() - t0:.1f} s)")
                if sanity_check:
                    try:
                        sanity_check_case(out_dir, out_dir=os.path.join(sanity_dir, name))
                    except Exception as e:
                        # the traceback, not just the message: a sanity failure is usually
                        # a missing key or a shape mismatch, and the line number names it
                        print(f"  [sanity fail] {name}: {e}")
                        traceback.print_exc()
        except Exception as e:
            print(f"[fail] {name}: {e}")
            traceback.print_exc()

"""
Maintaince code session.

"""

_ZONE_STYLE = {
    "surface3": ("steelblue",  0.9,  "wall"),
    "surface4": ("red",        0.9,  "inlet"),
    "surface5": ("yellow",     0.9,  "outlet_0"),
    "surface6": ("black",      0.9,  "outlet_1"),
}
_VTK_TYPE = {3: 5, 4: 10, 8: 12}   # tri, tet, hex


def _to_polydata(pos, cells):
    import pyvista as pv
    k = cells.shape[1]
    flat = np.concatenate([np.full((len(cells), 1), k), cells], axis=1).ravel()
    return pv.PolyData(pos, flat)


def _to_ugrid(pos, cells):
    import pyvista as pv
    k = cells.shape[1]
    flat = np.concatenate([np.full((len(cells), 1), k), cells], axis=1).ravel()
    return pv.UnstructuredGrid(flat, np.full(len(cells), _VTK_TYPE[k]), pos)


_WALL_PART_TO_ZONE = {"surface3": "wall", "surface4": "inlet",
                      "surface5": "outlet_0", "surface6": "outlet_1"}


def plot_case_mesh(case_dir, out_path=None):
    """1x2 plot: surface zones + blood slice.  Reads .pt files if present, else .encas.
    out_path=None  → interactive window.  out_path='...'  → save PNG off-screen."""
    import pyvista as pv

    off_screen = out_path is not None
    pl = pv.Plotter(off_screen=off_screen, shape=(1, 2), window_size=(1600, 800))
    pl.subplot(0, 0); pl.add_text("Surface zones", font_size=12)
    pl.subplot(0, 1); pl.add_text("Blood volume",  font_size=12)

    wall_pt  = os.path.join(case_dir, "wall_data.pt")
    blood_pt = os.path.join(case_dir, "blood_data.pt")

    if os.path.exists(wall_pt) and os.path.exists(blood_pt):
        wall_data  = torch.load(wall_pt,  weights_only=False)
        blood_data = torch.load(blood_pt, weights_only=False)
        for part_name, zone_name in _WALL_PART_TO_ZONE.items():
            if zone_name not in wall_data:
                continue
            color, opacity, label = _ZONE_STYLE[part_name]
            zone = wall_data[zone_name]
            pl.subplot(0, 0)
            pl.add_mesh(_to_polydata(zone["pos"].numpy(), zone["faces"].numpy()),
                        color=color, opacity=opacity,
                        show_edges=True, edge_color="dimgray", line_width=0.3, label=label)
        pl.subplot(0, 0); pl.add_legend(size=(0.25, 0.25))
        pl.subplot(0, 1)
        grid = _to_ugrid(blood_data["pos"].numpy(), blood_data["cells"].numpy())
        pl.add_mesh(grid.slice(normal="z", origin=grid.center),
                    color="lightcoral", show_edges=True, edge_color="white", line_width=0.5)
    else:
        encas = os.path.join(case_dir, "Results", "ensight_files.encas")
        if not os.path.exists(encas):
            print(f"[skip] no .pt or .encas in {case_dir}")
            return
        fix_encas_geometry_model(encas)
        case = ensightreader.read_case(encas)
        geofile = case.get_geometry_model()
        for _, part in geofile.parts.items():
            pos, cells = _read_part_geometry(geofile, part)
            if part.part_name == "blood":
                pl.subplot(0, 1)
                grid = _to_ugrid(pos, cells)
                pl.add_mesh(grid.slice(normal="z", origin=grid.center),
                            color="lightcoral", show_edges=True, edge_color="white", line_width=0.5)
            elif part.part_name in _ZONE_STYLE:
                color, opacity, label = _ZONE_STYLE[part.part_name]
                pl.subplot(0, 0)
                pl.add_mesh(_to_polydata(pos, cells), color=color, opacity=opacity,
                            show_edges=True, edge_color="dimgray", line_width=0.3, label=label)
        pl.subplot(0, 0); pl.add_legend(size=(0.25, 0.25))

    pl.link_views()
    pl.camera.zoom(2.5)
    if out_path:
        pl.screenshot(out_path)
        pl.close()
        print(f"[saved] {out_path}")
    else:
        pl.show()


def sanity_check_case(case_dir, out_dir=None, fps=10):
    """
    🔥🔥🔥
    Save mesh PNG + WSS GIF + velocity GIF from extracted .pt files.
    """
    import pyvista as pv

    if out_dir is None:
        out_dir = case_dir
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(case_dir)

    # --- mesh PNG ---
    plot_case_mesh(case_dir, out_path=os.path.join(out_dir, f"{name}_mesh.png"))

    # --- load .pt files ---
    wall_data  = torch.load(os.path.join(case_dir, "wall_data.pt"),  weights_only=False)
    blood_data = torch.load(os.path.join(case_dir, "blood_data.pt"), weights_only=False)
    wall = wall_data["wall"]

    # wall mesh
    pos_w   = wall["pos"].numpy()
    faces_w = wall["faces"].numpy()
    wall_mesh = _to_polydata(pos_w, faces_w)

    # WSS magnitude  [T, N_wall]
    wss_mag = torch.sqrt(
        wall["x_wall_shear"][:, :, 0] ** 2 +
        wall["y_wall_shear"][:, :, 0] ** 2 +
        wall["z_wall_shear"][:, :, 0] ** 2
    ).numpy()
    T = wss_mag.shape[0]
    wss_clim = [0, float(np.percentile(wss_mag, 99))]

    # velocity magnitude  [T, N_blood]
    vel_mag = torch.sqrt(
        blood_data["x_velocity"][:, :, 0] ** 2 +
        blood_data["y_velocity"][:, :, 0] ** 2 +
        blood_data["z_velocity"][:, :, 0] ** 2
    ).numpy()
    vel_clim = [0, float(np.percentile(vel_mag, 99))]

    # blood point cloud for transparent velocity visualisation
    blood_pcd = pv.PolyData(blood_data["pos"].numpy())
    blood_pcd.point_data["vel"] = vel_mag[0]

    # --- WSS GIF ---
    wall_mesh.point_data["wss"] = wss_mag[0]
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.open_gif(os.path.join(out_dir, f"{name}_wss.gif"), fps=fps)
    pl.add_mesh(wall_mesh, scalars="wss", clim=wss_clim, cmap="jet",
                scalar_bar_args={"title": "WSS mag"})
    pl.camera.zoom(2.5)
    for t in range(T):
        wall_mesh.point_data["wss"] = wss_mag[t]
        pl.render()
        pl.write_frame()
    pl.close()
    print(f"[saved] {name}_wss.gif")

    # --- velocity GIF (transparent point cloud) ---
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.open_gif(os.path.join(out_dir, f"{name}_vel.gif"), fps=fps)
    pl.add_points(blood_pcd, scalars="vel", clim=vel_clim, cmap="jet",
                  point_size=2, opacity=0.3,
                  scalar_bar_args={"title": "Vel mag (m/s)"})
    pl.camera.zoom(2.5)
    for t in range(T):
        blood_pcd.point_data["vel"] = vel_mag[t]
        pl.render()
        pl.write_frame()
    pl.close()
    print(f"[saved] {name}_vel.gif")


def print_parts(case_root, case_pattern="case_*"):
    """Print all parts and face counts for every case under case_root.

    Expects:  <case_root>/<case>/<Results>/ensight_files.encas
    """
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        encas = os.path.join(case_dir, "Results", "ensight_files.encas")
        if not os.path.exists(encas):
            print(f"{os.path.basename(case_dir)}: [no .encas]")
            continue
        print(f"\n=== {os.path.basename(case_dir)} ===")
        try:
            fix_encas_geometry_model(encas)
            case = ensightreader.read_case(encas)
            geofile = case.get_geometry_model()
            for part_id, part in geofile.parts.items():
                _, cells = _read_part_geometry(geofile, part)
                print(f"  part {part_id:3d}  {part.part_name:<30s}  faces={len(cells)}")
        except Exception as e:
            print(f"  [error] {e}")



def test_wall_coords_in_blood(encas_path):
    """Check that every wall zone node exists exactly in the blood node set."""
    from scipy.spatial import cKDTree

    fix_encas_geometry_model(encas_path)
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()

    blood_part = geofile.get_part_by_name("blood")
    blood_pos, _ = _read_part_geometry(geofile, blood_part)
    tree = cKDTree(blood_pos)

    for zone_name, part_name in WALL_PARTS:
        part = geofile.get_part_by_name(part_name)
        if part is None:
            continue
        wall_pos, _ = _read_part_geometry(geofile, part)
        dist, _ = tree.query(wall_pos, k=1, workers=-1)
        n_exact = int(np.sum(dist == 0.0))
        n_near  = int(np.sum(dist < 1e-6))
        print(f"  {zone_name:<12s}  nodes={len(wall_pos)}  "
              f"exact={n_exact}  near(<1e-6)={n_near}  max_dist={dist.max():.3e}")


if __name__ == "__main__":
    CASE_ROOT_list = ["/rds/general/user/wd123/ephemeral/Angioflow/AneuGv2"]
    SAVE_ROOT  = "/rds/general/user/wd123/ephemeral/Angioflow/AneuGv2/processed"

    # One log per run, kept under <SAVE_ROOT>/_logs. Every print below lands there as well
    # as on the console, with a traceback for any case that fails, so a run can be debugged
    # after the fact from the log alone. $PBS_JOBID is in the name when the scheduler set
    # it, so parallel array jobs never write to the same file.
    job = os.environ.get("PBS_JOBID", "").split(".")[0]
    log_name = "export_%s%s.log" % (time.strftime("%Y%m%d_%H%M%S"),
                                    "_job" + job if job else "")
    with tee_output(os.path.join(SAVE_ROOT, "_logs", log_name)):
        for CASE_ROOT in CASE_ROOT_list:
            print("case_root: %s" % CASE_ROOT)
            export_encase_to_pt(
                case_root=CASE_ROOT,
                save_root=SAVE_ROOT,
                n_frames=80,
                case_pattern="case_*",
                redo=False,
                sanity_check=True,
            )

"""
conda activate new
python -m angioflow.data.fluent_export
"""
