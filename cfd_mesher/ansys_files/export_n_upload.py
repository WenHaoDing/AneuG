"""Export Fluent cases to .pt (export.py) and upload each finished case to Hugging Face
while the next one is exporting.

Self-contained: the export code below is copied from export.py (keep the two in step).
The main thread runs the export case by case, exactly as export.py does. Each case whose
export is current (frames.json written) is handed to one background upload thread, which
commits whatever cases have queued up since its last commit in a single upload_folder
call, so a slow upload batches cases instead of falling behind one commit per case.

A case whose folder already exists on the Hub is not uploaded again unless OVERWRITE_HF
is set, so a resubmit after a walltime kill picks up where it left off. With
OVERWRITE_HF the Hub folder is replaced: files there that are not in the local case
folder are deleted in the same commit. With UPLOAD_SANITY the renders in
<SAVE_ROOT>/_sanity/<case>/ go to cfd_data/sanity/<case>/ in a second commit.

Edit the run configuration below (or override it on the command line, see --help). The
write token is read from $HF_TOKEN only and must never be written into a script; pass it
when submitting, through PBS only (not on the login node):

    export HF_TOKEN=hf_...
    qsub -v HF_TOKEN export_n_upload.sh
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import queue
import sys
import threading
import time
import traceback

import numpy as np
import torch
import ensightreader
import huggingface_hub
from huggingface_hub import HfApi

# ---- run configuration --------------------------------------------------------------
HPC_USER = "jp3025"
PROJECT = "AneuGv2"           # cases in /rds/general/user/<HPC_USER>/ephemeral/Angioflow/<PROJECT>
CASE_PATTERN = "cohort_*"     # "" or None: every case folder (one with a Results/ folder)
REPO_ID = "whding123/AneuGv2"
OVERWRITE_HF = False          # True: re-upload cases whose folder already exists on the Hub
SANITY = False                # True: render mesh PNG + WSS/velocity GIFs into <SAVE_ROOT>/_sanity
UPLOAD_SANITY = False         # True: also upload <SAVE_ROOT>/_sanity/<case>/ to cfd_data/sanity/<case>/
# --------------------------------------------------------------------------------------

PATH_IN_REPO = "cfd_data"     # cases land in <repo>/cfd_data/<case>/
SANITY_IN_REPO = PATH_IN_REPO + "/sanity"   # renders land in <repo>/cfd_data/sanity/<case>/
UPLOAD_ATTEMPTS = 3
RETRY_WAIT = 120              # s between attempts of the same batch


# ======================================================================================
# Export: copied from export.py
# ======================================================================================

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

# The journals run 1600 steps of 1e-3 s (two cycles) and export every 10 steps, so the
# .encas holds 160 frames. udf_waveform_5cycles.txt repeats every 0.8 s. The first cycle
# still carries the initialisation transient: export the second one.
CYCLE_PERIOD = 0.8
EXPORT_CYCLE = 2      # 1-based
N_FRAMES = 80         # frames per cycle

# Written last in each case folder, after both .pt files. A case is done only if this
# names the cycle we want, so exports from before it existed (first cycle) are redone.
FRAMES_JSON = "frames.json"

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


def _read_part_geometry(geofile, part):
    """Return (pos [N,3], cells [C,k]) with connectivity converted to 0-indexed."""
    with open(geofile.file_path, "rb") as fp:
        pos = np.asarray(part.read_nodes(fp), dtype=np.float64)
        cells = np.concatenate(
            [np.asarray(b.read_connectivity(fp), dtype=np.int64) - 1 for b in part.element_blocks],
            axis=0,
        )
    return pos, cells


def read_encas_times(encas_path):
    """Time values listed in the .encas TIME section, in timestep order."""
    with open(encas_path, "r") as f:
        lines = f.read().splitlines()
    n_steps, times, collecting = None, [], False
    for line in lines:
        s = line.strip()
        if collecting:
            try:
                times.extend(float(tok) for tok in s.split())
                continue
            except ValueError:
                break
        if s.startswith("number of steps:"):
            n_steps = int(s.split(":", 1)[1])
        elif s.startswith("time values:"):
            collecting = True
            rest = s.split(":", 1)[1].split()
            times.extend(float(tok) for tok in rest)
    if n_steps is not None and len(times) != n_steps:
        raise ValueError(f"{encas_path}: 'number of steps' is {n_steps} "
                         f"but {len(times)} time values are listed")
    return np.asarray(times, dtype=np.float64)


def select_cycle_frames(encas_path, n_frames=N_FRAMES, cycle=EXPORT_CYCLE,
                        period=CYCLE_PERIOD):
    """Timestep indices (and their times) of the frames in t ∈ ((cycle-1)·T, cycle·T].

    Selected by simulated time, not by index, and an incomplete or oddly exported run
    raises instead of quietly handing back frames from the wrong cycle.
    """
    times = read_encas_times(encas_path)
    tol = 0.25 * period / n_frames
    lo, hi = (cycle - 1) * period, cycle * period
    idx = np.nonzero((times > lo + tol) & (times <= hi + tol))[0]
    span = (f"{len(times)} frames in .encas, t = {times[0]:.4f} .. {times[-1]:.4f} s"
            if len(times) else "no time values in .encas")
    if len(idx) != n_frames:
        raise ValueError(f"cycle {cycle} (t in ({lo:.3f}, {hi:.3f}] s) has {len(idx)} "
                         f"frames, expected {n_frames}; {span}")
    dt = np.diff(times[idx])
    if not np.allclose(dt, period / n_frames, rtol=1e-3, atol=1e-9):
        raise ValueError(f"cycle {cycle} frames are not evenly spaced at "
                         f"{period / n_frames:.4f} s (dt {dt.min():.4f} .. {dt.max():.4f}); "
                         f"{span}")
    return idx.astype(np.int64), times[idx]


def _read_frames(case, part, keys, timesteps):
    """Read each `key` at the given timestep indices -> {key: [T, N, 1]}."""
    out = {}
    for key in keys:
        cols = []
        for t in timesteps:
            t = int(t)
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


def extract_blood_data(encas_path, timesteps, times):
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()
    part = geofile.get_part_by_name("blood")
    pos, cells = _read_part_geometry(geofile, part)
    data = {
        "pos": torch.from_numpy(pos).float(),
        "pos_frame": _read_pos_frame(case, part),            # [N, 3] coords from t=0 vars
        "cells": torch.from_numpy(cells),                    # [C, k] volumetric cell conn
        "frame_times": torch.from_numpy(times),              # [T] simulated time (s)
    }
    data.update(_read_frames(case, part, BLOOD_KEYS, timesteps))
    _print_data_shapes("blood", data)
    return data


def extract_wall_data(encas_path, timesteps, times):
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
            zone["frame_times"] = torch.from_numpy(times)    # [T] simulated time (s)
            zone.update(_read_frames(case, part, WALL_KEYS, timesteps))
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


def _save_atomic(obj, path):
    """torch.save via a temp file, so a walltime kill never leaves a truncated .pt."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def export_is_current(out_dir, n_frames=N_FRAMES, cycle=EXPORT_CYCLE):
    """True if out_dir holds both .pt files and a frames.json for this cycle."""
    if not all(os.path.exists(os.path.join(out_dir, f))
               for f in ("blood_data.pt", "wall_data.pt", FRAMES_JSON)):
        return False
    try:
        with open(os.path.join(out_dir, FRAMES_JSON)) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return False
    return info.get("cycle") == cycle and info.get("n_frames") == n_frames


def process_case(case_dir, save_root, n_frames=N_FRAMES, cycle=EXPORT_CYCLE):
    """Write `<save_root>/<case_name>/blood_data.pt`, `wall_data.pt` and frames.json."""
    encas = os.path.join(case_dir, "Results", "ensight_files.encas")
    if not os.path.exists(encas):
        print(f"[skip] {os.path.basename(case_dir)}: no .encas")
        return False
    fix_encas_geometry_model(encas)
    # before anything is written: a run that did not finish the cycle fails here and
    # leaves whatever is already in out_dir alone
    timesteps, times = select_cycle_frames(encas, n_frames, cycle)
    print(f"  cycle {cycle}: timesteps {timesteps[0]}..{timesteps[-1]}, "
          f"t = {times[0]:.4f} .. {times[-1]:.4f} s")
    out_dir = os.path.join(save_root, os.path.basename(case_dir))
    os.makedirs(out_dir, exist_ok=True)
    frames_path = os.path.join(out_dir, FRAMES_JSON)
    if os.path.exists(frames_path):
        os.remove(frames_path)   # the case is not current until both .pt files are rewritten
    _save_atomic(extract_blood_data(encas, timesteps, times),
                 os.path.join(out_dir, "blood_data.pt"))
    _save_atomic(extract_wall_data(encas, timesteps, times),
                 os.path.join(out_dir, "wall_data.pt"))
    copy_case_metadata(case_dir, out_dir)
    with open(frames_path, "w") as f:
        json.dump({"cycle": cycle, "period": CYCLE_PERIOD, "n_frames": n_frames,
                   "timesteps": timesteps.tolist(), "times": times.tolist()}, f, indent=1)
    return True


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

# ======================================================================================
# Upload
# ======================================================================================

def default_case_root(user, project):
    return f"/rds/general/user/{user}/ephemeral/Angioflow/{project}"


def default_save_root(user, project):
    return os.path.join(default_case_root(user, project), "processed")


def find_cases(case_root, pattern, save_root):
    """Case folders under case_root: those matching `pattern`, or with no pattern every
    folder that has a Results/ subfolder. save_root and _/. folders are never cases."""
    if pattern:
        dirs = glob.glob(os.path.join(case_root, pattern))
    else:
        dirs = [os.path.join(case_root, n) for n in os.listdir(case_root)]
    dirs = sorted(d for d in dirs
                  if os.path.isdir(d)
                  and os.path.abspath(d) != os.path.abspath(save_root)
                  and not os.path.basename(d).startswith(("_", ".")))
    if not pattern:
        ignored = [os.path.basename(d) for d in dirs
                   if not os.path.isdir(os.path.join(d, "Results"))]
        if ignored:
            print(f"[ignore] {len(ignored)} folder(s) without Results/: {', '.join(ignored)}")
        dirs = [d for d in dirs if os.path.isdir(os.path.join(d, "Results"))]
    return dirs


class Uploader(threading.Thread):
    """Single background thread: drains the queue and commits each drained batch."""

    def __init__(self, api, repo_id, save_root, overwrite, upload_sanity=False):
        super().__init__(name="uploader", daemon=True)
        self.api, self.repo_id, self.save_root = api, repo_id, save_root
        self.overwrite, self.upload_sanity = overwrite, upload_sanity
        self.q = queue.Queue()
        self.uploaded, self.failed = [], []
        self.sanity_uploaded, self.sanity_failed = [], []

    def submit(self, name):
        self.q.put(name)

    def finish(self):
        """Upload everything still queued, then return."""
        self.q.put(None)
        self.join()

    def run(self):
        done = False
        while not done:
            batch = [self.q.get()]
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            done = None in batch
            batch = [c for c in batch if c is not None]
            if batch:
                self._upload(batch)

    def _upload(self, batch):
        ok = self._commit("case", self.save_root, PATH_IN_REPO, batch,
                          ["*.tmp", "_logs/*", "_sanity/*", ".cache/*"])
        (self.uploaded if ok else self.failed).extend(batch)
        if not self.upload_sanity:
            return
        sanity_root = os.path.join(self.save_root, "_sanity")
        with_renders = [c for c in batch if os.path.isdir(os.path.join(sanity_root, c))]
        missing = sorted(set(batch) - set(with_renders))
        if missing:
            print(f"[upload sanity] no renders for: {', '.join(missing)}")
        if with_renders:
            ok = self._commit("sanity", sanity_root, SANITY_IN_REPO, with_renders, ["*.tmp"])
            (self.sanity_uploaded if ok else self.sanity_failed).extend(with_renders)

    def _commit(self, label, folder, path_in_repo, batch, ignore):
        """One upload_folder commit of <folder>/<case>/ -> <path_in_repo>/<case>/, retried."""
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            t0 = time.time()
            print(f"[upload {label}] {len(batch)} case(s), attempt {attempt}: "
                  f"{', '.join(batch)}")
            try:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=folder,
                    path_in_repo=path_in_repo,
                    repo_type="dataset",
                    allow_patterns=[f"{c}/*" for c in batch],
                    ignore_patterns=ignore,
                    # relative to path_in_repo: clears what is left of the old case folders
                    delete_patterns=[f"{c}/*" for c in batch] if self.overwrite else None,
                    commit_message=f"Upload {len(batch)} processed CFD case(s) ({label}): "
                                   + ", ".join(batch),
                )
                print(f"[upload {label} ok] {len(batch)} case(s) ({time.time() - t0:.1f} s)")
                return True
            except Exception as e:
                print(f"[upload {label} fail] attempt {attempt}: {e}")
                traceback.print_exc()
                if attempt < UPLOAD_ATTEMPTS:
                    time.sleep(RETRY_WAIT)
        return False


def cases_on_hub(api, repo_id):
    """{case: has frames.json} for every case folder on the Hub (empty if the repo is new)."""
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return {}
    out = {}
    for f in files:
        parts = f.split("/")
        # cfd_data/sanity/ holds the renders, it is not a case
        if len(parts) >= 3 and parts[0] == PATH_IN_REPO and parts[1] != "sanity":
            out[parts[1]] = out.get(parts[1], False) or parts[-1] == FRAMES_JSON
    return out


def _flag(p, name, default, help_text):
    """--name / --no-name pair (argparse.BooleanOptionalAction needs Python 3.9)."""
    dest = name.replace("-", "_")
    g = p.add_mutually_exclusive_group()
    g.add_argument(f"--{name}", dest=dest, action="store_true",
                   help=f"{help_text} (default: {default})")
    g.add_argument(f"--no-{name}", dest=dest, action="store_false")
    p.set_defaults(**{dest: default})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", default=HPC_USER)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--case-root", help="default: derived from --user and --project")
    p.add_argument("--save-root", help="default: <case-root>/processed")
    p.add_argument("--case-pattern", default=CASE_PATTERN,
                   help='"" processes every case folder')
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--private", action="store_true",
                   help="create the repo as private if it does not exist yet")
    _flag(p, "overwrite-hf", OVERWRITE_HF,
          "replace case folders that already exist on the Hub")
    _flag(p, "sanity", SANITY, "render mesh PNG + WSS/velocity GIFs after each export")
    _flag(p, "upload-sanity", UPLOAD_SANITY,
          f"also upload each case's renders to {SANITY_IN_REPO}/<case>/")
    p.add_argument("--redo", action="store_true",
                   help="re-export every case locally, even if current")
    args = p.parse_args()
    case_root = args.case_root or default_case_root(args.user, args.project)
    save_root = args.save_root or default_save_root(args.user, args.project)

    # everything that can fail without touching data is checked before the first export
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set: pass it at submit time, e.g. qsub -v HF_TOKEN "
                 "(never write it into a script)")
    if int(huggingface_hub.__version__.split(".")[0]) < 2:
        sys.exit(f"huggingface_hub {huggingface_hub.__version__} is too old, need >= 2.0: "
                 f'pip install -U "huggingface_hub>=2"')
    if not os.path.isdir(case_root):
        sys.exit(f"case root does not exist: {case_root}")

    job = os.environ.get("PBS_JOBID", "").split(".")[0]
    log_name = "export_n_upload_%s%s.log" % (time.strftime("%Y%m%d_%H%M%S"),
                                             "_job" + job if job else "")
    with tee_output(os.path.join(save_root, "_logs", log_name)):
        print(f"case_root: {case_root}\nsave_root: {save_root}\n"
              f"pattern: {args.case_pattern or '(none, every case folder)'}  "
              f"repo: {args.repo_id}  overwrite_hf: {args.overwrite_hf}  "
              f"sanity: {args.sanity}  upload_sanity: {args.upload_sanity}")
        api = HfApi(token=token)
        print(f"[auth] logged in as {api.whoami()['name']}")
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private,
                        exist_ok=True)
        on_hub = cases_on_hub(api, args.repo_id)
        print(f"[hub] {args.repo_id}: {len(on_hub)} case folder(s) already on the Hub")

        def want_upload(name):
            if name not in on_hub or args.overwrite_hf:
                return True
            note = "" if on_hub[name] else " without frames.json (first-cycle upload?)"
            print(f"  [hub skip] {name}: already on the Hub{note}; "
                  f"set OVERWRITE_HF to replace it")
            return False

        uploader = Uploader(api, args.repo_id, save_root, args.overwrite_hf,
                            args.upload_sanity)
        uploader.start()

        sanity_dir = os.path.join(save_root, "_sanity")

        def render_sanity(name, out_dir, only_if_missing=False):
            case_sanity = os.path.join(sanity_dir, name)
            if only_if_missing and all(
                    os.path.exists(os.path.join(case_sanity, f"{name}_{s}"))
                    for s in ("mesh.png", "wss.gif", "vel.gif")):
                return
            try:
                sanity_check_case(out_dir, out_dir=case_sanity)
            except Exception as e:
                print(f"  [sanity fail] {name}: {e}")
                traceback.print_exc()

        os.makedirs(save_root, exist_ok=True)
        export_failed = []
        case_dirs = find_cases(case_root, args.case_pattern, save_root)
        print(f"[cases] {len(case_dirs)} case(s)")
        for case_dir in case_dirs:
            name = os.path.basename(case_dir)
            out_dir = os.path.join(save_root, name)
            if not args.redo and export_is_current(out_dir):
                print(f"[skip] {name}: already exported (cycle {EXPORT_CYCLE})")
                if args.sanity:   # e.g. a walltime kill during the render
                    render_sanity(name, out_dir, only_if_missing=True)
                if want_upload(name):
                    uploader.submit(name)
                continue
            if os.path.exists(os.path.join(out_dir, "blood_data.pt")):
                print(f"[stale] {name}: existing export is not cycle {EXPORT_CYCLE}, "
                      f"re-extracting")
            t0 = time.time()
            try:
                if not process_case(case_dir, save_root):
                    continue
            except Exception as e:
                print(f"[fail] {name}: {e}")
                traceback.print_exc()
                export_failed.append(name)
                continue
            print(f"[ok] {name}  ({time.time() - t0:.1f} s)")
            if args.sanity:   # before submitting, so the renders exist when it uploads
                render_sanity(name, out_dir)
            if want_upload(name):
                uploader.submit(name)

        print("[export] done, waiting for the last upload")
        uploader.finish()
        print(f"[summary] uploaded {len(uploader.uploaded)}, upload failed "
              f"{len(uploader.failed)}, export failed {len(export_failed)}")
        if args.upload_sanity:
            print(f"[summary] sanity uploaded {len(uploader.sanity_uploaded)}, "
                  f"sanity upload failed {len(uploader.sanity_failed)}")
        if uploader.failed:
            print(f"  upload failed (resubmit to retry): {', '.join(uploader.failed)}")
        if uploader.sanity_failed:
            print(f"  sanity upload failed: {', '.join(uploader.sanity_failed)}")
        if export_failed:
            print(f"  export failed: {', '.join(export_failed)}")
    if uploader.failed or uploader.sanity_failed or export_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
export HF_TOKEN=hf_...

# AneuGv2 salvage: replace the first-cycle uploads
qsub -v "HF_TOKEN,HPC_USER=wd123,PROJECT=AneuGv2,CASE_PATTERN=cohort_*,REPO_ID=whding123/AneuGv2,SANITY=true,UPLOAD_SANITY=true,OVERWRITE_HF=true" export_n_upload.sh
qsub -v "HF_TOKEN,HPC_USER=ama324,PROJECT=AneuGv2,CASE_PATTERN=cohort_*,REPO_ID=whding123/AneuGv2,SANITY=false,UPLOAD_SANITY=false,OVERWRITE_HF=true" export_n_upload.sh
qsub -v "HF_TOKEN,HPC_USER=zy3326,PROJECT=AneuGv2,CASE_PATTERN=cohort_*,REPO_ID=whding123/AneuGv2,SANITY=false,UPLOAD_SANITY=false,OVERWRITE_HF=true" export_n_upload.sh



# Angioflow: no pattern, with sanity renders
qsub -v "HF_TOKEN,HPC_USER=wd123,PROJECT=ImperialNHS_fix,CASE_PATTERN=none,REPO_ID=whding123/Angioflow_fix,SANITY=true" export_n_upload.sh

# angioflow_fix
cd $HOME/Angioflow/angioflow_fix/0000_array_log
qstat -u $USER
qsub -l walltime=2:00:00 -v "HF_TOKEN,HPC_USER=$USER,PROJECT=angioflow_fix,CASE_PATTERN=none,REPO_ID=whding123/Angioflow_fix,SANITY=false" export_n_upload.sh

qsub -J 27-28 -l walltime=16:00:00 hpc_bash_dispatch.sh
"""
