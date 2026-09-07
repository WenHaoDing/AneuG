"""manage.py -- single entry point for tracking + re-running GHD-fitting
results across (potentially several) datasets.

Keeps one CSV manifest (default: runtime/manifest.csv, at the repo root --
deliberately NOT inside any one dataset's runtime/<Dataset>/ folder, so it
can carry rows for multiple datasets as more get processed) with one row
per case: where it lives, whether it finished/failed, its best chamfer, and
reviewer-owned columns (review_status, refit_config, refit_align_config,
refit_stage_a_ratio) that this script never overwrites once set.

LAYOUT: results are filed by CONFIG, not by dataset --
    runtime/<config>/<dataset>/<case>/
so every config's results for the whole corpus sit together and one case can
hold a result from each config side by side. (The older layout put everything
under runtime/<dataset>/; `scan --dataset-root` still folds that in.)

Each case row carries one status column PER CONFIG:
    cfg_default  cfg_with_opening  cfg_arap_init  cfg_clone  cfg_clone_target
      ✓ complete    ~ still fitting    ✗ ran and failed    (blank) never run
plus config_history, the reviewer's VERDICT per config ("default=redo;
clone=pass"). The two differ: a config can complete cleanly (✓) and still be
rejected on sight. "Already tried" means either -- so a case that has burned
two configs never offers them again on the third pass.

Config axes, each with its own named-preset registry:
  - fit config    (fit_configs.py's CONFIGS)     -- five presets; two of them
                  (arap_init, clone) also take a Stage-A MSE:mesh-health ratio.
  - align config  (align_configs.py's ALIGN_CONFIGS) -- alignment.py's --w-*
                  flags. Only applies to configs that run run_case.py; the
                  ARAP/clone pipelines run their own Stage 1 and refuse it.

Subcommands:

  scan          Walk runtime/<config>/<dataset>/ for EVERY config and upsert
                one row per case, filling the per-config status columns.
                No --dataset-root needed -- it finds the config folders itself.
                Safe to re-run while a batch is still going: a case touched in
                the last RUNNING_WINDOW_MIN minutes is marked ~ (in progress)
                rather than ✗ (failed), and ~ never counts as "already tried".

  next          Pop the single next case that still needs a human look --
                failed ones first (by name), then done-but-unreviewed ones
                worst-chamfer first -- and print its image path (a raw-mesh
                render for failed cases, cached under the system temp dir,
                NOT the fitting case folder; the pipeline's own
                sanity/sanity_final.png for done cases). Drives the
                interactive review loop: Claude reads/shows the image, asks
                for a verdict, calls `update`, then calls `next` again.

  update        Record one case's review verdict: --status
                {pass,skip,redo,reclip,fix} (+ --config NAME and/or
                --align-config NAME if redo -- at least one required, the
                other defaults to "default", + optional --notes). "reclip"
                flags a case as needing AneuSeg/reclip.py run manually
                before it's worth refitting here; "fix" flags some other
                manual intervention needed (bad landmarks, bad label.nrrd,
                etc.) before refitting. Never called directly by you --
                Claude writes this after you give a verdict in chat.

  review-list   Non-interactive overview: print up to --limit cases needing
                review per bucket (failed / done-unreviewed), for a quick
                scan without stepping through the interactive loop.

  review        The one-line interactive loop: pops each case's image open
                in VSCode, prompts in the terminal for a verdict
                (p/s/d/c/q), records it, moves to the next case. Run this
                yourself -- `next`/`update` below are its building blocks,
                useful if Claude is driving the review instead of you.

  requeue       Read the manifest, find every row with a non-empty
                refit_config and/or refit_align_config, and launch those
                cases (only those) in tmux workers, each config running its
                OWN script (run_case.py / fit_with_arap_init.py /
                clone_fit.py) into runtime/<config>/<dataset>/<case>/ -- never
                overwriting the case's result under a different config.

  configs       Print each config and the exact flags it resolves to.
  history       Show which configs each case has already tried, and what is
                left -- so a third attempt doesn't repeat a dead end.

Usage:
  # scan finds runtime/<config>/<dataset>/ on its own; --geometry-root is
  # only needed once, so requeue knows where each case's source geometry is.
  # ONE manifest holds every dataset (keyed by dataset+case), so scanning a
  # second dataset ADDS rows rather than replacing anything -- every other
  # subcommand takes --dataset to filter.
  python scripts/fit/manage.py scan --dataset ImperialNHS \
      --geometry-root "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"

  python scripts/fit/manage.py scan --dataset AneuX \
      --geometry-root "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX"

  python scripts/fit/manage.py review --dataset ImperialNHS
  python scripts/fit/manage.py review --dataset AneuX

  python scripts/fit/manage.py configs
  python scripts/fit/manage.py history --dataset ImperialNHS --tried-any
  python scripts/fit/manage.py next --dataset ImperialNHS
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status pass
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status redo --config clone --stage-a-ratio 5
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status redo --config arap_init

  python scripts/fit/manage.py requeue --gpus cuda:0 cuda:1 cuda:2
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_configs import (CONFIGS, OLD_FIT_ROOT, STAGE_A_RATIOS,  # noqa: E402
                         DEFAULT_STAGE_A_RATIO, stage_a_ratio_flags, resolve_flags,
                         output_dir as cfg_output_dir, build_fit_command,
                         old_case_dir)
from align_configs import ALIGN_CONFIGS, ALIGN_NEEDS_BRANCH, align_flags  # noqa: E402

# Fitting outputs, logs and the manifest live under runtime_fitting/ -- kept
# apart from runtime/ (scratch), runtime_dataset/ (the prepared corpus) and
# runtime_train/ (model checkpoints). These fits are expensive to reproduce, so
# they get their own root rather than sharing one with disposable scratch.
FITTING_ROOT = ROOT / "runtime_fitting"
DEFAULT_MANIFEST = FITTING_ROOT / "manifest.csv"
DEFAULT_PYTHON = "/home/yaplab2/miniconda3/envs/new/bin/python3"
DEFAULT_LOG_DIR = FITTING_ROOT / "_logs"
# Review-only renders (raw mesh for failed cases) live here, NOT in the
# fitting case folder -- these are throwaway review artifacts, not pipeline
# output, and are safe to wipe any time.
SCRATCH_DIR = Path(tempfile.gettempdir()) / "ghd_review_cache"
# Fixed path every case's image gets copied into before popping it open --
# same path each time means VSCode reuses/refreshes one tab across the whole
# review session instead of stacking a new tab per case.
CURRENT_IMAGE_PATH = SCRATCH_DIR / "_current_review.png"
CONDA_INIT = ('source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || '
              'source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate new')

# One status column per config: OK_MARK when that config produced a complete
# result, FAIL_MARK when it ran but left the output incomplete, blank when it
# was never run for that case.
OK_MARK, FAIL_MARK, RUNNING_MARK = "\u2713", "\u2717", "~"   # ✓ / ✗ / in progress

# A case being fitted RIGHT NOW looks identical on disk to one that crashed:
# the output dir exists but has no ghd_coefficients.npz yet. Distinguish them
# by recency -- if anything in the dir was touched within this many minutes,
# call it in-progress rather than failed. Prevents a mid-run scan from
# recording false failures (which would also wrongly count as "already tried").
RUNNING_WINDOW_MIN = 45
CONFIG_COLUMNS = [f"cfg_{name}" for name in CONFIGS]

# Which config's numbers fill the scalar columns (chamfer_best etc.) when more
# than one succeeded -- clone first, matching the run priority.
PRIMARY_ORDER = ["clone", "clone_no_opening", "arap_init", "with_opening",
                 "default", "clone_target"]

FIELDNAMES = [
    "dataset", "case_name", "geometry_dir", "aneurysm_type", "save_dir", "status",
    "config_used", "align_used", "chamfer_best", "chamfer_final", "best_iter",
    "last_run", *CONFIG_COLUMNS, "config_history", "accepted_config", "review_status", "refit_config",
    "refit_align_config", "refit_stage_a_ratio",
    # Which rank-ordered branch gets the extra endpoint weight, for the align
    # configs in ALIGN_NEEDS_BRANCH. Empty for every other align config.
    "refit_focus_branch",
    # When the redo was ASKED FOR. scan clears a redo only once the requested
    # config has a run NEWER than this. Without it, asking to redo a config
    # that already had an old successful run was "fulfilled" by the very next
    # scan -- which wiped review_status and pushed the case back into the
    # review queue, so reviewing appeared to lose progress.
    "refit_requested_at",
    "notes",
    # Hand-maintained outside this script: ticked when a case marked "fix" has
    # actually had its manual work done. Listed here so save_manifest carries it
    # through -- DictWriter writes only FIELDNAMES, so a column missing from
    # this list is silently dropped on the next write.
    "fixed",
]


# ── per-config VERDICT history ────────────────────────────────────────────
# The cfg_<name> columns record whether a config MECHANICALLY produced output.
# That is not the same as whether you accepted it: a config can complete fine
# and still be rejected on sight. config_history records the reviewer's verdict
# per config -- "default=redo; with_opening=redo; clone=pass" -- so a case that
# has already burned two configs never offers them again on the third pass.

def history_dict(row):
    out = {}
    for chunk in (row.get("config_history") or "").split(";"):
        chunk = chunk.strip()
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def history_str(d):
    return "; ".join(f"{k}={v}" for k, v in sorted(d.items()))


def record_verdict(row, config_name, verdict):
    """Remember that `config_name` was reviewed and given `verdict`."""
    if not config_name:
        return
    h = history_dict(row)
    h[config_name] = verdict
    row["config_history"] = history_str(h)


def tried_configs(row):
    """Configs already attempted -- reviewed with a verdict, or run and failed."""
    tried = set(history_dict(row))
    # A config that produced a result (OK) or crashed (FAIL) has been tried,
    # whether or not a verdict has been recorded for it yet -- at the review
    # prompt the verdict for the config being looked at is not written until
    # AFTER the answer, so keying off history alone would report it untried.
    # RUNNING_MARK is deliberately excluded: a case still being fitted has not
    # been tried yet and must stay retryable.
    tried.update(name for name in CONFIGS
                 if row.get(f"cfg_{name}") in (OK_MARK, FAIL_MARK))
    return tried


def untried_configs(row):
    return [n for n in CONFIGS if n not in tried_configs(row)]


def format_history(row):
    """One-line summary of what has been tried on this case."""
    h = history_dict(row)
    bits = []
    for name in CONFIGS:
        mark = row.get(f"cfg_{name}") or ""
        verdict = h.get(name)
        if verdict:
            bits.append(f"{name}={verdict}{' ' + mark if mark else ''}")
        elif mark:
            bits.append(f"{name}:{mark}")
    line = ", ".join(bits) if bits else "(nothing tried yet)"
    acc = row.get("accepted_config")
    if acc:
        line += f"   [ACCEPTED: {acc}]"
    return line


def _config_result_newest(runtime_root, config_name, dataset, case_name,
                          running_window=RUNNING_WINDOW_MIN):
    """(mark, dir, metrics, align_config) for a config's newest run on a case,
    looking across EVERY align variant, not just the default one.

    A redo under a non-default align config writes to
    runtime/<config>_<align>/..., which _config_result alone never looks at --
    so those runs were invisible: scan reported an older default-align run as
    the case's latest, and review showed its render. 68 of the 86 redos
    currently queued use a non-default align config, so this was most of them.

    Completed runs win over in-progress/failed ones, and among completed runs
    the newest by artefact mtime wins -- the same "newest is the one under
    review" rule scan already applies across configs.
    """
    found = []
    for align in ALIGN_CONFIGS:
        mark, d, m = _config_result(runtime_root, config_name, dataset, case_name,
                                    running_window, align_config=align)
        if mark:
            found.append((mark, d, m, align))
    if not found:
        return "", cfg_output_dir(config_name, dataset, case_name, runtime_root), None, "default"
    ok = [f for f in found if f[0] == OK_MARK]
    pool = ok if ok else found
    return max(pool, key=lambda f: _result_mtime(f[1]))


def _dir_age_minutes(d):
    """Minutes since anything in `d` was last written."""
    newest = max((f.stat().st_mtime for f in d.rglob("*") if f.is_file()), default=None)
    if newest is None:
        newest = d.stat().st_mtime
    return (datetime.datetime.now().timestamp() - newest) / 60.0


def _result_mtime(d):
    """Newest of a result dir's completion artefacts, or 0.0 if none exist."""
    ts = [f.stat().st_mtime for f in (d / "ghd_coefficients.npz", d / "metrics.json",
                                      d / "ghd_fitted.obj") if f.exists()]
    return max(ts) if ts else 0.0


def _config_result(runtime_root, config_name, dataset, case_name, running_window=RUNNING_WINDOW_MIN,
                   align_config="default"):
    """(mark, case_dir, metrics) for one config's result on one case.

    A dir without both ghd_coefficients.npz and ghd_fitted.obj either crashed
    or is still being written. Recently-touched means in-progress (~), stale
    means failed (✗) -- so scanning mid-run doesn't record false failures.
    """
    d = cfg_output_dir(config_name, dataset, case_name, runtime_root, align_config)
    if not d.is_dir():
        return "", d, None
    if not ((d / "ghd_coefficients.npz").exists() and (d / "ghd_fitted.obj").exists()):
        return (RUNNING_MARK if _dir_age_minutes(d) < running_window else FAIL_MARK), d, None
    m = None
    mp = d / "metrics.json"
    if mp.exists():
        try:
            m = json.loads(mp.read_text())
        except (ValueError, OSError):
            m = None
    return OK_MARK, d, m


def _combo_dir_name(dataset, align_cfg, fit_cfg):  # DEPRECATED -- see below
    """Folder name for one (align config, fit config) combination -- e.g.
    ImperialNHS_endpoint_focus_with_opening.

    DEPRECATED: results are now filed by config
    (runtime/<config>/<dataset>/<case>/, see fit_configs.output_dir), which is
    what scan() reads. Kept only to locate output from batches run under the
    old layout."""
    parts = [dataset]
    if align_cfg and align_cfg != "default":
        parts.append(align_cfg)
    if fit_cfg and fit_cfg != "default":
        parts.append(fit_cfg)
    return "_".join(parts)


def load_manifest(path):
    rows = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["dataset"], row["case_name"])] = {k: row.get(k, "") or "" for k in FIELDNAMES}
    return rows


def save_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["dataset"], r["case_name"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})


def _chamfer(row, key="chamfer_best"):
    try:
        return float(row[key])
    except (ValueError, TypeError, KeyError):
        return None


def _unsupported_aneurysm_type(geometry_dir):
    """True if geometry_dir/endpoints_manual.npy records aneurysm_type > 2 --
    this pipeline only supports 0=bifurcated, 1/2=sidewall; anything higher
    is a type the fitting code was never built for, so scan auto-skips it
    rather than surfacing it in review."""
    if not geometry_dir:
        return False
    ep_path = Path(geometry_dir) / "endpoints_manual.npy"
    if not ep_path.exists():
        return False
    try:
        ep = np.load(ep_path, allow_pickle=True).item()
        return int(ep["aneurysm_type"]) > 2
    except Exception:
        return False


def cmd_scan(args):
    """Upsert one manifest row per case, with a per-config status column.

    Walks runtime/<config>/<dataset>/ for every known config rather than a
    single output dir, because results are now filed by CONFIG, not dataset.
    --dataset-root still works as a legacy single-directory override.
    """
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    runtime_root = Path(args.runtime_root).resolve()
    geometry_root = Path(args.geometry_root).resolve() if args.geometry_root else None

    # every case that any config has produced output for, plus (if given) every
    # case the geometry root knows about -- so never-run cases still get a row
    case_names = set()
    for name in CONFIGS:
        d = runtime_root / name / args.dataset
        if d.is_dir():
            case_names.update(c.name for c in d.iterdir() if c.is_dir())
    if args.dataset_root:
        legacy = Path(args.dataset_root).resolve()
        if legacy.is_dir():
            case_names.update(c.name for c in legacy.iterdir() if c.is_dir())
    if geometry_root and geometry_root.is_dir() and args.include_unrun:
        case_names.update(c.name for c in geometry_root.iterdir()
                          if c.is_dir() and (c / "endpoints_manual.npy").exists())

    n_done = n_failed = n_pending = n_running = 0
    tally = {name: [0, 0, 0] for name in CONFIGS}       # [ok, failed, running]

    for case_name in sorted(case_names):
        key = (args.dataset, case_name)
        row = rows.get(key) or {k: "" for k in FIELDNAMES}
        row["dataset"] = args.dataset
        row["case_name"] = case_name
        if geometry_root:
            row["geometry_dir"] = str(geometry_root / case_name)

        results = {}
        for name in CONFIGS:
            # newest across ALL align variants -- cfg_<name> means "this fit
            # config's best run, whichever align config produced it"
            mark, d, m, align = _config_result_newest(runtime_root, name,
                                                      args.dataset, case_name)
            row[f"cfg_{name}"] = mark
            results[name] = (mark, d, m, align)
            if mark == OK_MARK:
                tally[name][0] += 1
            elif mark == FAIL_MARK:
                tally[name][1] += 1
            elif mark == RUNNING_MARK:
                tally[name][2] += 1

        # Scalar columns -- and therefore the render the review loop shows --
        # come from the MOST RECENTLY produced result, not from a fixed config
        # precedence. Reviewing a case means judging the fit you just ran; a
        # config-priority rule would keep showing an older run (e.g. `clone`)
        # after you had deliberately refitted with something else, so the image
        # on screen would not be the one your verdict is about.
        _ok = [(n, d) for n, (mk, d, _m, _a) in results.items() if mk == OK_MARK]
        primary = None
        if _ok:
            def _when(nd):
                d = nd[1]
                cands = [d / "ghd_coefficients.npz", d / "metrics.json", d / "ghd_fitted.obj"]
                ts = [f.stat().st_mtime for f in cands if f.exists()]
                return max(ts) if ts else 0.0
            primary = max(_ok, key=_when)[0]
        if primary:
            _, d, m, primary_align = results[primary]
            row["status"] = "done"
            row["config_used"] = primary
            # the align config that actually produced the run on screen, not
            # whatever was recorded when the row was last written
            row["align_used"] = primary_align
            row["save_dir"] = str(d)
            if m:
                if isinstance(m.get("chamfer_best"), (int, float)):
                    row["chamfer_best"] = f"{m['chamfer_best']:.6f}"
                if isinstance(m.get("chamfer_final"), (int, float)):
                    row["chamfer_final"] = f"{m['chamfer_final']:.6f}"
                if m.get("best_iter") is not None:
                    row["best_iter"] = str(m["best_iter"])
                if m.get("aneurysm_type") is not None:
                    row["aneurysm_type"] = str(m["aneurysm_type"])
            coeffs = d / "ghd_coefficients.npz"
            if coeffs.exists():
                row["last_run"] = datetime.datetime.fromtimestamp(
                    coeffs.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            n_done += 1

            # A requested redo is fulfilled only once the requested config has a
            # run NEWER than the request. Comparing on "cfg_X is OK" alone made
            # any redo of an already-run config self-clear on the next scan.
            if row["refit_config"] and row[f"cfg_{row['refit_config']}"] == OK_MARK:
                asked = row.get("refit_requested_at") or ""
                if not asked:
                    # Pre-dates this column: stamp it and leave the redo standing
                    # rather than guess. The next scan judges it properly.
                    row["refit_requested_at"] = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")
                else:
                    # A redo with a non-default align config writes to
                    # runtime/<config>_<align>/..., which _config_result never
                    # looks at (it takes no align_config), so cfg_<config>'s
                    # mtime would never move and the redo would stand forever.
                    # Judge fulfilment against the dir the rerun actually used.
                    _al = row.get("refit_align_config") or "default"
                    _d = cfg_output_dir(row["refit_config"], args.dataset, case_name,
                                        runtime_root, _al)
                    try:
                        _asked_ts = datetime.datetime.strptime(
                            asked, "%Y-%m-%d %H:%M:%S").timestamp()
                    except ValueError:
                        _asked_ts = 0.0
                    if _result_mtime(_d) > _asked_ts:
                        row["refit_config"] = ""
                        row["refit_align_config"] = ""
                        row["refit_stage_a_ratio"] = ""
                        row["refit_focus_branch"] = ""
                        row["refit_requested_at"] = ""
                        row["review_status"] = ""
        elif any(v[0] == RUNNING_MARK for v in results.values()):
            row["status"] = "running"
            n_running += 1
        elif any(v[0] == FAIL_MARK for v in results.values()):
            row["status"] = "failed"
            n_failed += 1
        else:
            row["status"] = "pending"
            n_pending += 1

        if not row["review_status"] and _unsupported_aneurysm_type(row["geometry_dir"]):
            row["review_status"] = "skip"
            if not row["notes"]:
                row["notes"] = "auto-skipped: unsupported aneurysm_type > 2"

        rows[key] = row

    save_manifest(manifest_path, rows)
    print(f"[{args.dataset}] {len(case_names)} case(s): {n_done} done, "
          f"{n_running} running, {n_failed} failed, {n_pending} pending")
    print("  per config:  " + "   ".join(
        f"{n}: {OK_MARK}{tally[n][0]} {FAIL_MARK}{tally[n][1]} {RUNNING_MARK}{tally[n][2]}"
        for n in CONFIGS))
    if n_running:
        print(f"  ({n_running} case(s) touched in the last {RUNNING_WINDOW_MIN} min -- "
              f"still being fitted, not failures. Re-scan when the run finishes.)")
    print(f"Manifest: {manifest_path} ({len(rows)} row(s) total, all datasets).")


def _ensure_offscreen_render():
    """Make pyvista render headlessly, even when DISPLAY points at an X server
    that cannot do GLX (e.g. an `ssh -X` forward).

    pv.system_supports_plotting() returns True whenever DISPLAY is merely SET,
    so the usual `if not DISPLAY: start_xvfb()` guard never fires over SSH --
    VTK then fails to create a GLX context and calls abort(), killing the
    process outright. That is not a catchable Python exception, so it has to be
    prevented rather than handled: drop DISPLAY and render on xvfb, which is
    what these offscreen sanity renders want in every environment anyway.
    """
    import os as _os
    import pyvista as _pv
    _os.environ.pop("DISPLAY", None)
    try:
        _pv.start_xvfb()
    except Exception:
        pass


def render_raw_mesh_sanity(mesh_path, out_path, n_angles=4):
    """Quick multi-angle screenshot of a raw .ply -- used for failed cases,
    which never got far enough to produce a fitted sanity_final.png. No
    canonical overlay, just the case's own clipped/unclipped surface, so the
    reviewer can see e.g. "oh, this one was never clipped" at a glance."""
    _ensure_offscreen_render()
    import pyvista as pv
    import trimesh


    mesh = trimesh.load(mesh_path, process=False)
    pv_mesh = pv.wrap(mesh)

    focal = mesh.vertices.mean(axis=0)
    diag = np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0))
    cam_dist = diag * 1.6 if diag > 0 else 1.0

    n_cols = min(2, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    plotter = pv.Plotter(off_screen=True, shape=(n_rows, n_cols),
                          window_size=(480 * n_cols, 480 * n_rows), border=True)
    for i in range(n_angles):
        row, col = divmod(i, n_cols)
        plotter.subplot(row, col)
        angle_deg = round(360 * i / n_angles)
        angle_rad = np.deg2rad(angle_deg)
        cam_pos = focal + cam_dist * np.array([np.cos(angle_rad), np.sin(angle_rad), 0.3])
        plotter.add_mesh(pv_mesh, color="lightcoral", show_edges=False)
        plotter.add_text(f"{angle_deg} deg", font_size=10, position="upper_edge")
        plotter.set_background("white")
        plotter.camera.position = cam_pos
        plotter.camera.focal_point = focal
        plotter.camera.up = (0.0, 0.0, 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def _raw_mesh_sanity_for(row):
    """Best-effort raw-mesh render for a FAILED row: prefer clipped_reconstruction
    (real kept mesh), then its _fallback variant (type-2 short-clip convention),
    then plain reconstruction.ply (case was never clipped at all -- usually the
    reason it failed: "that aneurysm is weird and I didn't clip anything")."""
    geo_dir = Path(row["geometry_dir"]) if row["geometry_dir"] else None
    if not geo_dir or not geo_dir.exists():
        return None
    candidates = [
        geo_dir / "clipped_reconstruction.ply",
        geo_dir / "clipped_reconstruction_fallback.ply",
        geo_dir / "reconstruction.ply",
    ]
    mesh_path = next((c for c in candidates if c.exists()), None)
    if mesh_path is None:
        return None
    out_path = SCRATCH_DIR / row["dataset"] / f"{row['case_name']}_raw_mesh_sanity.png"
    if not out_path.exists():
        try:
            render_raw_mesh_sanity(mesh_path, out_path)
        except Exception as e:
            print(f"  (render failed for {row['case_name']}: {e})", file=sys.stderr)
            return None
    return str(out_path)


def _review_queue(items):
    """failed rows (by name) first, then done-but-unreviewed worst-chamfer first."""
    failed = sorted([r for r in items if r["status"] == "failed" and not r["review_status"]],
                     key=lambda r: r["case_name"])
    done_unreviewed = [r for r in items if r["status"] == "done" and not r["review_status"]]
    done_unreviewed.sort(key=lambda r: (_chamfer(r) is None, -(_chamfer(r) or 0)))
    return failed, done_unreviewed


def case_runs(row, runtime_root=None):
    """Every config run that exists on disk for this case.

    Results are filed per config, so one case can hold several completed runs
    side by side, and a config can additionally have one run per align
    config. Returns [(config, mark, dir, metrics_or_None, align_config), ...] in
    PRIMARY_ORDER, i.e. the order used to pick which one fills the row's
    scalar columns.
    """
    runtime_root = Path(runtime_root) if runtime_root else FITTING_ROOT
    out = []
    for name in CONFIGS:
        # every (config, align) pair that produced something, not just the
        # default-align dir -- a redo under endpoint_focus/skeleton_focus is a
        # separate run on disk and has to be listed as one
        for align in ALIGN_CONFIGS:
            mark, d, m = _config_result(runtime_root, name, row["dataset"],
                                        row["case_name"], align_config=align)
            if mark:
                out.append((name, mark, d, m, align))

    # newest first -- the run under review is the newest one, so it heads the
    # list rather than being buried under an older config that merely sorts first
    def _when(t):
        d = t[2]
        ts = [f.stat().st_mtime for f in
              (d / "ghd_coefficients.npz", d / "metrics.json", d / "ghd_fitted.obj")
              if f.exists()]
        return max(ts) if ts else 0.0
    out.sort(key=_when, reverse=True)
    return out


def render_for(config_name, case_dir):
    """The sanity render a given config's run produced, if it exists."""
    p = Path(case_dir) / "sanity" / "sanity_final.png"
    return p if p.exists() else None


def format_runs(runs, current=None, current_align=None):
    """Multi-line summary of every run a case has, marking which is on screen."""
    if not runs:
        return "    (no runs on disk)"
    lines = []
    for name, mark, d, m, align in runs:
        label = name if align == "default" else f"{name}+{align}"
        cb = m.get("chamfer_best") if m else None
        cb_s = f"chamfer={cb:.6f}" if isinstance(cb, (int, float)) else "chamfer=-"
        _ts = [f.stat().st_mtime for f in
               (d / "ghd_coefficients.npz", d / "metrics.json", d / "ghd_fitted.obj")
               if f.exists()]
        cb_s += ("  " + datetime.datetime.fromtimestamp(max(_ts)).strftime("%m-%d %H:%M")) if _ts else ""
        extra = ""
        if m:
            if m.get("mesh_health_scale_stage_a") is not None:
                extra += f"  mh_scale={m['mesh_health_scale_stage_a']}"
            if m.get("lambda_rigid_stage_a") is not None:
                extra += f"  rigid_a={m['lambda_rigid_stage_a']}"
            if m.get("reframe_mean_surface_dist") is not None:
                extra += f"  reframe={m['reframe_mean_surface_dist']:.4f}mm"
        is_current = (name == current and
                      (current_align is None or align == current_align))
        if is_current:
            lines.append(f"    {mark} {label:<26} {cb_s}{extra}   <-- ON SCREEN")
            # only the shown run's path is printed. Listing every run's path
            # made it ambiguous which image the viewer had actually opened.
            img = render_for(name, d)
            if img:
                lines.append(f"        {img}")
        else:
            lines.append(f"    {mark} {label:<26} {cb_s}{extra}")
    return "\n".join(lines)


def choose_fallback_run(row, runs):
    """Prompt for which already-completed run to accept.

    Exploring configs is not monotonic: the second thing you try can be worse
    than the first. Without this you would have to refit the earlier config
    just to end up where you already were -- and the manifest would still point
    at the newest run, which is the one you rejected. Recording an
    accepted_config instead keeps every run on disk and simply names the winner.
    """
    usable = [(n, mk, d, m, al) for (n, mk, d, m, al) in runs if mk == OK_MARK]
    if not usable:
        print("  no completed runs to fall back to")
        return None
    print("  completed runs for this case:")
    for i, (n, mk, d, m, al) in enumerate(usable, 1):
        cb = (m or {}).get("chamfer_best")
        cb_s = f"chamfer={cb:.6f}" if isinstance(cb, (int, float)) else "chamfer=-"
        img = render_for(n, d)
        print(f"    [{i}] {(n if al == 'default' else n + '+' + al):<26} {cb_s}")
        if img:
            print(f"        {img}")
    raw = input(f"  accept which? [1-{len(usable)}, or config name, blank=cancel] > ").strip()
    if not raw:
        return None
    chosen = None
    if raw.isdigit() and 1 <= int(raw) <= len(usable):
        chosen = usable[int(raw) - 1][0]
    else:
        chosen = next((n for n, _, _, _ in usable if n == raw), None)
    if chosen is None:
        print(f"  {raw!r} is not one of the completed runs -- cancelled")
        return None
    return chosen


def _pop_image(image_path):
    """Copy image_path over CURRENT_IMAGE_PATH and open that fixed path in
    VSCode -- reopening the SAME path each call makes VSCode reuse/refresh
    its one already-open tab instead of stacking a new tab per case."""
    if not image_path:
        return
    CURRENT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(image_path, CURRENT_IMAGE_PATH)
    except Exception as e:
        print(f"  (could not stage image for preview: {e})")
        return
    try:
        subprocess.Popen(["code", str(CURRENT_IMAGE_PATH)])
    except Exception as e:
        print(f"  (could not auto-open image in VSCode: {e})")


def cmd_next(args):
    rows = load_manifest(Path(args.manifest))
    items = list(rows.values())
    if args.dataset:
        items = [r for r in items if r["dataset"] == args.dataset]
    items = [r for r in items if r.get("status") != "running"]
    failed, done_unreviewed = _review_queue(items)
    queue = failed + done_unreviewed
    if not queue:
        print("NOTHING_LEFT=1")
        return
    r = queue[0]
    if r["status"] == "failed":
        image = _raw_mesh_sanity_for(r) or ""
        image_kind = "raw_mesh"
    else:
        sanity = Path(r["save_dir"]) / "sanity" / "sanity_final.png"
        image = str(sanity) if sanity.exists() else ""
        image_kind = "fitted_sanity"
    print(f"dataset={r['dataset']}")
    print(f"case_name={r['case_name']}")
    print(f"status={r['status']}")
    print(f"config_used={r['config_used']}")
    print(f"chamfer_best={r['chamfer_best']}")
    print(f"image_kind={image_kind}")
    print(f"image={image}")
    print(f"remaining_failed={len(failed)}")
    print(f"remaining_done_unreviewed={len(done_unreviewed)}")
    if image and not args.no_pop:
        _pop_image(image)


def cmd_update(args):
    if not args.status and not args.accept_config:
        print("ERROR: give --status, or --accept-config to accept an existing run.")
        sys.exit(1)
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    key = (args.dataset, args.case)
    if key not in rows:
        print(f"ERROR: no manifest row for dataset={args.dataset!r} case={args.case!r}")
        sys.exit(1)
    if args.status == "redo":
        fit_cfg = args.config or "default"
        align_cfg = args.align_config or "default"
        # Results are filed per config, so "default" IS a new run for a case
        # fitted under some other config. Only refuse when the requested config
        # has genuinely already been tried -- handled by the tried-check below.
        if fit_cfg not in CONFIGS:
            print(f"ERROR: unknown fit config {fit_cfg!r}. Known configs: {sorted(CONFIGS)}")
            sys.exit(1)
        if align_cfg not in ALIGN_CONFIGS:
            print(f"ERROR: unknown align config {align_cfg!r}. Known align configs: {sorted(ALIGN_CONFIGS)}")
            sys.exit(1)
        if align_cfg in ALIGN_NEEDS_BRANCH and args.focus_branch is None:
            print(f"ERROR: align config {align_cfg!r} weights ONE branch's endpoint term 3x, "
                  f"so it needs --focus-branch <index> (rank-ordered, 0 = upstream/dome-first).")
            sys.exit(1)
        if args.focus_branch is not None:
            if align_cfg not in ALIGN_NEEDS_BRANCH:
                print(f"ERROR: --focus-branch given but align config {align_cfg!r} does not use "
                      f"one. Configs that do: {sorted(ALIGN_NEEDS_BRANCH)}")
                sys.exit(1)
            if args.focus_branch < 0:
                print(f"ERROR: --focus-branch must be >= 0, got {args.focus_branch}")
                sys.exit(1)
        if args.stage_a_ratio is not None:
            if not CONFIGS[fit_cfg]["supports_stage_a_ratio"]:
                print(f"ERROR: --stage-a-ratio given but config {fit_cfg!r} has no Stage A. "
                      f"Only these do: "
                      f"{sorted(c for c, v in CONFIGS.items() if v['supports_stage_a_ratio'])}")
                sys.exit(1)
            if args.stage_a_ratio <= 0:
                print(f"ERROR: --stage-a-ratio must be positive, got {args.stage_a_ratio}")
                sys.exit(1)

    row = rows[key]
    if args.accept_config:
        if args.accept_config not in CONFIGS:
            print(f"ERROR: unknown config {args.accept_config!r}. Known: {sorted(CONFIGS)}")
            sys.exit(1)
        if row.get(f"cfg_{args.accept_config}") != OK_MARK:
            print(f"ERROR: {args.accept_config!r} has no completed run for "
                  f"{args.dataset}/{args.case} (cfg column = "
                  f"{row.get(f'cfg_{args.accept_config}')!r}); nothing to accept.")
            sys.exit(1)
        if row.get("config_used") and row["config_used"] != args.accept_config:
            record_verdict(row, row["config_used"], "worse")
        record_verdict(row, args.accept_config, "pass")
        row["accepted_config"] = args.accept_config
        row["review_status"] = "pass"
        row["refit_config"] = row["refit_align_config"] = row["refit_stage_a_ratio"] = ""
        row["refit_focus_branch"] = ""
        if args.notes is not None:
            row["notes"] = args.notes
        rows[key] = row
        save_manifest(manifest_path, rows)
        print(f"Accepted {args.accept_config} for {args.dataset}/{args.case} "
              f"(newest run was {row.get('config_used')!r})")
        return

    if args.status == "redo" and args.config and args.config in tried_configs(row) and not args.force:
        prev = history_dict(row).get(args.config, "ran/failed")
        print(f"ERROR: {args.config!r} was already tried on {args.dataset}/{args.case} "
              f"(verdict: {prev}).")
        print(f"  history: {format_history(row)}")
        untried = untried_configs(row)
        print(f"  not yet tried: {', '.join(untried) if untried else '(none)'}")
        print("  Pass --force to run it again anyway.")
        sys.exit(1)
    record_verdict(row, row.get("config_used"), args.status)
    row["review_status"] = args.status
    row["refit_config"] = (args.config or "default") if args.status == "redo" else ""
    row["refit_align_config"] = (args.align_config or "default") if args.status == "redo" else ""
    row["refit_focus_branch"] = (str(args.focus_branch)
                                 if args.status == "redo" and args.focus_branch is not None else "")
    if args.status == "redo" and CONFIGS[args.config or "default"]["supports_stage_a_ratio"]:
        row["refit_stage_a_ratio"] = str(
            args.stage_a_ratio if args.stage_a_ratio is not None
            else (CONFIGS[args.config or "default"]["default_stage_a_ratio"]
                  or DEFAULT_STAGE_A_RATIO))
    else:
        row["refit_stage_a_ratio"] = ""
    if args.notes is not None:
        row["notes"] = args.notes
    rows[key] = row
    save_manifest(manifest_path, rows)
    print(f"Updated {args.dataset}/{args.case}: review_status={row['review_status']} "
          f"refit_config={row['refit_config']!r} refit_align_config={row['refit_align_config']!r} "
          f"refit_stage_a_ratio={row['refit_stage_a_ratio']!r}"
          f"{' focus_branch=' + row['refit_focus_branch'] if row.get('refit_focus_branch') else ''}")


_STATUS_KEYS = {"p": "pass", "pass": "pass", "s": "skip", "skip": "skip",
                "d": "redo", "redo": "redo", "c": "reclip", "reclip": "reclip",
                "f": "fix", "fix": "fix"}


def cmd_review(args):
    """One-line interactive REPL: pop image -> prompt -> record -> repeat,
    imitating AneuSeg/reclip.py's own input()-driven review loop."""
    manifest_path = Path(args.manifest)
    while True:
        rows = load_manifest(manifest_path)
        items = list(rows.values())
        if args.dataset:
            items = [r for r in items if r["dataset"] == args.dataset]
        items = [r for r in items if r.get("status") != "running"]
        failed, done_unreviewed = _review_queue(items)
        queue = failed + done_unreviewed
        if not queue:
            print("Nothing left to review.")
            return

        r = queue[0]
        key = (r["dataset"], r["case_name"])
        runs = case_runs(r)
        shown_config = None
        if r["status"] == "failed":
            image = _raw_mesh_sanity_for(r)
            kind = "raw mesh (never finished fitting)"
        else:
            shown_config = r["config_used"] or (runs[0][0] if runs else None)
            sanity = Path(r["save_dir"]) / "sanity" / "sanity_final.png"
            image = str(sanity) if sanity.exists() else None
            kind = f"config={shown_config}  chamfer_best={r['chamfer_best']}"

        print(f"\n=== {r['dataset']}/{r['case_name']} -- {kind}  ({len(queue)} left) ===")
        if len(runs) > 1:
            print(f"  {len(runs)} runs on disk for this case:")
            print(format_runs(runs, current=shown_config, current_align=r.get("align_used")))
        elif runs:
            print(f"  only run: {runs[0][0]}")
        if image:
            _pop_image(image)
        else:
            print("  (no image found)")

        raw = input("  [p]ass  [s]kip  [d]redo  [c]reclip  [f]ix  [b]ack(accept an "
                    "earlier run)  [q]uit > ").strip().lower()
        if raw in ("q", "quit"):
            print("Stopping.")
            return
        if raw in ("b", "back", "fallback"):
            picked = choose_fallback_run(rows[key], runs)
            if picked is None:
                continue
            row = rows[key]
            # the run on screen was rejected in favour of an earlier one
            if shown_config and shown_config != picked:
                record_verdict(row, shown_config, "worse")
            record_verdict(row, picked, "pass")
            row["accepted_config"] = picked
            row["review_status"] = "pass"
            row["refit_config"] = ""
            row["refit_align_config"] = ""
            row["refit_stage_a_ratio"] = ""
            row["refit_focus_branch"] = ""
            row["refit_requested_at"] = ""
            notes = input("  notes (optional) > ").strip()
            if notes:
                row["notes"] = notes
            rows[key] = row
            save_manifest(manifest_path, rows)
            print(f"  -> accepted {picked} (falling back from {shown_config})")
            continue

        status = _STATUS_KEYS.get(raw)
        if status is None:
            print(f"  unrecognized input {raw!r}, try again")
            continue

        fit_config = align_config = None
        stage_a_ratio = focus_branch = None
        if status == "redo":
            _row = rows[key]
            _tried = tried_configs(_row)
            _untried = untried_configs(_row)
            print(f"  already tried: {format_history(_row)}")
            if _untried:
                print(f"  not yet tried: {', '.join(_untried)}")
            else:
                print("  NOTE: every config has been tried on this case.")
            _prompt_default = _untried[0] if _untried else "default"
            fit_config = input(f"  fit config ({', '.join(CONFIGS)}) "
                               f"[{_prompt_default}] > ").strip() or _prompt_default
            if fit_config not in CONFIGS:
                print(f"  unknown fit config {fit_config!r}, not recorded -- try again")
                continue
            if fit_config in _tried:
                _prev = history_dict(_row).get(fit_config, "ran/failed")
                again = input(f"  {fit_config} was already tried (verdict: {_prev}). "
                              f"Run it again anyway? [y/N] > ").strip().lower()
                if again not in ("y", "yes"):
                    print("  not recorded -- pick a different config")
                    continue
            if not CONFIGS[fit_config]["supports_align"]:
                # arap_init / clone / clone_target run their own Stage 1 and
                # expose no --w-* flags, so offering an align config here just
                # records a pairing requeue will later refuse.
                align_config = "default"
                print(f"  ({fit_config} runs its own Stage 1 -- align config not applicable)")
            else:
                align_config = input(f"  align config ({', '.join(sorted(ALIGN_CONFIGS))}) "
                                     f"[default] > ").strip() or "default"
                if align_config not in ALIGN_CONFIGS:
                    print(f"  unknown align config {align_config!r}, not recorded -- try again")
                    continue
                if align_config in ALIGN_NEEDS_BRANCH:
                    print("  which branch gets 3x endpoint weight? rank-ordered, "
                          "0 = upstream/dome-first")
                    raw_b = input("  focus branch [0] > ").strip() or "0"
                    try:
                        focus_branch = int(raw_b)
                        if focus_branch < 0:
                            raise ValueError
                    except ValueError:
                        print(f"  focus branch must be a non-negative integer, got {raw_b!r} "
                              f"-- not recorded, try again")
                        continue
            # arap_init/clone have a Stage A whose MSE:mesh-health balance is
            # the main thing worth varying per case -- ask for it.
            if CONFIGS[fit_config]["supports_stage_a_ratio"]:
                opts = ", ".join(str(r) for r in STAGE_A_RATIOS)
                _dflt = CONFIGS[fit_config]["default_stage_a_ratio"] or DEFAULT_STAGE_A_RATIO
                print("  stage-A MSE:mesh-health ratio -- LOWER = stronger mesh health "
                      "= less warping.")
                print(f"    {opts}   (lower <-- clamps shape | warps more --> higher)")
                _prev_r = _row.get("refit_stage_a_ratio") or ""
                if _prev_r:
                    print(f"    last requested for this case: r={_prev_r}")
                raw_ratio = input(f"  ratio [{_dflt}] > ").strip()
                if raw_ratio:
                    try:
                        stage_a_ratio = float(raw_ratio)
                        if stage_a_ratio <= 0:
                            raise ValueError
                    except ValueError:
                        print(f"  ratio must be a positive number, got {raw_ratio!r} -- try again")
                        continue
                else:
                    stage_a_ratio = _dflt

        notes = input("  notes (optional) > ").strip() or None

        row = rows[key]
        # the verdict applies to the config whose render was just looked at
        record_verdict(row, row.get("config_used"), status)
        if status == "pass":
            row["accepted_config"] = row.get("config_used") or ""
        row["review_status"] = status
        row["refit_config"] = fit_config if status == "redo" else ""
        row["refit_align_config"] = align_config if status == "redo" else ""
        row["refit_stage_a_ratio"] = (str(stage_a_ratio) if status == "redo" and stage_a_ratio
                                      else "")
        row["refit_focus_branch"] = (str(focus_branch)
                                     if status == "redo" and focus_branch is not None else "")
        row["refit_requested_at"] = (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                     if status == "redo" else "")
        if notes:
            row["notes"] = notes
        rows[key] = row
        save_manifest(manifest_path, rows)
        _r = f", stage_a_ratio={stage_a_ratio}" if stage_a_ratio else ""
        print(f"  -> recorded {status}"
              + (f" (fit={fit_config}, align={align_config}{_r})" if status == "redo" else ""))


# ── assemble ─────────────────────────────────────────────────────────────────
# Files worth carrying into the assembled corpus. The fit itself is
# ghd_coefficients.npz + ghd_fitted.obj; the rest is what you need to
# reproduce or audit a case without going back to runtime/.
ASSEMBLE_CORE = ("ghd_coefficients.npz", "ghd_fitted.obj", "metrics.json")
ASSEMBLE_EXTRA = ("landmarks.npz", "final_aligned.obj", "alignment_result.npy",
                  "centerline.vtp", "config_used.txt", "align_used.txt")


def accepted_config_for(row):
    """Which config's result this case was PASSED on, and where that came from.

    Returns (config_name, source) or (None, reason).

    config_history is the authority: review records `<config>=pass` there for
    the run that was actually accepted, including the [b]ack fallback where the
    accepted config is NOT the one that was on screen (config_used names the
    render you looked at, which for those rows is the REJECTED config -- 6 rows
    in the current manifest). accepted_config agrees wherever it is set but is
    only populated for rows reviewed after that column was added (207 of 493).
    """
    passes = [t.split("=", 1)[0].strip()
              for t in (row.get("config_history") or "").split(";")
              if "=" in t and t.split("=", 1)[1].strip() == "pass"]
    passes = [c for c in passes if c in CONFIGS]
    accepted = (row.get("accepted_config") or "").strip()
    if len(passes) == 1:
        return passes[0], "config_history"
    if len(passes) > 1:
        # Never seen in practice, but a case re-passed under a second config
        # would land here -- accepted_config breaks the tie, else refuse to guess.
        if accepted in passes:
            return accepted, "accepted_config (history ambiguous)"
        return None, f"config_history has {len(passes)} passes ({', '.join(passes)})"
    if accepted in CONFIGS:
        return accepted, "accepted_config"
    used = (row.get("config_used") or "").strip()
    if used in CONFIGS:
        return used, "config_used (fallback)"
    return None, "no config recorded"


def resolve_rigid_checkpoint(src, tag):
    """(coeff_path, mesh_path, tag) for a rigid checkpoint, or (None, None, why).

    Checkpoints live in <run>/rigid_checkpoints/ as MATCHED PAIRS --
    ghd_coefficients_w0.050.npz alongside ghd_fitted_w0.050.obj. Both must come
    from the same tag: pairing checkpoint coefficients with the run's FINAL mesh
    would ship a mesh that its own coefficients do not reproduce.
    """
    tag = tag if tag.startswith("w") else f"w{tag}"
    ck = src / "rigid_checkpoints"
    coeff, mesh = ck / f"ghd_coefficients_{tag}.npz", ck / f"ghd_fitted_{tag}.obj"
    if not coeff.exists():
        avail = sorted(q.stem.split("_")[-1] for q in ck.glob("ghd_coefficients_w*.npz")) \
            if ck.is_dir() else []
        return None, None, (f"no {tag} checkpoint"
                            + (f" (has {', '.join(avail)})" if avail else " (no rigid_checkpoints/)"))
    if not mesh.exists():
        return None, None, f"{tag} coefficients present but ghd_fitted_{tag}.obj missing"
    return coeff, mesh, tag


def _write_label_data(dst, dataset, case, geometry_dir):
    """Precompute the small per-case inputs dataset/label_morpho.py needs, so
    the assembled corpus is self-contained for labelling on another machine.

    Best effort: a case whose geometry is unavailable simply gets no label
    data, and the labeller falls through to manual brushing for it.
    """
    try:
        # manage.py runs with scripts/fit on sys.path (for fit_configs), not the
        # repo root, so the dataset/ and ghd/ packages are not importable here
        # without this.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import trimesh
        from dataset.dome_label import (stage1_transform, world_from_aligned,
                                        dome_mask_for_case, DEFAULT_GAP_TOL)
        from dataset.preprocess_assembled import outward_branches
        from ghd.fitting.alignment import resolve_case_paths

        mesh = trimesh.load(dst / "ghd_fitted.obj", process=False)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        tf = stage1_transform(dst)
        if tf is None or not geometry_dir.is_dir():
            return
        R, sc, t = tf
        paths = resolve_case_paths(geometry_dir)

        dome = dome_mask_for_case(dataset, geometry_dir, world_from_aligned(V, R, sc, t),
                                  full_mesh_path=Path(paths["clipped_mesh"]))
        if dome is not None:
            np.save(dst / "dome_mask.npy",
                    {"mask": np.asarray(dome, dtype=bool), "gap_tol": DEFAULT_GAP_TOL,
                     "dataset": dataset}, allow_pickle=True)

        clipped = np.load(Path(paths["clipped_centerline"]), allow_pickle=True).item()
        ranking = np.load(Path(paths["branch_ranking"]))
        br = outward_branches(clipped, ranking, R, sc, t)
        np.save(dst / "branch_points.npy",
                np.array([np.asarray(b, dtype=np.float32) for b in br], dtype=object),
                allow_pickle=True)
    except Exception as exc:
        print(f"  (label data unavailable for {dataset}/{case}: {exc})")


def cmd_assemble(args):
    """Copy each passed case's ACCEPTED result into one flat corpus."""
    rows = load_manifest(Path(args.manifest))
    runtime_root = Path(args.runtime_root).resolve()
    out_root = Path(args.out_root).resolve()
    wanted = set(args.status)

    items = [r for r in rows.values() if (r.get("review_status") or "").strip() in wanted]
    if args.dataset:
        items = [r for r in items if r["dataset"] == args.dataset]
    items.sort(key=lambda r: (r["dataset"], r["case_name"]))

    files = list(ASSEMBLE_CORE) + ([] if args.core_only else list(ASSEMBLE_EXTRA))
    index, skipped = [], []
    n_copied = 0

    for r in items:
        ds, case = r["dataset"], r["case_name"]
        cfg, why = accepted_config_for(r)
        if cfg is None:
            skipped.append((ds, case, why))
            continue
        # align_used describes the run that was ON SCREEN at review time. That
        # is the right pairing only when the accepted config IS the shown one.
        # Via the [b]ack fallback they differ -- and pairing e.g. accepted
        # arap_init with a shown endpoint_focus_branch asks for
        # arap_init_endpoint_focus_branch/, which cannot exist (arap_init runs
        # its own Stage 1 and takes no align config). For that case, find the
        # accepted config's own newest run across align variants instead.
        align = (r.get("align_used") or "default").strip() or "default"
        if cfg != (r.get("config_used") or "").strip():
            _mark, _d, _m, align = _config_result_newest(runtime_root, cfg, ds, case)
        src = cfg_output_dir(cfg, ds, case, runtime_root, align)
        missing = [f for f in ASSEMBLE_CORE if not (src / f).exists()]
        if missing:
            skipped.append((ds, case, f"{cfg}: missing {', '.join(missing)} in {src}"))
            continue

        # Optionally take the fit from a rigid checkpoint rather than the final
        # iteration -- a higher rigid weight is less warped, so more stable.
        ck_coeff = ck_mesh = None
        ck_tag = ""
        if args.rigid_checkpoint:
            ck_coeff, ck_mesh, info = resolve_rigid_checkpoint(src, args.rigid_checkpoint)
            if ck_coeff is None:
                skipped.append((ds, case, f"{cfg}: {info}"))
                continue
            ck_tag = info

        dst = out_root / ds / case
        if dst.exists() and not args.overwrite:
            skipped.append((ds, case, "already assembled (use --overwrite)"))
            continue

        if not args.dry_run:
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            _skip = {"ghd_coefficients.npz", "ghd_fitted.obj"} if ck_coeff else set()
            for f in files:
                if f not in _skip and (src / f).exists():
                    shutil.copy2(src / f, dst / f)
            if ck_coeff:
                # land under the CANONICAL names so downstream code needs no
                # special case; assembled_from.json records which tag it is
                shutil.copy2(ck_coeff, dst / "ghd_coefficients.npz")
                shutil.copy2(ck_mesh, dst / "ghd_fitted.obj")
                if args.keep_final:
                    shutil.copy2(src / "ghd_coefficients.npz", dst / "ghd_coefficients_final.npz")
                    shutil.copy2(src / "ghd_fitted.obj", dst / "ghd_fitted_final.obj")
            # Labelling happens on a LOCAL machine from the assembled corpus
            # alone, so everything dataset/label_morpho.py needs has to live
            # here. The two things it would otherwise reach back to the
            # geometry directory for are precomputed now:
            #   dome_mask.npy     the automatic dome proposal (label.nrrd is
            #                     39 MB/case -- 9.4 GB across ImperialNHS --
            #                     so shipping the source is not an option)
            #   branch_points.npy the OUTWARD centerline in aligned space,
            #                     which auto_uncap needs to place its cut
            #                     planes (centerline.vtp holds only the short
            #                     Stage-1 alignment segments)
            # Both are a few KB. The dome tolerance is baked in, so it is
            # recorded alongside for traceability.
            if not args.no_label_data:
                _write_label_data(dst, ds, case, Path(r.get("geometry_dir", "")))

            # provenance: which run this came from, so the corpus is auditable
            (dst / "assembled_from.json").write_text(json.dumps({
                "dataset": ds, "case_name": case, "config": cfg,
                "align_config": align, "config_source": why,
                "source_dir": str(src), "geometry_dir": r.get("geometry_dir", ""),
                "rigid_checkpoint": ck_tag or "final",
                "chamfer_best": r.get("chamfer_best", ""),
                "aneurysm_type": r.get("aneurysm_type", ""),
                "review_status": r.get("review_status", ""),
                "assembled_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2))
        n_copied += 1
        index.append({"dataset": ds, "case_name": case, "config": cfg,
                      "align_config": align, "config_source": why,
                      "rigid_checkpoint": ck_tag or "final",
                      "aneurysm_type": r.get("aneurysm_type", ""),
                      "chamfer_best": r.get("chamfer_best", ""),
                      "review_status": r.get("review_status", ""),
                      "source_dir": str(src), "geometry_dir": r.get("geometry_dir", "")})

    if not args.dry_run and index:
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / "index.csv", "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
            wtr.writeheader()
            wtr.writerows(index)

    tag = "WOULD assemble" if args.dry_run else "assembled"
    _src = f" [from rigid checkpoint {args.rigid_checkpoint}]" if args.rigid_checkpoint else ""
    print(f"{tag} {n_copied} case(s){_src} -> {out_root}")
    by_ds, by_cfg, by_src = {}, {}, {}
    for e in index:
        by_ds[e["dataset"]] = by_ds.get(e["dataset"], 0) + 1
        by_cfg[e["config"]] = by_cfg.get(e["config"], 0) + 1
        by_src[e["config_source"]] = by_src.get(e["config_source"], 0) + 1
    if by_ds:
        print("  by dataset :", ", ".join(f"{k}={v}" for k, v in sorted(by_ds.items())))
        print("  by config  :", ", ".join(f"{k}={v}" for k, v in sorted(by_cfg.items())))
        print("  chosen via :", ", ".join(f"{k}={v}" for k, v in sorted(by_src.items())))
    if skipped:
        print(f"\n  SKIPPED {len(skipped)}:")
        for ds, case, why in skipped[:args.max_skipped]:
            print(f"    {ds}/{case:<40} {why}")
        if len(skipped) > args.max_skipped:
            print(f"    ... and {len(skipped) - args.max_skipped} more")
    if not args.dry_run and index:
        print(f"\n  index: {out_root / 'index.csv'}")


def cmd_review_list(args):
    rows = load_manifest(Path(args.manifest))
    items = list(rows.values())
    if args.dataset:
        items = [r for r in items if r["dataset"] == args.dataset]
    items = [r for r in items if r.get("status") != "running"]
    failed, done_unreviewed = _review_queue(items)

    print(f"=== needs review: {len(failed)} failed, {len(done_unreviewed)} done-but-unreviewed (showing up to {args.limit} each) ===")
    for r in failed[:args.limit]:
        img = _raw_mesh_sanity_for(r) if args.render_failed else None
        print(f"FAILED\t{r['dataset']}\t{r['case_name']}\t{img or '(no raw mesh found under geometry_dir)'}")
        print(f"      tried: {format_history(r)}")
    for r in done_unreviewed[:args.limit]:
        sanity = Path(r["save_dir"]) / "sanity" / "sanity_final.png"
        print(f"DONE\t{r['dataset']}\t{r['case_name']}\tconfig={r['config_used']}\t"
              f"chamfer_best={r['chamfer_best']}\t{sanity}")
        _runs = case_runs(r)
        if len(_runs) > 1:
            print(format_runs(_runs, current=r["config_used"], current_align=r.get("align_used")))
        print(f"      tried: {format_history(r)}")


def cmd_requeue(args):
    rows = load_manifest(Path(args.manifest))
    pending = [r for r in rows.values() if r["refit_config"] or r["refit_align_config"]]
    if args.dataset:
        pending = [r for r in pending if r["dataset"] == args.dataset]

    # A row's refit_config is normally cleared for any verdict other than
    # "redo", so a fix/reclip case cannot reach requeue by the usual route.
    # This is a guard for the abnormal one: a manifest hand-edited outside
    # manage.py (it is a plain CSV, and gets opened in a spreadsheet) could
    # leave both set, and refitting a case whose verdict says it needs manual
    # work first just burns GPU on a known-bad input.
    if args.only_fixed:
        pending = [r for r in pending if (r.get("fixed") or "").strip()]
        print(f"--only-fixed: restricted to {len(pending)} hand-fixed case(s)")

    if args.skip_status:
        blocked = [r for r in pending if r["review_status"] in set(args.skip_status)]
        if blocked:
            pending = [r for r in pending if r["review_status"] not in set(args.skip_status)]
            print(f"Skipping {len(blocked)} case(s) whose review_status is in "
                  f"{sorted(set(args.skip_status))} (pass --skip-status with no values "
                  f"to requeue them anyway):")
            for r in blocked[:10]:
                print(f"  {r['dataset']}/{r['case_name']}  status={r['review_status']}  "
                      f"refit_config={r['refit_config']!r}")
            if len(blocked) > 10:
                print(f"  ... and {len(blocked) - 10} more")
    if not pending:
        print("No manifest rows have refit_config/refit_align_config set -- nothing to requeue.")
        return

    unknown_fit = sorted({r["refit_config"] for r in pending
                          if r["refit_config"] and r["refit_config"] not in CONFIGS})
    unknown_align = sorted({r["refit_align_config"] for r in pending
                            if r["refit_align_config"] and r["refit_align_config"] not in ALIGN_CONFIGS})
    if unknown_fit or unknown_align:
        print(f"ERROR: unknown config name(s) in manifest -- fit: {unknown_fit} "
              f"(known: {sorted(CONFIGS)}), align: {unknown_align} (known: {sorted(ALIGN_CONFIGS)})")
        sys.exit(1)

    # arap_init/clone run their own Stage 1 and expose no --w-* flags, so a
    # non-default align config paired with them would be silently ignored.
    bad_align = [(r["case_name"], r["refit_config"], r["refit_align_config"]) for r in pending
                 if r["refit_config"] in CONFIGS
                 and not CONFIGS[r["refit_config"]]["supports_align"]
                 and r["refit_align_config"] not in ("", "default")]
    if bad_align:
        print("ERROR: these configs run their own Stage 1 and cannot take an align config:")
        for case, fc, ac in bad_align:
            print(f"  {case}: fit={fc} align={ac}")
        sys.exit(1)

    # clone needs the old project's fitted result to exist for each case.
    # Use old_case_dir(), not a raw path join: the old project spelled AneuX as
    # "AnueX", so OLD_FIT_ROOT/<dataset> misses every AneuX case. It also checks
    # landmarks.npz and metrics.json, which clone_fit.py needs as well -- this
    # guard used to look only for ghd_fitted.obj.
    missing_old = [r["case_name"] for r in pending
                   if r["refit_config"] in CONFIGS and CONFIGS[r["refit_config"]]["needs_old_case"]
                   and old_case_dir(r["dataset"], r["case_name"]) is None]
    if missing_old:
        print(f"ERROR: 'clone' requested but no old fitted mesh under {OLD_FIT_ROOT} for: "
              f"{missing_old}")
        sys.exit(1)

    # "cuda:2" -> 1 worker; "cuda:2:6" -> 6 concurrent workers on that card.
    # Same syntax run_all.py takes. Each worker is its own tmux session with
    # its own serial queue, so N workers means N jobs running there at once.
    slots = []                                   # [(device, session_suffix), ...]
    for spec in (args.gpus or ["cuda:0", "cuda:1", "cuda:2"]):
        bits = spec.split(":")
        if len(bits) == 3:
            device, n = f"{bits[0]}:{bits[1]}", int(bits[2])
        else:
            device, n = spec, 1
        for k in range(n):
            slots.append((device, f"{device.replace(':', '_')}_w{k}" if n > 1
                          else device.replace(":", "_")))
    buckets = [[] for _ in slots]
    for i, r in enumerate(pending):
        buckets[i % len(slots)].append(r)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_launched = 0
    for g, (gpu, suffix) in enumerate(slots):
        queue = buckets[g]
        if not queue:
            continue
        session = f"ghd_requeue_{suffix}"
        subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)

        # build_fit_command emits RELATIVE script paths (ghd/fitting/run_case.py
        # etc.), and a tmux session inherits whatever cwd manage.py was launched
        # from -- scripts/fit/ when run from there, which resolved every path one
        # level too deep and failed all 200 jobs instantly. Anchor to the repo
        # root so the worker runs correctly regardless of where it was launched.
        # Cap BLAS/OpenMP threads per job. Each fit is CPU-bound for much of its
        # life (Stage 1 does mesh fusion / planarize / capping on CPU), and
        # unbounded numpy/torch threading had single jobs taking 2-3 cores each:
        # 10 workers drove load to 134 on a 36-core box, ~4x oversubscribed, and
        # GPU utilisation FELL because jobs sat waiting for cores. Capping trades
        # per-job speed for far less thrash, which is the better deal when the
        # work is many independent cases rather than one big one.
        lines = [CONDA_INIT, f"cd '{ROOT}'",
                 f"export OMP_NUM_THREADS={args.threads_per_job}",
                 f"export MKL_NUM_THREADS={args.threads_per_job}",
                 f"export OPENBLAS_NUM_THREADS={args.threads_per_job}",
                 f"export NUMEXPR_NUM_THREADS={args.threads_per_job}"]
        for r in queue:
            fit_cfg = r["refit_config"] or "default"
            align_cfg = r["refit_align_config"] or "default"
            cfg = CONFIGS[fit_cfg]
            # Same builder run_all.py uses, so a redo lands in exactly the
            # layout scan() looks in -- runtime/<config>/<dataset>/<case>/ --
            # and never overwrites the case's result under a different config.
            cmd, out_path = build_fit_command(
                fit_cfg, r["dataset"], r["case_name"], r["geometry_dir"],
                str(FITTING_ROOT), args.python, gpu, args.n_iter,
                align_flags=align_flags(align_cfg, r.get("refit_focus_branch")),
                stage_a_ratio=r.get("refit_stage_a_ratio"),
                align_config=align_cfg)
            if args.rigid_warmup_iters and CONFIGS[fit_cfg]["script"].endswith("run_case.py"):
                cmd += f" --rigid-warmup-iters {args.rigid_warmup_iters}"
            out_dir = str(out_path)
            _ratio_note = (f", r={r['refit_stage_a_ratio']}"
                           if cfg["supports_stage_a_ratio"] and r.get("refit_stage_a_ratio") else "")
            guarded = not args.overwrite
            if guarded:
                lines.append(f"if [ -f '{out_dir}/metrics.json' ]; then")
                lines.append(f"  echo '[{gpu}] SKIP {r['case_name']} ({fit_cfg}) -- already done'")
                lines.append("else")
            # Re-checked at RUN time, not just when the plan was built: a worker
            # runs for hours, and another worker (or a second requeue launched
            # to use idle GPUs) may finish this case meanwhile. run_all.py has
            # the same guard; requeue lacked it, so overlapping launches would
            # redo each other's work.
            lines.append(f"echo '[{gpu}] === REDO {r['case_name']} (align={align_cfg}, fit={fit_cfg}{_ratio_note}) -> {out_dir} ==='")
            lines.append(f"if {cmd}; then")
            lines.append(f"  echo '{fit_cfg}' > \"{out_dir}/config_used.txt\"")
            lines.append(f"  echo '{align_cfg}' > \"{out_dir}/align_used.txt\"")
            lines.append(f"  echo '[{gpu}] REDO OK: {r['case_name']}'")
            lines.append("else")
            lines.append(f"  echo '[{gpu}] REDO FAILED: {r['case_name']}'")
            lines.append("fi")
            if guarded:
                lines.append("fi")
        lines.append(f"echo ALL_DONE_{session}")
        worker_script = log_dir / f"{session}.sh"
        worker_script.write_text("\n".join(lines) + "\n")

        print(f"Session {session} on {gpu}: {len(queue)} redo job(s) queued")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, f"bash '{worker_script}'"])
        subprocess.run(["tmux", "pipe-pane", "-t", session, "-o", f"cat >> '{log_dir / (session + '.log')}'"])
        n_launched += len(queue)

    print(f"Launched {n_launched} redo job(s) across {sum(1 for b in buckets if b)} worker(s) "
          f"on {len({d for d, _ in slots})} GPU(s).")
    print("Run `scan` again once they finish -- fulfilled redo rows auto-clear refit_config.")


def cmd_show(args):
    """Open one case's sanity render -- for a chosen config, or list what exists."""
    rows = load_manifest(Path(args.manifest))
    key = (args.dataset, args.case)
    if key not in rows:
        print(f"no manifest row for {args.dataset}/{args.case}")
        sys.exit(1)
    r = rows[key]
    runs = case_runs(r)
    if not runs:
        print(f"{args.dataset}/{args.case}: no runs on disk yet")
        return
    print(f"{args.dataset}/{args.case} -- {len(runs)} run(s):")
    print(format_runs(runs, current=args.config or r["config_used"], current_align=r.get("align_used")))
    want = args.config or r["config_used"] or runs[0][0]
    match = next((t for t in runs if t[0] == want), None)
    if match is None:
        print(f"\nno run for config {want!r}; available: {[t[0] for t in runs]}")
        sys.exit(1)
    img = render_for(match[0], match[2])
    if img:
        print(f"\nopening {want}: {img}")
        _pop_image(str(img))
    else:
        print(f"\n{want} has no sanity/sanity_final.png (dir: {match[2]})")


def cmd_history(args):
    """What has been tried on each case, and what is left to try."""
    rows = load_manifest(Path(args.manifest))
    items = [r for r in rows.values()
             if (not args.dataset or r["dataset"] == args.dataset)
             and (not args.case or r["case_name"] == args.case)]
    if not items:
        print("no matching manifest rows")
        return
    if args.exhausted:
        items = [r for r in items if not untried_configs(r)]
    elif args.tried_any:
        items = [r for r in items if tried_configs(r)]
    items.sort(key=lambda r: (r["dataset"], r["case_name"]))

    width = max(len(r["case_name"]) for r in items)
    print(f"{'case':<{width}}  {'status':<8}  tried -> untried")
    print("-" * (width + 60))
    for r in items:
        un = untried_configs(r)
        print(f"{r['case_name']:<{width}}  {r['status'] or '-':<8}  {format_history(r)}")
        if args.runs:
            runs = case_runs(r)
            if runs:
                print(format_runs(runs, current=r["config_used"], current_align=r.get("align_used")))
        print(f"{'':<{width}}  {'':<8}  untried: {', '.join(un) if un else '(none -- all configs exhausted)'}")
    print(f"\n{len(items)} case(s). "
          f"{sum(1 for r in items if not untried_configs(r))} have exhausted every config.")


def cmd_configs(args):
    """Print every config with its fully resolved flags -- what `requeue`
    would actually run, without running anything."""
    for name, cfg in CONFIGS.items():
        dflt = cfg["default_stage_a_ratio"]
        print(f"\n{name}")
        print(f"  script        {cfg['script']}")
        print(f"  output flag   {cfg['out_flag']}")
        bits = []
        if cfg["supports_align"]:
            bits.append("align configs OK")
        else:
            bits.append("no align config (runs its own Stage 1)")
        if cfg["needs_old_case"]:
            bits.append("needs --old-case-dir")
        if not cfg["supports_eta_min"]:
            bits.append("no --eta-min")
        print(f"  notes         {'; '.join(bits)}")
        if cfg["supports_stage_a_ratio"]:
            print(f"  stage-A ratio default r={dflt}  (selectable: {STAGE_A_RATIOS})")
            for r in ([dflt] if args.ratio is None else [args.ratio]):
                print(f"  flags @r={r}   {resolve_flags(name, r)}")
        else:
            print("  stage-A ratio n/a (no Stage A)")
            print(f"  flags         {resolve_flags(name) or '(none -- script defaults)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan a dataset's output dir, upsert manifest rows.")
    p_scan.add_argument("--dataset", required=True)
    p_scan.add_argument("--runtime-root", default=str(FITTING_ROOT),
                        help="Root holding runtime/<config>/<dataset>/<case>/ (default: runtime/).")
    p_scan.add_argument("--dataset-root", default=None,
                        help="Legacy single output dir to also fold in, e.g. runtime/ImperialNHS.")
    p_scan.add_argument("--include-unrun", action="store_true",
                        help="Also create rows for geometry cases nothing has fitted yet.")
    p_scan.add_argument("--geometry-root", default=None, help="Source case geometry dir (needed once, for requeue).")
    p_scan.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_review = sub.add_parser("review", help="Interactive loop: pop image, prompt for verdict, repeat.")
    p_review.add_argument("--dataset", default=None)
    p_review.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_next = sub.add_parser("next", help="Pop the single next case needing review.")
    p_next.add_argument("--dataset", default=None)
    p_next.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_next.add_argument("--no-pop", action="store_true", help="Don't auto-open the image in VSCode.")

    p_update = sub.add_parser("update", help="Record one case's review verdict.")
    p_update.add_argument("--dataset", required=True)
    p_update.add_argument("--case", required=True)
    p_update.add_argument("--status", required=False, default=None,
                          choices=["pass", "skip", "redo", "reclip", "fix"],
                          help="pass=good as-is, skip=give up on it, redo=refit with --config, "
                               "reclip=needs AneuSeg/reclip.py run manually before refitting, "
                               "fix=needs some other manual intervention before refitting.")
    p_update.add_argument("--config", default=None,
                          help="Fit config preset name (from fit_configs.py). Defaults to "
                               "'default' if --status redo and only --align-config is given.")
    p_update.add_argument("--accept-config", default=None,
                          help="Accept THIS config's existing run as the case's answer, even if a "
                               "later run exists -- the fallback case: you tried something else, "
                               "it came out worse, and you want the earlier result. Sets "
                               "review_status=pass and records accepted_config.")
    p_update.add_argument("--force", action="store_true",
                          help="Allow re-requesting a config that was already tried on this case.")
    p_update.add_argument("--stage-a-ratio", type=float, default=None,
                          help=f"Stage-A MSE:mesh-health ratio, for the configs with a Stage A "
                               f"(arap_init, clone, clone_no_opening). rigid = 1/ratio, so LOWER "
                               f"= stronger mesh health = LESS warping: if a case warps too much "
                               f"at r5, go DOWN to r2/r1/r0.5, not up. Options: {STAGE_A_RATIOS}. "
                               f"Defaults to each config's own (clone 5, arap_init 20).")
    p_update.add_argument("--align-config", default=None,
                          help="Align (Stage 1) config preset name (from align_configs.py). "
                               "Defaults to 'default' if --status redo and only --config is given.")
    p_update.add_argument("--notes", default=None)
    p_update.add_argument("--focus-branch", type=int, default=None,
                          help="Rank-ordered branch index (0 = upstream/dome-first) whose "
                               "ENDPOINT term gets 3x the weight of the others. Required by "
                               "align configs that single out one branch "
                               "(endpoint_focus_branch); rejected for the rest.")
    p_update.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_list = sub.add_parser("review-list", help="Print cases needing review, worst first.")
    p_list.add_argument("--dataset", default=None)
    p_list.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_list.add_argument("--limit", type=int, default=8)
    p_list.add_argument("--no-render-failed", dest="render_failed", action="store_false",
                         help="Skip raw-mesh rendering for failed cases (default: render).")

    p_show = sub.add_parser("show", help="Open one case's render for a given config.")
    p_show.add_argument("--dataset", required=True)
    p_show.add_argument("--case", required=True)
    p_show.add_argument("--config", default=None,
                        help="Which config's render to open. Defaults to the row's config_used.")
    p_show.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_hist = sub.add_parser("history", help="Show which configs each case has already tried.")
    p_hist.add_argument("--dataset", default=None)
    p_hist.add_argument("--case", default=None)
    p_hist.add_argument("--tried-any", action="store_true",
                        help="Only cases that have tried at least one config.")
    p_hist.add_argument("--runs", action="store_true",
                        help="Also list every run on disk per case, with its chamfer, its "
                             "key Stage-A settings and the path to its sanity render.")
    p_hist.add_argument("--exhausted", action="store_true",
                        help="Only cases where every config has been tried.")
    p_hist.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_cfg = sub.add_parser("configs", help="Print each fit config and its resolved flags.")
    p_cfg.add_argument("--ratio", type=float, default=None,
                       help="Show flags at this Stage-A ratio instead of each config's default.")

    p_req = sub.add_parser("requeue", help="Launch tmux redo jobs for every refit_config-marked row.")
    p_req.add_argument("--dataset", default=None)
    p_req.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_req.add_argument("--gpus", nargs="+", default=None,
                       help="Device, optionally with a worker count: 'cuda:2:6' runs 6 concurrent "
                            "jobs on cuda:2. Jobs need ~2 GB each, so utilisation rather than "
                            "memory is usually the limit.")
    p_req.add_argument("--skip-status", nargs="*", default=["fix", "reclip"],
                       help="Do not requeue cases whose review_status is one of these -- they "
                            "were judged to need manual work BEFORE refitting, so refitting "
                            "them now just repeats a known-bad input. Default: fix reclip. "
                            "Pass with no values to disable the filter.")
    p_req.add_argument("--rigid-warmup-iters", type=int, default=0,
                       help="OVERRIDE the warm-up length: high rigid weight for this many "
                            "iterations first (5 -> lambda_rigid_start), checkpointed at 5/4/3, "
                            "on top of --n-iter rather than inside it. The 'default' config "
                            "already carries 1000, so leave this alone unless you want a "
                            "different length; 0 means 'do not override'. Passing it here "
                            "appends after the config's own flags, and argparse takes the last "
                            "occurrence. Only configs running run_case.py accept it.")
    p_req.add_argument("--only-fixed", action="store_true",
                       help="Only cases whose 'fixed' column is ticked, i.e. the ones you have "
                            "hand-repaired.")
    p_req.add_argument("--overwrite", action="store_true",
                       help="Re-run even cases that already have a completed fit. Without this a "
                            "worker skips any case whose metrics.json exists.")
    p_req.add_argument("--threads-per-job", type=int, default=2,
                       help="Cap BLAS/OpenMP threads per fitting job. Unbounded threading let "
                            "single jobs take 2-3 cores and drove load to ~4x the core count, "
                            "which slowed everything down. 0 disables the cap.")
    p_req.add_argument("--n-iter", type=int, default=10000)
    p_asm = sub.add_parser("assemble", help="Copy each passed case's ACCEPTED result "
                                            "into one flat corpus under --out-root.")
    p_asm.add_argument("--dataset", default=None, help="Limit to one dataset (default: all).")
    p_asm.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_asm.add_argument("--runtime-root", default=str(FITTING_ROOT))
    # The prepared dataset lives OUTSIDE runtime/, which is scratch that gets
    # rewritten and pruned by scan/requeue. runtime_dataset/ is the durable
    # side: assembled corpus in, processed training set out.
    p_asm.add_argument("--out-root", default=str(ROOT / "runtime_dataset" / "assembled"))
    p_asm.add_argument("--status", nargs="+", default=["pass"],
                       help="Which review_status values to assemble. Default: pass.")
    p_asm.add_argument("--rigid-checkpoint", default=None, metavar="TAG",
                       help="Take the fit from this rigid checkpoint instead of the final "
                            "iteration, e.g. 'w0.050' (or '0.050'). A higher rigid weight is "
                            "less warped, so more stable. The checkpoint's coefficients AND its "
                            "matching mesh are both taken, and land under the canonical names "
                            "ghd_coefficients.npz / ghd_fitted.obj; assembled_from.json and "
                            "index.csv record which tag it was. Cases lacking the tag are "
                            "skipped and listed.")
    p_asm.add_argument("--keep-final", action="store_true",
                       help="With --rigid-checkpoint, also copy the final-iteration fit as "
                            "ghd_coefficients_final.npz / ghd_fitted_final.obj.")
    p_asm.add_argument("--core-only", action="store_true",
                       help="Copy only the fit itself (coefficients, mesh, metrics) "
                            "rather than the landmarks/alignment extras too.")
    p_asm.add_argument("--overwrite", action="store_true",
                       help="Replace a case already present under --out-root.")
    p_asm.add_argument("--no-label-data", action="store_true",
                       help="Skip precomputing dome_mask.npy / branch_points.npy. Without "
                            "them the assembled corpus is not self-contained for "
                            "dataset/label_morpho.py on a machine that lacks the geometry.")
    p_asm.add_argument("--dry-run", action="store_true",
                       help="Report what would be copied, write nothing.")
    p_asm.add_argument("--max-skipped", type=int, default=20)

    p_req.add_argument("--python", default=DEFAULT_PYTHON)
    p_req.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))

    args = parser.parse_args()
    {
        "scan": cmd_scan, "review": cmd_review, "next": cmd_next, "update": cmd_update,
        "review-list": cmd_review_list, "requeue": cmd_requeue,
        "configs": cmd_configs, "history": cmd_history, "show": cmd_show,
        "assemble": cmd_assemble,
    }[args.command](args)


if __name__ == "__main__":
    main()
