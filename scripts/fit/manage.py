"""manage.py -- single entry point for tracking + re-running GHD-fitting
results across (potentially several) datasets.

Keeps one CSV manifest (default: runtime/manifest.csv, at the repo root --
deliberately NOT inside any one dataset's runtime/<Dataset>/ folder, so it
can carry rows for multiple datasets as more get processed) with one row
per case: where it lives, whether it finished/failed, its best chamfer, and
reviewer-owned columns (review_status, refit_config, refit_align_config)
that this script never overwrites once set.

Two independent config axes, each with its own named-preset registry:
  - fit config    (fit_configs.py's CONFIGS)     -- ghd_fit.py's --lambda-* flags.
  - align config  (align_configs.py's ALIGN_CONFIGS) -- alignment.py's --w-* flags.
A redo can set either or both; requeue combines whichever are set into one
run_case.py invocation and saves to a combo-specific folder (see
_combo_dir_name), e.g. runtime/ImperialNHS_endpoint_focus_with_opening.

Five subcommands:

  scan          Walk a dataset's runtime/<Dataset>/ output dir (+ its
                _failed_worker_*.log files), upsert rows into the manifest.
                Safe to re-run any time (e.g. while a batch is still going).

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
                cases (only those) in tmux workers using the named
                combination from fit_configs.py + align_configs.py. Each
                non-default combo writes to its OWN save root --
                runtime/<dataset>_<align_config>_<fit_config> (either half
                omitted if "default") -- never overwriting the case's
                existing fit under a different combo. Once they finish,
                `scan` that new root as its own dataset name (e.g. --dataset
                ImperialNHS_with_opening --dataset-root
                runtime/ImperialNHS_with_opening) to track it going forward.

Usage:
  python scripts/fit/manage.py scan --dataset ImperialNHS \
      --dataset-root "runtime/ImperialNHS" \
      --geometry-root "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"

  python scripts/fit/manage.py next --dataset ImperialNHS
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status pass
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status redo --config with_opening
  python scripts/fit/manage.py update --dataset ImperialNHS --case <name> --status redo --align-config endpoint_focus --config with_geo_dist

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
from fit_configs import CONFIGS  # noqa: E402
from align_configs import ALIGN_CONFIGS  # noqa: E402

DEFAULT_MANIFEST = ROOT / "runtime" / "manifest.csv"
DEFAULT_PYTHON = "/home/yaplab2/miniconda3/envs/new/bin/python3"
DEFAULT_LOG_DIR = ROOT / "runtime" / "_logs"
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

FIELDNAMES = [
    "dataset", "case_name", "geometry_dir", "aneurysm_type", "save_dir", "status",
    "config_used", "align_used", "chamfer_best", "chamfer_final", "best_iter",
    "last_run", "review_status", "refit_config", "refit_align_config", "notes",
]


def _combo_dir_name(dataset, align_cfg, fit_cfg):
    """Folder name for one (align config, fit config) combination -- e.g.
    ImperialNHS_endpoint_focus_with_opening. "default" on either axis is
    omitted from the name, so a fit-only or align-only redo keeps the
    simpler existing naming (e.g. ImperialNHS_with_opening)."""
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


def _read_aneurysm_type(geometry_dir):
    """Reads geometry_dir/endpoints_manual.npy's aneurysm_type (0=bifurcated,
    1/2=sidewall) -- returns "" if unavailable rather than raising, since
    scan must keep going even for a malformed/missing case."""
    if not geometry_dir:
        return ""
    ep_path = Path(geometry_dir) / "endpoints_manual.npy"
    if not ep_path.exists():
        return ""
    try:
        ep = np.load(ep_path, allow_pickle=True).item()
        return str(int(ep["aneurysm_type"]))
    except Exception:
        return ""


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
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    dataset_root = Path(args.dataset_root).resolve()
    geometry_root = Path(args.geometry_root).resolve() if args.geometry_root else None

    failed_names = set()
    for log_path in sorted(dataset_root.glob("_failed_worker_*.log")):
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if line:
                failed_names.add(line)

    case_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    n_done = n_failed = n_pending = 0
    for case_dir in case_dirs:
        case_name = case_dir.name
        key = (args.dataset, case_name)
        row = rows.get(key) or {k: "" for k in FIELDNAMES}
        row["dataset"] = args.dataset
        row["case_name"] = case_name
        row["save_dir"] = str(case_dir)
        if geometry_root:
            row["geometry_dir"] = str(geometry_root / case_name)
        if not row.get("aneurysm_type"):
            row["aneurysm_type"] = _read_aneurysm_type(row["geometry_dir"])

        coeffs = case_dir / "ghd_coefficients.npz"
        fitted = case_dir / "ghd_fitted.obj"
        metrics_path = case_dir / "metrics.json"
        if coeffs.exists() and fitted.exists():
            row["status"] = "done"
            n_done += 1
            config_marker = case_dir / "config_used.txt"
            row["config_used"] = config_marker.read_text().strip() if config_marker.exists() else (row["config_used"] or "default")
            align_marker = case_dir / "align_used.txt"
            row["align_used"] = align_marker.read_text().strip() if align_marker.exists() else (row["align_used"] or "default")
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                if isinstance(m.get("chamfer_best"), (int, float)):
                    row["chamfer_best"] = f"{m['chamfer_best']:.6f}"
                if isinstance(m.get("chamfer_final"), (int, float)):
                    row["chamfer_final"] = f"{m['chamfer_final']:.6f}"
                if m.get("best_iter") is not None:
                    row["best_iter"] = str(m["best_iter"])
            row["last_run"] = datetime.datetime.fromtimestamp(coeffs.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

            # A previously-requested redo (on either axis) that now shows the
            # requested config(s) is fulfilled: clear the request and flag it
            # for a fresh look. An axis that was never requested (blank) is
            # trivially satisfied -- only the axes actually asked for matter.
            had_request = bool(row["refit_config"]) or bool(row["refit_align_config"])
            if had_request:
                want_fit = row["refit_config"] or "default"
                want_align = row["refit_align_config"] or "default"
                if row["config_used"] == want_fit and row["align_used"] == want_align:
                    row["refit_config"] = ""
                    row["refit_align_config"] = ""
                    row["review_status"] = ""
        elif case_name in failed_names:
            row["status"] = "failed"
            n_failed += 1
            row["last_run"] = datetime.datetime.fromtimestamp(case_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            row["status"] = "pending"
            n_pending += 1

        if not row["review_status"] and _unsupported_aneurysm_type(row["geometry_dir"]):
            row["review_status"] = "skip"
            if not row["notes"]:
                row["notes"] = "auto-skipped: unsupported aneurysm_type > 2"

        rows[key] = row

    save_manifest(manifest_path, rows)
    print(f"[{args.dataset}] scanned {len(case_dirs)} case dir(s): {n_done} done, {n_failed} failed, {n_pending} pending.")
    print(f"Manifest: {manifest_path} ({len(rows)} row(s) total, all datasets).")


def render_raw_mesh_sanity(mesh_path, out_path, n_angles=4):
    """Quick multi-angle screenshot of a raw .ply -- used for failed cases,
    which never got far enough to produce a fitted sanity_final.png. No
    canonical overlay, just the case's own clipped/unclipped surface, so the
    reviewer can see e.g. "oh, this one was never clipped" at a glance."""
    import pyvista as pv
    import trimesh

    if pv.system_supports_plotting() is False or not os.environ.get("DISPLAY"):
        pv.start_xvfb()

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
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    key = (args.dataset, args.case)
    if key not in rows:
        print(f"ERROR: no manifest row for dataset={args.dataset!r} case={args.case!r}")
        sys.exit(1)
    if args.status == "redo":
        fit_cfg = args.config or "default"
        align_cfg = args.align_config or "default"
        if fit_cfg == "default" and align_cfg == "default":
            print("ERROR: --status redo requires --config and/or --align-config "
                  "(at least one must differ from default, otherwise there's "
                  "nothing new to run).")
            sys.exit(1)
        if fit_cfg not in CONFIGS:
            print(f"ERROR: unknown fit config {fit_cfg!r}. Known configs: {sorted(CONFIGS)}")
            sys.exit(1)
        if align_cfg not in ALIGN_CONFIGS:
            print(f"ERROR: unknown align config {align_cfg!r}. Known align configs: {sorted(ALIGN_CONFIGS)}")
            sys.exit(1)

    row = rows[key]
    row["review_status"] = args.status
    row["refit_config"] = (args.config or "default") if args.status == "redo" else ""
    row["refit_align_config"] = (args.align_config or "default") if args.status == "redo" else ""
    if args.notes is not None:
        row["notes"] = args.notes
    rows[key] = row
    save_manifest(manifest_path, rows)
    print(f"Updated {args.dataset}/{args.case}: review_status={row['review_status']} "
          f"refit_config={row['refit_config']!r} refit_align_config={row['refit_align_config']!r}")


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
        failed, done_unreviewed = _review_queue(items)
        queue = failed + done_unreviewed
        if not queue:
            print("Nothing left to review.")
            return

        r = queue[0]
        key = (r["dataset"], r["case_name"])
        if r["status"] == "failed":
            image = _raw_mesh_sanity_for(r)
            kind = "raw mesh (never finished fitting)"
        else:
            sanity = Path(r["save_dir"]) / "sanity" / "sanity_final.png"
            image = str(sanity) if sanity.exists() else None
            kind = f"fitted sanity, chamfer_best={r['chamfer_best']}"

        print(f"\n=== {r['dataset']}/{r['case_name']} -- {kind}  ({len(queue)} left) ===")
        if image:
            _pop_image(image)
        else:
            print("  (no image found)")

        raw = input("  [p]ass  [s]kip  [d]redo  [c]reclip  [f]ix  [q]uit > ").strip().lower()
        if raw in ("q", "quit"):
            print("Stopping.")
            return
        status = _STATUS_KEYS.get(raw)
        if status is None:
            print(f"  unrecognized input {raw!r}, try again")
            continue

        fit_config = align_config = None
        if status == "redo":
            fit_config = input(f"  fit config ({', '.join(sorted(CONFIGS))}) [default] > ").strip() or "default"
            if fit_config not in CONFIGS:
                print(f"  unknown fit config {fit_config!r}, not recorded -- try again")
                continue
            align_config = input(f"  align config ({', '.join(sorted(ALIGN_CONFIGS))}) [default] > ").strip() or "default"
            if align_config not in ALIGN_CONFIGS:
                print(f"  unknown align config {align_config!r}, not recorded -- try again")
                continue
            if fit_config == "default" and align_config == "default":
                print("  both configs are default -- nothing new to run, not recorded, try again")
                continue

        notes = input("  notes (optional) > ").strip() or None

        row = rows[key]
        row["review_status"] = status
        row["refit_config"] = fit_config if status == "redo" else ""
        row["refit_align_config"] = align_config if status == "redo" else ""
        if notes:
            row["notes"] = notes
        rows[key] = row
        save_manifest(manifest_path, rows)
        print(f"  -> recorded {status}" + (f" (fit={fit_config}, align={align_config})" if status == "redo" else ""))


def cmd_review_list(args):
    rows = load_manifest(Path(args.manifest))
    items = list(rows.values())
    if args.dataset:
        items = [r for r in items if r["dataset"] == args.dataset]
    failed, done_unreviewed = _review_queue(items)

    print(f"=== needs review: {len(failed)} failed, {len(done_unreviewed)} done-but-unreviewed (showing up to {args.limit} each) ===")
    for r in failed[:args.limit]:
        img = _raw_mesh_sanity_for(r) if args.render_failed else None
        print(f"FAILED\t{r['dataset']}\t{r['case_name']}\t{img or '(no raw mesh found under geometry_dir)'}")
    for r in done_unreviewed[:args.limit]:
        sanity = Path(r["save_dir"]) / "sanity" / "sanity_final.png"
        print(f"DONE\t{r['dataset']}\t{r['case_name']}\tchamfer_best={r['chamfer_best']}\t{sanity}")


def cmd_requeue(args):
    rows = load_manifest(Path(args.manifest))
    pending = [r for r in rows.values() if r["refit_config"] or r["refit_align_config"]]
    if args.dataset:
        pending = [r for r in pending if r["dataset"] == args.dataset]
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

    gpus = args.gpus or ["cuda:0", "cuda:1", "cuda:2"]
    buckets = [[] for _ in gpus]
    for i, r in enumerate(pending):
        buckets[i % len(gpus)].append(r)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    n_launched = 0
    for g, gpu in enumerate(gpus):
        queue = buckets[g]
        if not queue:
            continue
        session = f"ghd_requeue_{gpu.replace(':', '_')}"
        subprocess.run(["tmux", "kill-session", "-t", session], stderr=subprocess.DEVNULL)

        lines = [CONDA_INIT]
        for r in queue:
            fit_cfg = r["refit_config"] or "default"
            align_cfg = r["refit_align_config"] or "default"
            extra_flags = f"{ALIGN_CONFIGS[align_cfg]} {CONFIGS[fit_cfg]}".strip()
            # Combo-specific save root -- e.g. runtime/ImperialNHS_endpoint_focus_
            # with_opening -- so a redo NEVER overwrites the case's existing fit
            # under a different combo; every combo lives on disk side by side.
            # "default" on both axes keeps the original runtime/<dataset> location.
            dataset_dir_name = _combo_dir_name(r["dataset"], align_cfg, fit_cfg)
            save_root = str(ROOT / "runtime" / dataset_dir_name)
            out_dir = f"{save_root}/{r['case_name']}"
            cmd = (f'{args.python} ghd/fitting/run_case.py --case-dir "{r["geometry_dir"]}" '
                   f'--save-root "{save_root}" --device {gpu} --n-iter {args.n_iter} --eta-min 1e-4 {extra_flags}').strip()
            lines.append(f"echo '[{gpu}] === REDO {r['case_name']} (align={align_cfg}, fit={fit_cfg}) -> {out_dir} ==='")
            lines.append(f"if {cmd}; then")
            lines.append(f"  echo '{fit_cfg}' > \"{out_dir}/config_used.txt\"")
            lines.append(f"  echo '{align_cfg}' > \"{out_dir}/align_used.txt\"")
            lines.append(f"  echo '[{gpu}] REDO OK: {r['case_name']}'")
            lines.append("else")
            lines.append(f"  echo '[{gpu}] REDO FAILED: {r['case_name']}'")
            lines.append("fi")
        lines.append(f"echo ALL_DONE_{session}")
        worker_script = log_dir / f"requeue_worker_{g}.sh"
        worker_script.write_text("\n".join(lines) + "\n")

        print(f"Session {session} on {gpu}: {len(queue)} redo job(s) queued")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, f"bash '{worker_script}'"])
        subprocess.run(["tmux", "pipe-pane", "-t", session, "-o", f"cat >> '{log_dir / (session + '.log')}'"])
        n_launched += len(queue)

    print(f"Launched {n_launched} redo job(s) across {sum(1 for b in buckets if b)} GPU session(s).")
    print("Run `scan` again once they finish -- fulfilled redo rows auto-clear refit_config.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Scan a dataset's output dir, upsert manifest rows.")
    p_scan.add_argument("--dataset", required=True)
    p_scan.add_argument("--dataset-root", required=True, help="e.g. runtime/ImperialNHS")
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
    p_update.add_argument("--status", required=True, choices=["pass", "skip", "redo", "reclip", "fix"],
                          help="pass=good as-is, skip=give up on it, redo=refit with --config, "
                               "reclip=needs AneuSeg/reclip.py run manually before refitting, "
                               "fix=needs some other manual intervention before refitting.")
    p_update.add_argument("--config", default=None,
                          help="Fit config preset name (from fit_configs.py). Defaults to "
                               "'default' if --status redo and only --align-config is given.")
    p_update.add_argument("--align-config", default=None,
                          help="Align (Stage 1) config preset name (from align_configs.py). "
                               "Defaults to 'default' if --status redo and only --config is given.")
    p_update.add_argument("--notes", default=None)
    p_update.add_argument("--manifest", default=str(DEFAULT_MANIFEST))

    p_list = sub.add_parser("review-list", help="Print cases needing review, worst first.")
    p_list.add_argument("--dataset", default=None)
    p_list.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_list.add_argument("--limit", type=int, default=8)
    p_list.add_argument("--no-render-failed", dest="render_failed", action="store_false",
                         help="Skip raw-mesh rendering for failed cases (default: render).")

    p_req = sub.add_parser("requeue", help="Launch tmux redo jobs for every refit_config-marked row.")
    p_req.add_argument("--dataset", default=None)
    p_req.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_req.add_argument("--gpus", nargs="+", default=None)
    p_req.add_argument("--n-iter", type=int, default=7500)
    p_req.add_argument("--python", default=DEFAULT_PYTHON)
    p_req.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))

    args = parser.parse_args()
    {
        "scan": cmd_scan, "review": cmd_review, "next": cmd_next, "update": cmd_update,
        "review-list": cmd_review_list, "requeue": cmd_requeue,
    }[args.command](args)


if __name__ == "__main__":
    main()
