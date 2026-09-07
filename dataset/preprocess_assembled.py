"""
Preprocess the ASSEMBLED AneuG corpus into training checkpoints.

One script for both datasets: `manage.py assemble` already puts ImperialNHS and
AneuX in the same layout, so the dataset is just a directory level and
everything needed comes from the assembled case directory itself -- no
per-dataset roots, no geometry lookup.

    runtime_dataset/assembled/<dataset>/<case>/{ghd_coefficients.npz,
                                metrics.json, ghd_fitted.obj, landmarks.npz,
                                alignment_result.npy, assembled_from.json}
                ->  runtime_dataset/AneuG_processed/<case>.npy
                    runtime_dataset/AneuG_processed/sanity/<case>.png

Saved fields are what the downstream models actually read. VesselSkeletonDataset
(dataset/skeleton_dataset.py, and its Fourier subclass) uses exactly:

    case, aneurysm_type, canonical_type,
    ghd["phi"], ghd["log_scale"],
    clipped_centerline["coordinate_system"] == "ghd",
    clipped_centerline["branch_points"]

Three small extras ride along: w_rot and t_vec complete the Stage-2 pose (phi
alone does not reproduce the fitted mesh), and s_can is the normalisation the
fit used. Everything the old preprocess_ImperialNHS.py emitted beyond that --
R_equivalent/s_equivalent/t_equivalent, base_branch_ids, branch_ranking,
per-branch group ids, the sources block -- is derived, unused downstream, or
does not exist in AneuG's outputs at all.

Output is FLAT (<output_dir>/<case>.npy), matching VesselSkeletonDataset's
root.glob("*.npy"). Case names do not collide across the two datasets
(checked: 0 of 493), so no dataset subdirectory is needed and the dataset class
needs no change. Sanity PNGs go in a sanity/ subdirectory so the glob ignores them.

WHY THIS LOOKS NOTHING LIKE preprocess_ImperialNHS.py

That script inverts the OLD project's parameterisation -- ghd_world_alignment
with (R, s, t_vec, s_can, R_frame, neck_pt) -- to push world-space centerlines
into GHD space, and reconstructs with ||V_can||max * 1.10 * 2.50.

Neither applies here:

  * AneuG's s_can is plain ||V_can||max (5.375418 bifurcated / 6.037298
    sidewall), so the old *1.10*2.50 would inflate every mesh by 2.75x.
    s_can is read per case from metrics.json instead.

  * landmarks.npz has no neck_pt_world in AneuG at all, so the old affine
    block is not merely redundant here, it is unbuildable.

WHICH CENTERLINE, AND WHY NOT centerline.vtp

The branches must be the ones running OUTWARD from the aneurysm -- the vessel
tree continuing away from the clipped region -- not anything contained in the
mesh. Stage 1's centerline.vtp is NOT that: it holds the short resampled
segments used to drive alignment, which sit entirely inside the fitted mesh
(63/63, 42/42, 42/42 points inside, arc ~3-5 units vs the real branches'
36-45 mm). Using it produces centerlines buried in the mesh.

The right source is clipped_centerline.npy (via resolve_case_paths, so a
hand-repaired "_manual" centerline wins), whose branches lie outside the
clipped mesh and start near the aneurysm. Those are in WORLD space, so they
need Stage 1's similarity applied:

    aligned = s * (world @ R_frame.T) + t_vec

R_frame and t_vec are in landmarks.npz for all 493 cases; the scale s is not
stored there and alignment_result.npy has it for only 397 -- the 96 arap_init
runs write none, that pipeline running its own Stage 1. s is therefore
recovered from the aneurysm_centroid correspondence, which landmarks.npz
stores post-transform while endpoints_manual.npy has it pre-transform:

    s = ((c_aligned - t) . (c_world @ R.T)) / ||c_world @ R.T||^2

On cases where alignment_result.npy exists this reproduces the stored s to
~4e-9, so it is used only where the file is missing and the stored value is
preferred everywhere else.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_ASSEMBLED_ROOT = _REPO_ROOT / "runtime_dataset" / "assembled"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "runtime_dataset" / "AneuG_processed"
CANONICAL_ROOT = _REPO_ROOT / "dataset" / "canonical"

# type 0 -> bifurcated canonical; types 1 and 2 -> sidewall
CANONICAL_BY_TYPE = {0: ("bifurcated", CANONICAL_ROOT / "Bifurcated"),
                     1: ("sidewall", CANONICAL_ROOT / "Sidewall"),
                     2: ("sidewall", CANONICAL_ROOT / "Sidewall")}


def iter_assembled_cases(assembled_root, dataset=None):
    """(dataset, case, case_dir) for every assembled case -- from index.csv
    when present (that is what assemble wrote), else by walking the tree."""
    assembled_root = Path(assembled_root)
    index = assembled_root / "index.csv"
    if index.exists():
        with open(index) as fh:
            for row in csv.DictReader(fh):
                if dataset and row["dataset"] != dataset:
                    continue
                d = assembled_root / row["dataset"] / row["case_name"]
                if d.is_dir():
                    yield row["dataset"], row["case_name"], d
        return
    for ds in sorted(p for p in assembled_root.iterdir() if p.is_dir()):
        if dataset and ds.name != dataset:
            continue
        for case in sorted(p for p in ds.iterdir() if p.is_dir()):
            if (case / "ghd_coefficients.npz").exists():
                yield ds.name, case.name, case


def stage1_transform(case_dir, geometry_dir, endpoints_path):
    """(R, s, t) of Stage 1's similarity: aligned = s * (world @ R.T) + t.

    R and t come from landmarks.npz (present for all cases). s is taken from
    alignment_result.npy where the pipeline wrote one, and otherwise recovered
    from the aneurysm_centroid correspondence -- see the module docstring.
    """
    lm = np.load(Path(case_dir) / "landmarks.npz", allow_pickle=True)
    R = np.asarray(lm["R_frame"], dtype=np.float64)
    t = np.asarray(lm["t_vec"], dtype=np.float64)

    ar = Path(case_dir) / "alignment_result.npy"
    if ar.exists():
        res = np.load(ar, allow_pickle=True).item()
        if "s" in res:
            return R, float(res["s"]), t

    ep = np.load(endpoints_path, allow_pickle=True).item()
    c_world = np.asarray(ep["aneurysm_centroid"], dtype=np.float64)
    c_aligned = np.asarray(lm["aneurysm_centroid"], dtype=np.float64)
    x = c_world @ R.T
    denom = float(x @ x)
    if denom < 1e-12:
        raise ValueError("cannot recover Stage-1 scale: aneurysm centroid at the origin")
    return R, float((c_aligned - t) @ x / denom), t


def outward_branches(clipped, branch_ranking, R, s, t):
    """Clipped-centerline branches, world -> fitted space, in rank order.

    Rank order is what makes branch 0 mean the same thing across cases, so it
    is applied here rather than left to the model. Point order is left alone:
    these branches already start near the aneurysm and run outward, which is
    the direction the skeleton dataset assumes (start_points + local offsets).
    """
    connected = [int(i) for i in clipped["connected_branch_ids"]]
    upstream = int(clipped["upstream_id"])
    base = [upstream] + [i for i in connected if i != upstream]
    ranking = np.asarray(branch_ranking, dtype=np.int64)
    if len(ranking) != len(base):
        raise ValueError(f"branch_ranking length {len(ranking)} != branch count {len(base)}")
    if sorted(ranking.tolist()) != list(range(len(base))):
        raise ValueError(f"branch_ranking is not a permutation: {ranking}")

    out = []
    for i in ranking:
        branch = clipped["branches"][base[int(i)]]
        if branch.get("pts") is None:
            raise ValueError(f"branch {base[int(i)]} has no clipped centerline points")
        pts = np.asarray(branch["pts"], dtype=np.float64)
        out.append((s * (pts @ R.T) + t).astype(np.float32))
    return out


def process_case(dataset, case, case_dir, output_dir, overwrite=False):
    out = Path(output_dir) / f"{case}.npy"
    if out.exists() and not overwrite:
        return out

    from ghd.fitting.alignment import resolve_case_paths

    coeffs = dict(np.load(case_dir / "ghd_coefficients.npz", allow_pickle=True))
    metrics = json.loads((case_dir / "metrics.json").read_text())
    prov = json.loads((case_dir / "assembled_from.json").read_text()) \
        if (case_dir / "assembled_from.json").exists() else {}

    atype = int(metrics["aneurysm_type"])
    if atype not in CANONICAL_BY_TYPE:
        raise ValueError(f"unsupported aneurysm_type={atype}")
    canonical_type, _ = CANONICAL_BY_TYPE[atype]

    # The centerline lives with the GEOMETRY, not the fit. resolve_case_paths is
    # the fit's own rule (manual clip beats automatic, fallback beats primary),
    # so calling it keeps this centerline the one the fit was built against.
    geom = Path(prov.get("geometry_dir") or metrics.get("case_dir") or "")
    if not geom.is_dir():
        raise FileNotFoundError(f"geometry dir missing or unrecorded: {geom}")
    paths = resolve_case_paths(geom)
    for key in ("clipped_centerline", "branch_ranking", "endpoints"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"{Path(paths[key]).name} missing in {geom}")

    R, s, t = stage1_transform(case_dir, geom, Path(paths["endpoints"]))
    branch_points = outward_branches(
        np.load(Path(paths["clipped_centerline"]), allow_pickle=True).item(),
        np.load(Path(paths["branch_ranking"])), R, s, t)

    record = {
        "case": case,
        "dataset": dataset,
        "aneurysm_type": atype,
        "canonical_type": canonical_type,
        "ghd": {
            "phi": np.asarray(coeffs["phi"], dtype=np.float32),
            "log_scale": np.asarray(coeffs["log_scale"], dtype=np.float64),
            "w_rot": np.asarray(coeffs["w_rot"], dtype=np.float32),
            "t_vec": np.asarray(coeffs["t_vec"], dtype=np.float32),
        },
        "clipped_centerline": {
            "coordinate_system": "ghd",
            "branch_points": branch_points,
        },
        "s_can": np.array(float(metrics["s_can"]), dtype=np.float64),
        "provenance": {"config": prov.get("config", ""),
                       "rigid_checkpoint": prov.get("rigid_checkpoint", ""),
                       "chamfer_best": metrics.get("chamfer_best", "")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, record, allow_pickle=True)
    return out


# ── reconstruction + sanity ──────────────────────────────────────────────────

def _load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _so3(w):
    w = np.asarray(w, dtype=np.float64)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def reconstruct(record):
    """phi + pose -> vertices, reproducing ghd_fit.py's render():

        V = ((V0 + U @ phi) @ R.T * exp(log_s) + t) * s_can,  V0 = V_can / s_can

    i.e. the same space as ghd_fitted.obj and centerline.vtp.
    """
    _, root = CANONICAL_BY_TYPE[record["aneurysm_type"]]
    verts, faces = _load_obj(root / "mesh.obj")
    eig = np.load(root / "eigenvectors.npy").astype(np.float32, copy=False)
    phi = np.asarray(record["ghd"]["phi"], dtype=np.float32)
    if eig.shape[0] != verts.shape[0] or eig.shape[1] != phi.shape[0]:
        raise ValueError(f"basis mismatch: verts={verts.shape[0]} eig={eig.shape} phi={phi.shape}")

    s_can = float(np.asarray(record["s_can"]))
    R = _so3(record["ghd"]["w_rot"])
    s = float(np.exp(float(np.asarray(record["ghd"]["log_scale"]))))
    t = np.asarray(record["ghd"]["t_vec"], dtype=np.float64)
    V = (verts / s_can + eig @ phi) @ R.T * s + t
    return (V * s_can).astype(np.float32), faces


def _set_axes_equal(ax, *sets):
    pts = np.concatenate([p for p in sets if p is not None and len(p)], axis=0)
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max() / 2) or 1.0
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def save_sanity(processed_path, case_dir, save_path):
    """Overlay the mesh rebuilt from the SAVED phi (gray) on the fit's own
    ghd_fitted.obj (orange) plus the stored centerlines.

    The two meshes come from different places -- one rebuilt here from phi and
    s_can, one written by the fitter -- so if the normalisation is wrong they
    separate visibly. The reported max vertex deviation is the same check as
    a number.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rec = np.load(processed_path, allow_pickle=True).item()
    verts, faces = reconstruct(rec)
    branches = [np.asarray(b, dtype=np.float32) for b in rec["clipped_centerline"]["branch_points"]]

    ref_v = ref_f = None
    dev = None
    ref_path = Path(case_dir) / "ghd_fitted.obj"
    if ref_path.exists():
        ref_v, ref_f = _load_obj(ref_path)
        if ref_v.shape == verts.shape:
            dev = float(np.abs(ref_v - verts).max())

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.5)
    if ref_v is not None:
        ax.plot_trisurf(ref_v[:, 0], ref_v[:, 1], ref_v[:, 2], triangles=ref_f,
                        color="darkorange", edgecolor="none", alpha=0.15)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branches), 1)))
    for i, pts in enumerate(branches):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, color=colors[i % len(colors)],
                label=f"branch {i}")
        ax.scatter(*pts[0], s=18, color=colors[i % len(colors)])

    p = rec["provenance"]
    ax.set_title(f"{rec['case']}  [{rec['dataset']}]  type {rec['aneurysm_type']} "
                 f"({rec['canonical_type']})\nconfig={p.get('config','?')} "
                 f"ckpt={p.get('rigid_checkpoint','?')} chamfer={p.get('chamfer_best','?')}\n"
                 f"rebuilt from phi: gray | ghd_fitted.obj: orange"
                 + (f" | max dev {dev:.2e}" if dev is not None else ""))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_axes_equal(ax, verts, ref_v, *branches)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path, dev


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess the assembled AneuG corpus (both datasets) into "
                    "one training .npy per case.")
    ap.add_argument("--assembled-root", default=str(DEFAULT_ASSEMBLED_ROOT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--dataset", default=None, help="Limit to one dataset (default: all).")
    ap.add_argument("--case", action="append", dest="cases", help="Only these cases. Repeatable.")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N cases (quick eyeball).")
    ap.add_argument("--aneurysm-types", nargs="+", type=int, choices=(0, 1, 2), default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-sanity", dest="sanity", action="store_false")
    ap.set_defaults(sanity=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    wanted = set(args.cases) if args.cases else None
    ok = failed = skipped = 0
    devs = []

    for ds, case, case_dir in iter_assembled_cases(args.assembled_root, args.dataset):
        if wanted is not None and case not in wanted:
            continue
        # count ATTEMPTS, so --limit still terminates when cases are failing
        if args.limit is not None and (ok + failed) >= args.limit:
            break
        try:
            if args.aneurysm_types is not None:
                t = int(json.loads((case_dir / "metrics.json").read_text())["aneurysm_type"])
                if t not in args.aneurysm_types:
                    skipped += 1
                    continue
            path = process_case(ds, case, case_dir, out_dir, overwrite=args.overwrite)
            if args.sanity:
                _, dev = save_sanity(path, case_dir, out_dir / "sanity" / f"{case}.png")
                if dev is not None:
                    devs.append(dev)
            ok += 1
            print(f"[OK] {ds}/{case}" + (f"  max dev {devs[-1]:.2e}" if args.sanity and devs else ""))
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {ds}/{case}: {exc}")

    print(f"\nProcessed {ok}, failed {failed}, skipped-by-type {skipped} -> {out_dir}")
    if devs:
        print(f"rebuild-vs-ghd_fitted.obj deviation: max {max(devs):.3e}, "
              f"median {float(np.median(devs)):.3e} over {len(devs)} case(s)")


if __name__ == "__main__":
    main()

"""
conda activate new

# quick eyeball pass -- 6 cases with sanity renders
python dataset/preprocess_assembled.py --limit 6 --overwrite

# everything, no renders
python dataset/preprocess_assembled.py --overwrite --no-sanity
"""
