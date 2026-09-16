"""Flag ImperialNHS CFD cases whose surface is rougher than the physiological reference.

For each ImperialNHS case in the registry, geomagic_processed.obj -- the surface the CFD
mesh was built from -- is measured with mesh_regularizer's MeshRegularizer.assess(). The
shape is compared against the reference roughness model without being modified: a
remeshed COPY at the reference edge length is measured, because that is what makes the
scores comparable to the reference.

REFERENCE MODEL. mesh_regularizer's default model, whatever that currently is; its name is
recorded with every result and in the registry. At the time of writing it is
reference_roughness__t0.3-0.5-0.8-1-1.2__g1.6: target scales 0.3, 0.5, 0.8, 1.0 and 1.2 mm,
guard scale 1.6 mm, built from 315 AneuX cases.

WHAT COUNTS AS TOO ROUGH. Roughness is scored per scale as signed SD above the reference
mean. A case is flagged when any TARGET scale sits more than TOLERANCE_SD (1.0) above it --
the same rule by which the volume mesher decides a surface needs smoothing. The guard
scale is recorded but never flags: it only protects against over-smoothing. A surface
smoother than the reference is not flagged either.

EMPTY MESHES. Some geomagic_processed.obj files hold no geometry at all: a 104-byte file
whose Open3D header reads "number of vertices: 0" (20 of 289 ImperialNHS cases when this
was written). Their CFD ran from a mesh built before the file was emptied, so there is
nothing here to measure. They get status empty_mesh rather than error.

These columns are added to _meta/case_registry.xlsx. Every existing roughness_* column is
removed first, so results from an earlier reference model never linger beside new ones.
Cases from other datasets are left blank.

    roughness_status            rough | ok | empty_mesh | missing_mesh | error
    roughness_flag              True only for status 'rough'
    roughness_verdict           rougher_than_reference | within_reference | smoother_than_reference
    roughness_worst_target_sd   the largest signed SD over the target scales
    roughness_sd_<scale>        signed SD above the reference mean, one column per model scale
    roughness_guard_below_band  how far the guard scale sits below the reference band
    roughness_reference_model   the reference model the values were measured against

CHECKPOINTS. Every result is written as soon as it is measured to
_meta/roughness_assessment__<reference model>.json, keyed by case and the mesh file's size
and modification time. There is one file per reference model, so results against an
earlier model are kept for comparison rather than overwritten, and a re-run skips only
cases already measured against the same model on an unchanged mesh.

    python fix_cfd/check_roughness.py --workers 2             # measure, then write registry
    python fix_cfd/check_roughness.py --registry_only         # write registry from the checkpoint
    python fix_cfd/check_roughness.py --cases A B --dry_run   # try a few, write nothing
"""

import os

# One thread per worker: numpy inside the regularizer would otherwise take every core in
# every worker process. Set before numpy is imported anywhere.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
META_DIR = os.path.join(DATASET_ROOT, "_meta")
case_registry = os.path.join(META_DIR, "case_registry.xlsx")

DATASET = "ImperialNHS"
MESH_NAME = "geomagic_processed.obj"
TOLERANCE_SD = 1.0
N_SEEDS = 4000          # what the reference models themselves were scanned with

_REG = None             # one MeshRegularizer per worker process


def _init_worker():
    global _REG
    from mesh_regularizer import MeshRegularizer
    _REG = MeshRegularizer(tolerance=TOLERANCE_SD, verbose=False)


def _assess(case, mesh_path):
    t0 = time.time()
    try:
        a = _REG.assess(mesh_path, n_seeds=N_SEEDS)
        return case, {"assessment": a, "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        return case, {"error": repr(exc), "seconds": round(time.time() - t0, 1)}


def _is_empty_mesh(path):
    """True for an OBJ with no vertices, such as Open3D's 'number of vertices: 0' stub."""
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                return False
    return True


def _fingerprint(path):
    st = os.stat(path)
    return {"mesh_size": st.st_size, "mesh_mtime": st.st_mtime}


def _load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1, default=float)
    os.replace(tmp, path)        # atomic: a crash mid-write cannot corrupt the checkpoint


def _scale_col(s):
    return "roughness_sd_%g" % float(s)


def _columns(scales):
    return (["roughness_status", "roughness_flag", "roughness_verdict",
             "roughness_worst_target_sd"] + [_scale_col(s) for s in scales]
            + ["roughness_guard_below_band", "roughness_reference_model"])


def _row(entry, scales, model_name):
    blank = {c: None for c in _columns(scales)}
    if entry is None:
        return blank
    base = {**blank, "roughness_flag": False, "roughness_reference_model": model_name}
    if entry.get("missing"):
        return {**base, "roughness_status": "missing_mesh"}
    if entry.get("empty"):
        return {**base, "roughness_status": "empty_mesh"}
    if "error" in entry:
        return {**base, "roughness_status": "error"}
    a = entry["assessment"]
    rough = bool(a["needs_regularization"])
    row = {**base, "roughness_status": "rough" if rough else "ok", "roughness_flag": rough,
           "roughness_verdict": a["verdict"],
           "roughness_worst_target_sd": a["worst_target_excess_sd"],
           "roughness_guard_below_band": a.get("guard_below_band")}
    sd = a["signed_excess_sd"]                   # keyed by str(scale), e.g. "1.0"
    for s in scales:
        row[_scale_col(s)] = sd.get(str(float(s)))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=case_registry)
    p.add_argument("--cache", default=None,
                   help="checkpoint file (default: _meta/roughness_assessment__<model>.json)")
    p.add_argument("--workers", type=int, default=2,
                   help="parallel processes; each holds about 2 GB")
    p.add_argument("--cases", nargs="*", default=None, help="only these ImperialNHS cases")
    p.add_argument("--registry_only", action="store_true",
                   help="skip measuring; rebuild the registry columns from the checkpoint")
    p.add_argument("--dry_run", action="store_true", help="measure but do not write the registry")
    args = p.parse_args()

    import pandas as pd
    from mesh_regularizer import MeshRegularizer

    ref = MeshRegularizer(tolerance=TOLERANCE_SD, verbose=False)
    model_name = ref.model.config_name
    scales = tuple(float(s) for s in ref.model.scales)
    cache_path = args.cache or os.path.join(META_DIR, "roughness_assessment__%s.json" % model_name)
    print("reference model: %s | targets %s mm, guard %s mm | checkpoint %s"
          % (model_name, list(ref.target_scales), ref.guard_scale, cache_path), flush=True)

    reg = pd.read_excel(args.registry)
    imp = reg[reg["dataset"] == DATASET]
    if args.cases:
        imp = imp[imp["case"].isin(args.cases)]
    cache = _load_cache(cache_path)

    if not args.registry_only:
        todo = []
        for _, r in imp.iterrows():
            mesh = os.path.join(r["case_dir"], MESH_NAME)
            if not os.path.isfile(mesh):
                cache[r["case"]] = {"missing": True, "reference_model": model_name}
                continue
            fp = _fingerprint(mesh)
            if fp["mesh_size"] < 10_000 and _is_empty_mesh(mesh):
                cache[r["case"]] = {"empty": True, "reference_model": model_name, **fp}
                continue
            old = cache.get(r["case"])
            if (old and "assessment" in old and old.get("reference_model") == model_name
                    and all(old.get(k) == v for k, v in fp.items())):
                continue                          # already measured on this mesh, this model
            todo.append((r["case"], mesh, fp))
        _save_cache(cache, cache_path)            # keep empty/missing records even if nothing runs
        print("%d %s case(s): %d to measure, %d skipped (already measured, empty or missing)"
              % (len(imp), DATASET, len(todo), len(imp) - len(todo)), flush=True)

        if todo:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                     initializer=_init_worker) as pool:
                futs = {pool.submit(_assess, c, m): fp for c, m, fp in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    case, entry = fut.result()
                    cache[case] = {**entry, **futs[fut], "reference_model": model_name}
                    _save_cache(cache, cache_path)
                    if "error" in entry:
                        print("[%d/%d] %s ERROR %s" % (i, len(todo), case, entry["error"]),
                              flush=True)
                    else:
                        a = entry["assessment"]
                        print("[%d/%d] %s %s worst %+.2f SD (%.0f s)"
                              % (i, len(todo), case, a["verdict"], a["worst_target_excess_sd"],
                                 entry["seconds"]), flush=True)

    rows = {r["case"]: _row(cache.get(r["case"]), scales, model_name) for _, r in imp.iterrows()}
    status = pd.Series({c: v["roughness_status"] for c, v in rows.items()})
    print("\n%s roughness: %s" % (DATASET, status.value_counts(dropna=False).to_dict()))
    rough = sorted(c for c, v in rows.items() if v["roughness_flag"])
    empty = sorted(c for c, v in rows.items() if v["roughness_status"] == "empty_mesh")
    if empty:
        print("empty processed mesh, not measurable (%d): %s" % (len(empty), ", ".join(empty)))
    print("rougher than the reference (> %.1f SD at a target scale): %d" % (TOLERANCE_SD, len(rough)))

    if args.dry_run:
        print("--dry_run: registry not written")
        return
    if args.cases:
        print("--cases given: registry not written, so a partial run cannot blank other cases")
        return

    cols = _columns(scales)
    new = pd.DataFrame.from_dict(rows, orient="index")[cols]
    reg = reg.drop(columns=[c for c in reg.columns if str(c).startswith("roughness_")])
    reg = reg.merge(new, left_on="case", right_index=True, how="left")
    backup = "%s.bak_%s" % (args.registry, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(args.registry, backup)
    reg.to_excel(args.registry, index=False)
    print("registry updated: %s  (backup: %s)" % (args.registry, os.path.basename(backup)))


if __name__ == "__main__":
    main()
