"""Shape metrics of aneurysm sac masks (label 2), for a 'does this look like a dome' check.

All on a binary sac mask at isotropic voxels (real masks are resampled to 0.35 mm, the patch
spacing; generated sacs are already at 0.35 mm), per face-connected sac piece:
  volume_mm3
  elongation   sqrt(l1 / l3), principal-axis eigenvalues of the voxel coordinates (1 = round)
  flatness     sqrt(l2 / l3)
  thickness    inscribed-sphere diameter / length along the first principal axis (1 = ball,
               small = stripe or sliver)
  solidity     volume / convex-hull volume (1 = convex, small = crescent, smear)
Two modes:
  python scripts/ostium/sac_shape.py
      reference: every ImperialNHS label.nrrd (one per aneurysm folder; read only, never written),
      -> dataset/sac_shape_reference.json (per-sac metrics and quantiles)
  python scripts/ostium/sac_shape.py --generated runtime_infer/ostium/<run>
      the merged sacs (merged_XX.npz, label 2) of run_merge.py, compared with the reference
      -> <run>/sac_shape.json, sac_shape.md

The check is meant to rule out only extreme, weird shapes, so each limit sits beyond the most
extreme real sacs: a margin past the real 1st / 99th percentile (LIMITS). A generated sac is flagged
'not_dome_like' when any metric is past its limit.
"""
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GEOM = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
from inference.sac_shape import FACE, LIMITS, VOX, limits, shape_metrics  # noqa: E402


def real_case(f):
    import nrrd
    d, h = nrrd.read(f)
    sp = np.linalg.norm(np.asarray(h["space directions"], float), axis=1)    # per array axis
    sac = d == 2
    lab, n = ndi.label(sac, structure=FACE)
    rows = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        piece = lab[sl] == k
        if piece.sum() * np.prod(sp) < 1.0:                                  # < 1 mm3: label noise
            continue
        piece = np.pad(piece, 2).astype(np.float32)
        r = ndi.zoom(piece, sp / VOX, order=1) > 0.5
        rows.append(dict(case=Path(f).parent.name, piece=k, n_pieces=n, **shape_metrics(r)))
    return rows


def generated(infer):
    """Metrics of every merged sac in a run_merge.py output folder: the whole label 2 and its main
    piece, against the reference limits."""
    infer = Path(infer)
    ref = json.loads((REPO / "dataset" / "sac_shape_reference.json").read_text())
    lim = limits(ref)
    rows = []
    for f in sorted(infer.glob("*/merged_*.npz")):
        sac = np.load(f)["seg"] == 2
        lab, n = ndi.label(sac, structure=FACE)
        main = lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else sac
        r = dict(case=f.parent.name, sample=int(f.stem.split("_")[1]), pieces=int(n), **shape_metrics(main))
        r["main_piece_share"] = float(main.sum() / max(sac.sum(), 1))
        r["past_limit"] = [m for m, (side, v) in lim.items() if m in r and (r[m] > v if side == "high" else r[m] < v)]
        rows.append(r)
    (infer / "sac_shape.json").write_text(json.dumps(dict(limits=lim, sacs=rows), indent=1))
    keys = ["volume_mm3", "elongation", "flatness", "thickness", "solidity"]
    L = [f"# Sac shape: {infer.name}", "", f"Real reference: {ref['n_sacs']} sacs from {ref['n_masks']} masks at {ref['vox_mm']} mm. "
         "Limits (flag past): " + ", ".join(f"{m} {'>' if s == 'high' else '<'} {v:.2f}" for m, (s, v) in lim.items()), "",
         "| metric | real p1 | real p50 | real p99 | generated p1 | generated p50 | generated p99 |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        g = np.array([r[k] for r in rows if k in r])
        q = ref["quantiles"][k]
        L.append(f"| {k} | {q['1']:.2f} | {q['50']:.2f} | {q['99']:.2f} | {np.percentile(g, 1):.2f} | {np.median(g):.2f} | {np.percentile(g, 99):.2f} |")
    L += ["", f"Flagged not_dome_like: {sum(bool(r['past_limit']) for r in rows)} of {len(rows)}", "",
          "| case | sample | volume mm3 | pieces | elongation | flatness | thickness | solidity | past limit |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("thickness", 0)):
        L.append(f"| {r['case']} | {r['sample']} | {r['volume_mm3']:.1f} | {r['pieces']} | {r.get('elongation', np.nan):.2f} | "
                 f"{r.get('flatness', np.nan):.2f} | {r.get('thickness', np.nan):.2f} | {r.get('solidity', np.nan):.2f} | {', '.join(r['past_limit'])} |")
    (infer / "sac_shape.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    if "--generated" in sys.argv:
        return generated(sys.argv[sys.argv.index("--generated") + 1])
    fs = sorted(glob.glob(f"{GEOM}/*/label.nrrd"))
    uniq = fs                                                               # aneurysm1/2 have their own masks
    with Pool(8) as P:
        rows = [r for rs in P.map(real_case, uniq) for r in rs]
    out = REPO / "dataset" / "sac_shape_reference.json"
    keys = ["volume_mm3", "elongation", "flatness", "thickness", "solidity"]
    q = {k: {p: float(np.percentile([r[k] for r in rows if k in r], p)) for p in (1, 2, 5, 10, 25, 50, 75, 90, 95, 99)} for k in keys}
    out.write_text(json.dumps(dict(n_masks=len(uniq), n_sacs=len(rows), vox_mm=VOX, quantiles=q, sacs=rows), indent=1))
    print(len(uniq), "masks,", len(rows), "sacs ->", out)
    for k in keys:
        print(k, {p: round(v, 2) for p, v in q[k].items()})


if __name__ == "__main__":
    main()
