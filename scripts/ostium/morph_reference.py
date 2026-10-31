"""Morphological markers of every real aneurysm in the ostium dataset: the condition labels of the
morphology-conditioned ostium model (train_ostium_model.py --morph_cond 1).

For each case: the fitted complex (GHD phi + scale, placed by its affine in the ostium frame), the
frozen morphology sensor's dome probability on it, then models/dome_markers.DomeMarkers: dome
volume (mm3), height along the neck normal (mm), mean neck radius (mm) and aspect ratio
height / neck radius. Also the same markers on the complex decoded through the VAE (from its z),
which is what the model can produce at best. The markers do not depend on the augmentation
(rotation, crop, site jitter), so one value per case.

Writes dataset/ostium_morph_reference.json: per case (dataset, case, markers) and the condition
statistics (mean, std of [log volume, aspect ratio]) the model standardises with.

  python scripts/ostium/morph_reference.py --gpu 3          # never GPU 2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))
OUT = REPO / "dataset" / "ostium_morph_reference.json"


def markers(dm, dec, T, phi, log_scale, aff, atype, scale):
    """Markers (mm) of complexes of one type: GHD phi, log scale (b,), affine (b, 10) in the ostium frame."""
    V = T.local_vertices(dec.vcan(phi, atype), log_scale, aff)
    p, *_ = T.sensor_rim(dec, phi, V, atype.expand(len(phi)))
    rec = dec.mc.get(int(atype))
    import torch
    F = rec.ghd_forward_as_Meshes(torch.zeros(1, 432, device=rec.GHD_eigvec.device),
                                  denormalize_shape=False).faces_packed().to(V.device)
    o = dm(V, p, F)
    mm = scale[:, 0] * 10
    return dict(volume_mm3=(o["volume"] * mm ** 3), height_mm=o["height"] * mm, neck_radius_mm=o["neck_radius"] * mm,
                aspect_ratio=o["aspect_ratio"], rim_points=o["rim_points"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    a = ap.parse_args()
    assert a.gpu != "2", "never GPU 2"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    import train_ostium_model as T
    from dataset.ostium_dataset import FOLLOW_BATCH, OstiumGraphDataset
    from models.dome_markers import DomeMarkers
    from torch_geometric.data import Batch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = OstiumGraphDataset("all", augment=False, complex_targets=True, seed=0)
    dec = T.Decoder(ds, dev, with_sensor=True)
    dm = DomeMarkers()
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = Batch.from_data_list([ds[i]], follow_batch=FOLLOW_BATCH).to(dev)
            t = b.atype[:1]
            fit = markers(dm, dec, T, b.phi, b.log_scale, b.x_cx[:, 16:], t, b.scale)
            phi_v, l_v = dec.phi_from_z(b.x_cx[:, :16], b.atype)
            vae = markers(dm, dec, T, phi_v, l_v, b.x_cx[:, 16:], t, b.scale)
            r = dict(ds=b.ds[0], case=b.case[0], atype=int(t), **{k: round(float(v[0]), 4) for k, v in fit.items()},
                     vae={k: round(float(v[0]), 4) for k, v in vae.items()})
            rows.append(r)
            if i % 50 == 0:
                print(i, r["case"], r["volume_mm3"], r["aspect_ratio"], "vae", r["vae"]["volume_mm3"], flush=True)
    ok = [r for r in rows if r["rim_points"] > 0 and r["volume_mm3"] > 0]
    c = np.array([[np.log(r["volume_mm3"]), r["aspect_ratio"]] for r in ok])
    q = lambda k: [float(np.percentile([r[k] for r in ok], p)) for p in (1, 5, 25, 50, 75, 95, 99)]
    rel = np.array([r["vae"]["volume_mm3"] / r["volume_mm3"] - 1 for r in ok])
    dar = np.array([r["vae"]["aspect_ratio"] - r["aspect_ratio"] for r in ok])
    ref = dict(n=len(rows), n_valid=len(ok), condition=["log_volume_mm3", "aspect_ratio"],
               mean=c.mean(0).tolist(), std=c.std(0).tolist(),
               quantiles_pct=[1, 5, 25, 50, 75, 95, 99], quantiles={k: q(k) for k in ("volume_mm3", "aspect_ratio", "height_mm", "neck_radius_mm")},
               vae_vs_fit=dict(volume_rel_median=float(np.median(rel)), volume_rel_abs_p90=float(np.percentile(np.abs(rel), 90)),
                               ar_diff_median=float(np.median(dar)), ar_absdiff_p90=float(np.percentile(np.abs(dar), 90))),
               cases=rows)
    OUT.write_text(json.dumps(ref, indent=1))
    print(json.dumps({k: v for k, v in ref.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
