"""Check models/dome_markers.py (differentiable dome volume / aspect ratio).

  1 analytic: spherical caps cut from icospheres by a plane (p linear in z, p = 0.5 on the plane):
    volume pi h^2 (3R - h) / 3, height h, neck radius sqrt(R^2 - z0^2), at three mesh resolutions
  2 implementation: on generated complexes (run_inference.py samples), the same p = 0.5 cut surface
    closed explicitly with the fan from the neck centre, volume by trimesh (watertight checked)
  3 pipeline dome: trimesh volume of geometry/ostium.closed_dome (whole faces with all three
    vertices p > 0.5, largest patch, holes filled, fan over each rim loop): the dome merge.py uses
  4 gradients: autograd vs central differences along random directions, in V and in p (float64)

  python scripts/ostium/test_dome_markers.py [--runs runtime_infer/ostium/<run> ...] [--out <folder>]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def spherical_caps(dm):
    import torch
    import trimesh
    rows = []
    for sub in (3, 4, 5):
        s = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
        V = torch.tensor(s.vertices)[None]
        F = torch.tensor(s.faces, dtype=torch.long)
        for z0 in (-0.5, 0.0, 0.5):
            p = (0.5 + (V[..., 2] - z0) / 4).clamp(0, 1)
            o = dm(V, p, F)
            h = 1 - z0
            rows.append(dict(subdivisions=sub, faces=len(F), z0=z0,
                             volume=float(o["volume"]), volume_true=np.pi * h ** 2 * (3 - h) / 3,
                             height=float(o["height"]), height_true=h,
                             neck_radius=float(o["neck_radius"]), neck_radius_true=float(np.sqrt(1 - z0 ** 2))))
    return rows


def explicit_cut_volume(dm, V, p, F):
    """The cut surface + a fan from the neck centre over its boundary, built as a separate numpy
    mesh; trimesh volume. Returns (volume, watertight, fan triangles, crossing edges)."""
    import trimesh
    tri, ok, X, cross = dm.dome_triangles(V, p, F)
    T = tri[0][ok[0]].detach().numpy()
    o = X[0][cross[0]].mean(0).detach().numpy()
    # boundary of the cut surface: triangle edges used once (vertices matched by position)
    seg, fan = {}, []
    for t in T:
        for k in range(3):
            a, b = tuple(np.round(t[k], 12)), tuple(np.round(t[(k + 1) % 3], 12))
            key = (min(a, b), max(a, b))
            seg.setdefault(key, []).append((a, b))
    for key, uses in seg.items():
        if len(uses) == 1:                                    # boundary segment, oriented as in its face
            a, b = uses[0]
            fan.append([b, a, tuple(o)])
    T2 = np.concatenate([T, np.array(fan)], 0)
    m = trimesh.Trimesh(T2.reshape(-1, 3), np.arange(len(T2) * 3).reshape(-1, 3), process=True)
    return float(m.volume), bool(m.is_watertight), len(fan), int(cross[0].sum())


def gradient_check(dm, V, p, F, n_dir=4, eps=1e-6, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    out = []
    for name in ("V", "p"):
        for _ in range(n_dir):
            Vv, pv = V.clone().requires_grad_(name == "V"), p.clone().requires_grad_(name == "p")
            r = dm(Vv, pv, F)
            f = r["volume"].sum() + r["aspect_ratio"].sum()
            (gr,) = torch.autograd.grad(f, Vv if name == "V" else pv)
            d = torch.randn(gr.shape, generator=g, dtype=gr.dtype)
            if name == "p":
                d = d * 1e-2
            def fval(s):
                rr = dm(V + s * d if name == "V" else V, p + s * d if name == "p" else p, F)
                return float(rr["volume"].sum() + rr["aspect_ratio"].sum())
            num = (fval(eps) - fval(-eps)) / (2 * eps)
            ana = float((gr * d).sum())
            out.append(dict(wrt=name, autograd=ana, numeric=num, rel_err=abs(ana - num) / max(abs(num), 1e-12)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[str(REPO / "runtime_infer/ostium/abl_mlp_crop_norot_ep1000"),
                                                  str(REPO / "runtime_infer/ostium/abl_mlp_crop_norot_ep1000_seed1")])
    ap.add_argument("--out", default=str(REPO / "runtime_debug/dome_markers"))
    a = ap.parse_args()
    import torch
    torch.set_default_dtype(torch.float64)
    from geometry.ostium import closed_dome
    from models.dome_markers import DomeMarkers
    dm = DomeMarkers()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = spherical_caps(dm)
    rows, grads = [], []
    files = sorted(f for r in a.runs for f in glob.glob(f"{r}/*/sample_*.npz"))
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        V = torch.tensor(z["verts_mm"], dtype=torch.float64)[None]
        p = torch.tensor(z["p_dome"], dtype=torch.float64)[None]
        F = torch.tensor(z["faces"], dtype=torch.long)
        if not (p > 0.5).any():
            continue
        r = dm(V, p, F)
        v_cut, wt, n_fan, n_rim = explicit_cut_volume(dm, V, p, F)
        m = closed_dome(z["verts_mm"], z["faces"], z["p_dome"] > 0.5)
        met = json.loads(str(z["metrics"]))
        rows.append(dict(sample=str(Path(f).relative_to(REPO)), volume=float(r["volume"]), volume_cut_trimesh=v_cut,
                         cut_watertight=wt, volume_pipeline_trimesh=float(m.volume), pipeline_watertight=bool(m.is_watertight),
                         height=float(r["height"]), neck_radius=float(r["neck_radius"]), aspect_ratio=float(r["aspect_ratio"]),
                         plaus_height=met.get("dome_height_mm"), plaus_ar=met.get("aspect_ratio")))
        if i % 15 == 0:
            grads += gradient_check(dm, V, p, F)
    rep = dict(caps=caps, samples=rows, gradients=grads)
    (out / "test_dome_markers.json").write_text(json.dumps(rep, indent=1))
    L = ["# Dome markers check", "", "## 1 Spherical caps (R = 1)", "",
         "| subdiv | faces | z0 | volume | true | err | height | true | neck r | true |", "|---" * 10 + "|"]
    for c in caps:
        L.append(f"| {c['subdivisions']} | {c['faces']} | {c['z0']} | {c['volume']:.4f} | {c['volume_true']:.4f} | "
                 f"{c['volume'] / c['volume_true'] - 1:+.2%} | {c['height']:.4f} | {c['height_true']:.4f} | "
                 f"{c['neck_radius']:.4f} | {c['neck_radius_true']:.4f} |")
    v = np.array([r["volume"] for r in rows])
    vc = np.array([r["volume_cut_trimesh"] for r in rows])
    vp = np.array([r["volume_pipeline_trimesh"] for r in rows])
    rel_c, rel_p = v / vc - 1, v / vp - 1
    q = lambda x: f"median {np.median(x):+.2%}, 5-95% [{np.percentile(x, 5):+.2%}, {np.percentile(x, 95):+.2%}], max |{np.abs(x).max():.2%}|"
    L += ["", f"## 2 Same cut surface, closed explicitly, trimesh volume ({len(rows)} generated complexes)", "",
          f"relative difference: {q(rel_c)}; watertight {sum(r['cut_watertight'] for r in rows)}/{len(rows)}", "",
          "## 3 Against the pipeline's closed dome (whole faces p > 0.5, cleaned, trimesh)", "",
          f"relative difference: {q(rel_p)}; pipeline dome watertight {sum(r['pipeline_watertight'] for r in rows)}/{len(rows)}",
          f"volumes (mm3): median {np.median(v):.1f}, range {v.min():.1f} - {v.max():.1f}", "",
          "## 4 Gradients (volume + aspect ratio, central differences, float64)", "",
          "| wrt | autograd | numeric | rel err |", "|---|---|---|---|"]
    for g in grads:
        L.append(f"| {g['wrt']} | {g['autograd']:.6g} | {g['numeric']:.6g} | {g['rel_err']:.1e} |")
    ar = np.array([r["aspect_ratio"] for r in rows])
    par = np.array([r["plaus_ar"] for r in rows if r["plaus_ar"] is not None], float)
    L += ["", f"aspect ratio (height / neck radius): median {np.median(ar):.2f}; half of it {np.median(ar) / 2:.2f} vs the "
              f"plausibility check's height / neck diameter (generated ring, frame normal) median {np.median(par):.2f}"]
    (out / "test_dome_markers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
