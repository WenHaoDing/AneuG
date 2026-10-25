"""Reconstruction experiment for the canonical Fourier ostium code (TASK §5.2).

For each case in runtime_skeleton/ with ostium.npz + skeleton_graph_points.npz:
  E1  error of the H-harmonic ring against the raw rim, H = 1..8
  E2  canonical form: shifting the raw loop's start / reversing it must not change the code
  E3  flip: encoding with the opposite tangent sign, then decoding, gives the same ring
  E4  stability: coefficient / ring / frame change under sub-voxel jitter of the raw rim
Writes runtime_skeleton/ostium_fourier/{results.json, summary.txt, error_vs_H.png,
<case>_fourier.png}.

  python vessel_edit/exp_ostium_fourier.py
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from geometry import load_point_graph  # noqa: E402
from geometry.ostium import (coeffs_to_vector, decode_ostium, encode_ostium, flip_code,  # noqa: E402
                             local_scale, rim_points)

OUT = REPO / "runtime_skeleton" / "ostium_fourier"
HS = [1, 2, 3, 4, 5, 6, 8]


def curve_dist(A, B):
    """Symmetric point-to-polyline distance between two closed curves (mean, max), mm."""
    def one(P, Q):
        Q2 = np.vstack([Q, Q[:1]])
        a, b = Q2[:-1], Q2[1:]
        ab = b - a
        t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / (ab ** 2).sum(-1)[None], 0, 1)
        d = np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
        return d
    d = np.r_[one(A, B), one(B, A)]
    return float(d.mean()), float(d.max())


def run_case(case_dir, rng):
    o = np.load(case_dir / "ostium.npz", allow_pickle=True)
    pg = load_point_graph(case_dir / "skeleton_graph_points.npz")
    rim = rim_points(o["verts"], o["faces"], o["dome"])
    res = dict(case=case_dir.name, rim_points=len(rim))
    # E1: reconstruction error vs H
    res["E1"] = {}
    for H in HS:
        code = encode_ostium(rim, pg, H=H)
        res["E1"][H] = curve_dist(decode_ostium(code, pg, M=512), rim)
    code5 = encode_ostium(rim, pg, H=5)
    res["p"], res["scale_mm"], res["r_at_P"] = code5["p"], code5["scale"], float(pg["radius"][code5["p"]])
    res["offset_over_scale"] = float(np.linalg.norm(code5["offset"]))
    res["coeff_vec_norm"] = float(np.linalg.norm(coeffs_to_vector(code5["coeffs"])))
    # E2: canonical form
    v0 = coeffs_to_vector(code5["coeffs"])
    worst = 0.0
    for k in rng.integers(1, len(rim), 5):
        for r in (np.roll(rim, k, 0), np.roll(rim, k, 0)[::-1]):
            v = coeffs_to_vector(encode_ostium(r, pg, H=5)["coeffs"])
            worst = max(worst, float(np.abs(v - v0).max()))
    res["E2_max_coeff_change"] = worst
    # E3: flip
    fl = flip_code(code5, pg)
    a, b = decode_ostium(code5, pg, M=512), decode_ostium(fl, pg, M=512)
    res["E3_flip_ring_change_mm"] = curve_dist(a, b)
    res["E3_frame_e1_dot"] = float(code5["frame"][0] @ fl["frame"][0])
    # E4: jitter
    out = {}
    for sig in (0.05, 0.1, 0.2):
        dv, dr, de1, dp = [], [], [], 0
        for _ in range(20):
            j = encode_ostium(rim + rng.normal(0, sig, rim.shape), pg, H=5)
            dv.append(np.linalg.norm(coeffs_to_vector(j["coeffs"]) - v0) / np.linalg.norm(v0))
            dr.append(curve_dist(decode_ostium(j, pg, M=256), decode_ostium(code5, pg, M=256))[0])
            de1.append(np.degrees(np.arccos(np.clip(j["frame"][0] @ code5["frame"][0], -1, 1))))
            dp += int(j["p"] != code5["p"])
        out[sig] = dict(rel_coeff_change=float(np.median(dv)), rel_coeff_change_max=float(np.max(dv)),
                        ring_change_mm=float(np.median(dr)), e1_angle_deg_max=float(np.max(de1)),
                        site_changed=f"{dp}/20")
    res["E4"] = out
    return res, rim, pg


def figure(case, rim, pg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = {H: encode_ostium(rim, pg, H=H) for H in (3, 4, 5)}
    Rf, C = codes[5]["frame"], decode_ostium(codes[5], pg, M=512).mean(0)
    to_local = lambda X: (X - C) @ Rf.T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [((0, 1), "e1-e2 plane (looking down n, from the sac)"),
             ((0, 2), "e1-n plane (side)"), ((1, 2), "e2-n plane (side)")]
    cols = {3: "tab:green", 4: "tab:orange", 5: "tab:blue"}
    for ax, ((i, j), title) in zip(axes, views):
        R = to_local(np.vstack([rim, rim[:1]]))
        ax.plot(R[:, i], R[:, j], color="0.5", lw=1, marker=".", ms=3, label="raw rim")
        for H, code in codes.items():
            D = to_local(decode_ostium(code, pg, M=512))
            D = np.vstack([D, D[:1]])
            ax.plot(D[:, i], D[:, j], color=cols[H], lw=1.8, label=f"H = {H}")
        s0 = to_local(decode_ostium(codes[5], pg, M=512))[0]
        ax.plot(s0[i], s0[j], "k*", ms=12, label="start (H = 5)")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(["e1", "e2", "n"][i] + " (mm)")
        ax.set_ylabel(["e1", "e2", "n"][j] + " (mm)")
    axes[0].legend(fontsize=7)
    fig.suptitle(case)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = []
    for f in sorted(glob.glob(str(REPO / "runtime_skeleton" / "*" / "ostium.npz"))):
        d = Path(f).parent
        if not (d / "skeleton_graph_points.npz").exists():
            continue
        res, rim, pg = run_case(d, rng)
        figure(d.name, rim, pg, OUT / f"{d.name}_fourier.png")
        results.append(res)
        print(d.name, "done", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r in results:
        ax[0].plot(HS, [r["E1"][H][0] for H in HS], marker="o", label=r["case"][:10])
        ax[1].plot(HS, [r["E1"][H][1] for H in HS], marker="o")
    for a, t in zip(ax, ("mean", "max")):
        a.set_xlabel("harmonics H")
        a.set_ylabel(f"{t} distance to raw rim (mm)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_H.png", dpi=120)

    lines = ["E1  distance of the H-harmonic ring to the raw rim, mean / max (mm)",
             "    case                    " + "  ".join(f"H={H:<9d}" for H in HS)]
    for r in results:
        lines.append(f"    {r['case'][:22]:22s}  " + "  ".join(f"{r['E1'][H][0]:.2f}/{r['E1'][H][1]:.2f}  " for H in HS))
    lines.append("\nE2  canonical form: max coefficient change after shifting the start / reversing")
    lines += [f"    {r['case'][:22]:22s}  {r['E2_max_coeff_change']:.2e}" for r in results]
    lines.append("\nE3  flip: ring change after flip + decode (mean/max mm); e1 . e1_flipped")
    lines += [f"    {r['case'][:22]:22s}  {r['E3_flip_ring_change_mm'][0]:.4f}/{r['E3_flip_ring_change_mm'][1]:.4f}   {r['E3_frame_e1_dot']:+.3f}"
              for r in results]
    lines.append("\nE4  jitter of the raw rim (20 trials): median relative coefficient change, "
                 "median ring change (mm), max e1 rotation (deg), site changes")
    for r in results:
        lines.append(f"    {r['case'][:22]:22s}  " + "   ".join(
            f"s={s}: {v['rel_coeff_change']:.3f} {v['ring_change_mm']:.3f}mm {v['e1_angle_deg_max']:.1f}deg {v['site_changed']}"
            for s, v in r["E4"].items()))
    lines.append("\nscale: median graph radius within 3 mm of P (mm), radius at P, |C - P| / scale")
    lines += [f"    {r['case'][:22]:22s}  {r['scale_mm']:.2f}  {r['r_at_P']:.2f}  {r['offset_over_scale']:.2f}"
              for r in results]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
