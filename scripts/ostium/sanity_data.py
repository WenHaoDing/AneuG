"""Sanity checks of the ostium/complex dataset targets, before any training.

Writes runtime_debug/ostium_model/sanity_data/ (report.md, results.json, figures).

  S1  ostium code round trip (no augmentation): decoded H=5 ring vs the raw rim
  S2  augmentation consistency (training mode, several draws per case):
        rotation   the code decoded in the rotated scene = the rotated ring
        sigma      decoding at sigma = +1 and -1 gives the same physical ring
        jitter     the code at the jittered site P' gives the same ring as at P
        trimming   every point within keep_mm of the rim (and 15 mm of the site) is kept, the crop
                   is one connected piece, and the site is not at the crop's centre
  S3  complex targets: the 26-number target (z + affine in the ostium frame) rebuilds the placed
      fit: exactly with the true phi, and via the frozen VAE decode of z (its reconstruction error)
  S4  site label and positional encodings: soft label sums to 1 and peaks at P; eigenvalues
      non-negative and ascending; random-walk encodings in [0, 1]
  S5  morphology sensor on the true phi: soft rim (p = 0.5 edge crossings) vs the target ring,
      i.e. the floor of the auxiliary rim loss

  CUDA_VISIBLE_DEVICES="" python scripts/ostium/sanity_data.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "runtime_debug" / "ostium_model" / "sanity_data"


def curve_dist(A, B):
    from scipy.spatial import cKDTree
    d = np.r_[cKDTree(B).query(A)[0], cKDTree(A).query(B)[0]]
    return float(d.mean()), float(d.max())


def dense_ring(code, pg, M=256):
    from geometry.ostium import decode_ostium
    return decode_ostium(code, pg, M=M)


def main():
    import scipy.sparse as sp
    import torch
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.transform import Rotation
    from dataset.ostium_dataset import OstiumGraphDataset, adjacency, geodesic
    from geometry.ostium import encode_ostium, _arc_resample, placed_fit, vector_to_coeffs
    from geometry.complex_target import affine_in_frame, place_from_frame, rotate_scene
    OUT.mkdir(parents=True, exist_ok=True)
    ds = OstiumGraphDataset("all", seed=0, augment=True)
    ds_val = OstiumGraphDataset("all", seed=0, augment=False, complex_targets=False)
    n = len(ds)
    print(f"{n} cases", flush=True)
    res = dict(n_cases=n, S1=[], S2=[], S3=[], S4=[], S5=[])
    rng = np.random.default_rng(1)

    # ------------------------------------------------------------------ S1
    for i in range(n):
        d = ds.case_data(i)
        pg = dict(pos=d["pos"], edge_index=d["ei"], radius=d["radius"], meta=dict(root_point=0))
        code = encode_ostium(d["rim"], pg, H=5)
        m, M = curve_dist(dense_ring(code, pg), d["rim"])
        res["S1"].append(dict(case=d["case"], mean=m, max=M))
    print("S1 done", flush=True)

    # ------------------------------------------------------------------ S2
    for i in range(n):
        d = ds.case_data(i)
        for _ in range(3):
            sel, A0 = ds.trim(d, rng)
            # trimming: guarantee + connectivity + centrality
            from scipy.spatial import cKDTree
            gP0 = geodesic(A0, d["site"])
            must = np.flatnonzero((cKDTree(d["rim"]).query(d["pos"])[0] < ds.keep_mm) & (gP0 < 15.0))
            kept = np.isin(must, sel).all()
            remap = -np.ones(len(d["pos"]), int)
            remap[sel] = np.arange(len(sel))
            msk = (remap[d["ei"][0]] >= 0) & (remap[d["ei"][1]] >= 0)
            ei = remap[d["ei"][:, msk]]
            Acrop = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(len(sel), len(sel)))
            ncomp = connected_components(Acrop, directed=False)[0]
            pos = d["pos"][sel]
            P = int(remap[d["site"]])
            dc = np.linalg.norm(pos - pos.mean(0), axis=1)
            centrality = float((dc < dc[P]).mean())   # 0 = site is the point nearest the centroid
            # rotation, sigma and jitter consistency of the code
            G = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
            posr, rimr = pos @ G.T, d["rim"] @ G.T
            pg = dict(pos=posr, edge_index=ei, radius=d["radius"][sel], meta=dict(root_point=0))
            A = adjacency(posr, ei)
            gP = geodesic(A, P)
            Pj = int(rng.choice(np.flatnonzero(gP <= ds.jitter_mm)))
            ring_ref = dense_ring(encode_ostium(rimr, pg, H=5, site=P, sign=1), pg)
            ring_rot = dense_ring(encode_ostium(d["rim"], dict(pg, pos=pos), H=5, site=P, sign=1), dict(pg, pos=pos)) @ G.T
            ring_sig = dense_ring(encode_ostium(rimr, pg, H=5, site=P, sign=-1), pg)
            ring_jit = dense_ring(encode_ostium(rimr, pg, H=5, site=Pj, sign=1), pg)
            res["S2"].append(dict(case=d["case"], kept_all=bool(kept), components=int(ncomp),
                                  crop_frac=len(sel) / len(d["pos"]), centrality=centrality,
                                  rot=curve_dist(ring_rot, ring_ref)[1], sigma=curve_dist(ring_sig, ring_ref)[1],
                                  jitter=curve_dist(ring_jit, ring_ref)[1], jitter_pts=int(abs(Pj - P))))
    print("S2 done", flush=True)

    # ------------------------------------------------------------------ S3
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    mc = MultiCanonicalGHDReconstruct(str(REPO / "dataset" / "canonical"))
    vae, mean, std = ds._vae, ds._vae_mean, ds._vae_std

    def vcan(phi, t):
        rec = mc.get(t)
        ph = torch.as_tensor(np.asarray(phi, np.float32), device=rec.GHD_eigvec.device).reshape(1, -1)
        return rec.ghd_forward_as_Meshes(ph, denormalize_shape=False).verts_packed().detach().cpu().numpy().astype(float)

    for i in range(n):
        d = ds.case_data(i)
        V_true, _, _ = placed_fit(d["case"], d["ds"])
        G = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
        Qr, br = rotate_scene(d["Q"], d["b"], G)
        # an ostium frame for the rotated scene
        pg = dict(pos=d["pos"] @ G.T, edge_index=d["ei"], radius=d["radius"], meta=dict(root_point=0))
        code = encode_ostium(d["rim"] @ G.T, pg, H=5, sign=int(rng.choice([-1, 1])))
        s = code["scale"]
        C = pg["pos"][code["p"]] + code["offset"] * s
        aff = affine_in_frame(Qr, d["k"], br, C, code["frame"], s)
        Vc = vcan(d["phi"], d["atype"])
        V_exact = place_from_frame(Vc, d["log_scale"], aff, C, code["frame"], s)
        err_exact = float(np.abs(V_exact - V_true @ G.T).max())
        # VAE: decode z -> phi', l'
        with torch.no_grad():
            out, sc = vae.decode(torch.as_tensor(d["z"], dtype=torch.float32)[None], torch.tensor([d["atype"]]))
        xd = torch.cat([out, sc], 1)[0] * std + mean
        phi_v, l_v = xd[:-1].numpy().reshape(-1, 3), float(np.log(max(float(xd[-1]), 1e-6)))
        V_vae = place_from_frame(vcan(phi_v, d["atype"]), l_v, aff, C, code["frame"], s)
        dv = np.linalg.norm(V_vae - V_true @ G.T, axis=1)
        res["S3"].append(dict(case=d["case"], exact_max_mm=err_exact, vae_mean_mm=float(dv.mean()),
                              vae_max_mm=float(dv.max()), aff_scale=float(aff[6]), aff_trans=float(np.linalg.norm(aff[7:]))))
    print("S3 done", flush=True)

    # ------------------------------------------------------------------ S4
    for i in range(min(n, 60)):
        s_ = ds[i]
        q = s_.site_q.numpy()
        ev1, ev2 = s_.eig_1.numpy()[0], s_.eig_2.numpy()[0]
        res["S4"].append(dict(case=s_.case, q_sum=float(q.sum()), argmax_is_site=bool(q.argmax() == int(s_.site_true)),
                              eig_ok=bool((ev1 >= -1e-8).all() and (np.diff(ev1[ev1 > 0]) >= -1e-8).all()),
                              rw_ok=bool(((s_.rw_1 >= -1e-6) & (s_.rw_1 <= 1 + 1e-6)).all()),
                              n0=int(s_.num_nodes), n1=int(s_.n1), n2=int(s_.n2)))
    print("S4 done", flush=True)

    # ------------------------------------------------------------------ S5
    try:
        from models.morphoformer import SensorUncapper
        sens_ckpt = sorted((REPO / "runtime_train/morphoformer/morphology_sensor/h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5").glob("*.pth"))[-1]
        su = SensorUncapper.from_checkpoint(str(sens_ckpt), mc, device="cpu")
        for i in range(min(n, 40)):
            d = ds.case_data(i)
            phi = torch.as_tensor(d["phi"], dtype=torch.float32)[None]
            with torch.no_grad():
                out = su.model(phi, torch.tensor([d["atype"]]))[4]      # forward returns (..., out dict)
            V_true, F, _ = placed_fit(d["case"], d["ds"])
            p = torch.sigmoid(out["dome_logits"][0]).numpy()[:len(V_true)]
            dome_lab = np.asarray(np.load(REPO / "runtime_dataset/AneuG_morpho" / f"{d['case']}.npy", allow_pickle=True).item()["manual_dome"], bool)
            p = p[:len(V_true)]
            e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
            pi, pj = p[e[:, 0]], p[e[:, 1]]
            w = pi * (1 - pj) + pj * (1 - pi)
            lam = np.clip((0.5 - pi) / np.where(np.abs(pj - pi) < 1e-6, 1e-6, pj - pi), 0, 1)
            X = V_true[e[:, 0]] + lam[:, None] * (V_true[e[:, 1]] - V_true[e[:, 0]])
            pg = dict(pos=d["pos"], edge_index=d["ei"], radius=d["radius"], meta=dict(root_point=0))
            ring = dense_ring(encode_ostium(d["rim"], pg, H=5), pg)
            from scipy.spatial import cKDTree
            keep = w > 0.2
            fwd = float(np.sum(w * cKDTree(ring).query(X)[0] ** 2) / w.sum())
            bwd = float(np.mean(cKDTree(X[keep]).query(ring)[0] ** 2)) if keep.any() else float("nan")
            res["S5"].append(dict(case=d["case"], dome_agree=float(((p > 0.5) == dome_lab).mean()),
                                  rim_to_ring_rms_mm=fwd ** 0.5, ring_to_rim_rms_mm=bwd ** 0.5))
        print("S5 done", flush=True)
    except Exception as ex:  # report, don't hide
        res["S5_error"] = repr(ex)[:400]
        print("S5 failed:", repr(ex)[:300], flush=True)

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    report(res)
    figures(ds, rng)


def pct(v, q=(50, 95, 100)):
    v = np.asarray(v, float)
    return " / ".join(f"{np.percentile(v, x):.3g}" for x in q)


def report(res):
    S1, S2, S3, S4, S5 = (res[k] for k in ("S1", "S2", "S3", "S4", "S5"))
    L = [f"# Data sanity checks ({res['n_cases']} cases)", "",
         "Each row is median / 95th percentile / max over cases (and draws).", "",
         "| check | what | result |", "|---|---|---|",
         f"| S1 | H=5 ring vs raw rim, mean distance (mm) | {pct([r['mean'] for r in S1])} |",
         f"| S1 | H=5 ring vs raw rim, max distance (mm) | {pct([r['max'] for r in S1])} |",
         f"| S2 | rotation: code in rotated scene vs rotated ring, max (mm) | {pct([r['rot'] for r in S2])} |",
         f"| S2 | sigma flip: same physical ring, max (mm) | {pct([r['sigma'] for r in S2])} |",
         f"| S2 | site jitter (0.3 mm): same physical ring, max (mm) | {pct([r['jitter'] for r in S2])} |",
         f"| S2 | trimming keeps every point near the ostium | {sum(r['kept_all'] for r in S2)}/{len(S2)} |",
         f"| S2 | crop is one connected piece | {sum(r['components'] == 1 for r in S2)}/{len(S2)} |",
         f"| S2 | crop size (fraction of the graph) | {pct([r['crop_frac'] for r in S2], (5, 50, 95))} (5th/50th/95th) |",
         f"| S2 | site centrality (0 = site nearest the crop centre, 0.5 = typical point) | {pct([r['centrality'] for r in S2], (5, 50, 95))} (5th/50th/95th) |",
         f"| S3 | 26-number target with true phi -> placed fit, max (mm) | {pct([r['exact_max_mm'] for r in S3])} |",
         f"| S3 | via VAE decode of z, mean vertex error (mm) | {pct([r['vae_mean_mm'] for r in S3])} |",
         f"| S3 | via VAE decode of z, max vertex error (mm) | {pct([r['vae_max_mm'] for r in S3])} |",
         f"| S4 | site label sums to 1 / peaks at P / eigenvalues ok / RW ok | "
         f"{sum(abs(r['q_sum'] - 1) < 1e-5 for r in S4)}/{sum(r['argmax_is_site'] for r in S4)}/"
         f"{sum(r['eig_ok'] for r in S4)}/{sum(r['rw_ok'] for r in S4)} of {len(S4)} |",
         f"| S4 | points per level (0 / 1 / 2), median | {np.median([r['n0'] for r in S4]):.0f} / {np.median([r['n1'] for r in S4]):.0f} / {np.median([r['n2'] for r in S4]):.0f} |"]
    if S5:
        L += [f"| S5 | sensor dome vs dome label, vertex agreement | {pct([r['dome_agree'] for r in S5])} |",
              f"| S5 | soft rim -> target ring, rms (mm) | {pct([r['rim_to_ring_rms_mm'] for r in S5])} |",
              f"| S5 | target ring -> soft rim, rms (mm) | {pct([r['ring_to_rim_rms_mm'] for r in S5])} |"]
    elif "S5_error" in res:
        L += [f"| S5 | sensor | failed: {res['S5_error'][:150]} |"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figures(ds, rng, n_fig=6):
    """Training samples as the model sees them: crop (radius), site, target ring, placed complex."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from geometry.complex_target import place_from_frame
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    mc = MultiCanonicalGHDReconstruct(str(REPO / "dataset" / "canonical"))
    idx = rng.choice(len(ds), size=min(n_fig, len(ds)), replace=False)
    fig = plt.figure(figsize=(5 * len(idx), 10))
    for col, i in enumerate(idx):
        s_ = ds[int(i)]
        pos = s_.pos.numpy() * ds.unit
        ring = s_.ring_img.numpy()[0] * ds.unit
        R = s_.frame.numpy()[0]
        C = s_.C.numpy()[0] * ds.unit
        sc = float(s_.scale) * ds.unit
        rec = mc.get(int(s_.atype))
        ph = torch.as_tensor(s_.phi.numpy(), device=rec.GHD_eigvec.device).reshape(1, -1)
        Vc = rec.ghd_forward_as_Meshes(ph, denormalize_shape=False).verts_packed().detach().cpu().numpy()
        V = place_from_frame(Vc, float(s_.log_scale), s_.x_cx.numpy()[0][16:], C, R, sc)
        for row, proj in enumerate(((0, 1), (0, 2))):
            ax = fig.add_subplot(2, len(idx), row * len(idx) + col + 1)
            ax.scatter(pos[:, proj[0]], pos[:, proj[1]], c=s_.radius.numpy(), s=1, cmap="turbo")
            ax.scatter(V[::3, proj[0]], V[::3, proj[1]], s=0.3, c="0.6", alpha=0.4)
            ax.plot(np.r_[ring[:, proj[0]], ring[:1, proj[0]]], np.r_[ring[:, proj[1]], ring[:1, proj[1]]], "k-", lw=1.5)
            p = pos[int(s_.site)]
            ax.plot(p[proj[0]], p[proj[1]], "r*", ms=10)
            ax.set_aspect("equal")
            ax.set_title(f"{s_.case[:22]}\n{'xy' if row == 0 else 'xz'}; sigma={int(s_.sigma)}", fontsize=8)
    fig.suptitle("training samples: crop coloured by radius, site (red *), target ring (black), placed GHD complex from the 26-number target (grey)")
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=110)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
