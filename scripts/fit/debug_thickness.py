#!/usr/bin/env python3
"""Reproduce the mesh_thickness_loss crash on one case, with checkpoints.

WHY THIS IS ODD: MeshThickness runs on the DEFORMED CANONICAL -- fixed
topology (4143 verts / 8282 faces), and every index inside it comes from
argsort, so indices are bounded by construction. An out-of-range index should
be impossible. Two explanations survive that:

  1. NaN/Inf in the mesh. ec_dist is divided by its own std; a degenerate
     deformation makes that 0, and NaN propagates into the sort keys.
  2. The assert is ASYNCHRONOUS. CUDA reports a device-side assert at the next
     synchronising call, so the traceback names this function while the real
     fault is an earlier kernel. CUDA_LAUNCH_BLOCKING=1 forces the report to
     land where it actually happens.

So: run with CUDA_LAUNCH_BLOCKING=1, and checkpoint the live mesh + phi every
iteration so the state at the failing step can be inspected afterwards.

  python scripts/fit/debug_thickness.py --case C0030_cut2 --dataset AneuX
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"      # must precede torch import
os.environ.pop("DISPLAY", None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ghd.fitting.losses.mesh_thickness_loss import MeshThickness  # noqa: E402


def probe(mesh, tag, out_dir, it):
    """Everything that could make the sort keys meaningless."""
    V = mesh.verts_packed()
    F = mesh.faces_packed()
    rep = {
        "iter": it, "n_verts": int(V.shape[0]), "n_faces": int(F.shape[0]),
        "verts_nan": int(torch.isnan(V).sum()), "verts_inf": int(torch.isinf(V).sum()),
        "verts_absmax": float(V.abs().max()),
    }
    vn = mesh.verts_normals_packed()
    fn = mesh.faces_normals_packed()
    rep["vert_normal_nan"] = int(torch.isnan(vn).sum())
    rep["face_normal_nan"] = int(torch.isnan(fn).sum())
    # degenerate faces -> zero-area -> undefined normal -> NaN in the sort keys
    tri = V[F]
    area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=-1) / 2
    rep["zero_area_faces"] = int((area < 1e-12).sum())
    rep["min_face_area"] = float(area.min())
    # the exact quantity the loss divides by
    centro = tri.mean(dim=-2)
    ec = torch.cdist(V, centro)
    std = ec.std(dim=-1)
    rep["ec_std_min"] = float(std.min())
    rep["ec_std_zero_rows"] = int((std < 1e-20).sum())
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--dataset", default="AneuX")
    ap.add_argument("--out", default=str(ROOT / "runtime" / "debug_thickness"))
    ap.add_argument("--n-iter", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=int, default=100, help="checkpoint interval")
    args = ap.parse_args()

    out_dir = Path(args.out) / f"{args.dataset}__{args.case}"
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    print(f"CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}  -> tracebacks are exact")
    print(f"checkpoints -> {out_dir}\n")

    from ghd.fitting.alignment import load_case_data, load_canonical_data, fit_alignment, \
        save_stage1_checkpoints, DOME_LOADERS
    from ghd.fitting.ghd_fit import load_target
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    from pytorch3d.structures import Meshes
    from pytorch3d.transforms import axis_angle_to_matrix

    dev = torch.device(args.device)
    geo = Path("/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry") / args.dataset / args.case
    sys.path.insert(0, str(ROOT / "scripts" / "fit"))
    from fit_configs import dome_source_for
    case = load_case_data(geo, dome_points_loader=DOME_LOADERS[dome_source_for(args.dataset)])
    atype = case["aneurysm_type"]
    canon_dir = ROOT / "dataset" / "canonical" / ("Bifurcated" if atype == 0 else "Sidewall")
    canonical = load_canonical_data(canon_dir)
    transform, _ = fit_alignment(case, canonical, epochs=800, device=str(dev), log_every=800)
    save_stage1_checkpoints(out_dir, transform, case, canonical)

    mc = MultiCanonicalGHDReconstruct(ROOT / "dataset" / "canonical", device=dev)
    recon = mc.get(atype)
    V0, Fc, U = recon.canonical_Meshes.verts_packed(), recon.canonical_Meshes.faces_packed(), recon.GHD_eigvec
    target = load_target(out_dir, recon.norm_canonical, n_surface_samples=20000)

    phi = torch.nn.Parameter(torch.zeros(U.shape[1], 3, device=dev))
    w = torch.nn.Parameter(torch.zeros(3, device=dev))
    log_s = torch.nn.Parameter(torch.zeros((), device=dev))
    t = torch.nn.Parameter(torch.zeros(3, device=dev))
    thickness = MeshThickness(r=0.2)

    def render():
        Vd = V0 + torch.einsum("nm,mc->nc", U, phi)
        R = axis_angle_to_matrix(w.unsqueeze(0)).squeeze(0)
        return Meshes(verts=[(Vd @ R.T) * torch.exp(log_s) + t], faces=[Fc])

    opt = torch.optim.Adam([phi, w, log_s, t], lr=1e-3)
    reports = []
    print(f"{'iter':>6} {'nan':>5} {'zeroA':>6} {'minA':>10} {'ec_std_min':>12} {'absmax':>9}")
    for it in range(args.n_iter):
        mesh = render()
        if it % args.every == 0:
            rep = probe(mesh, "live", out_dir, it)
            reports.append(rep)
            print(f"{it:6d} {rep['verts_nan']:5d} {rep['zero_area_faces']:6d} "
                  f"{rep['min_face_area']:10.3e} {rep['ec_std_min']:12.3e} {rep['verts_absmax']:9.3f}",
                  flush=True)
            np.savez(out_dir / "meshes" / f"state_{it:05d}.npz",
                     verts=mesh.verts_packed().detach().cpu().numpy(),
                     phi=phi.detach().cpu().numpy())
        try:
            dist, dist_v_norm, _, sign = thickness(mesh)
        except Exception:
            print(f"\n!!! thickness FAILED at iter {it}\n")
            traceback.print_exc()
            rep = probe(mesh, "fail", out_dir, it)
            print("\nstate at failure:", rep)
            np.savez(out_dir / "FAILURE_state.npz",
                     verts=mesh.verts_packed().detach().cpu().numpy(),
                     phi=phi.detach().cpu().numpy(), iter=it)
            trimesh.Trimesh(vertices=mesh.verts_packed().detach().cpu().numpy(),
                            faces=Fc.cpu().numpy(), process=False).export(out_dir / "FAILURE_mesh.obj")
            np.savez(out_dir / "probe_history.npz",
                     **{k: np.array([r[k] for r in reports]) for k in reports[0]})
            return
        loss = dist.mean() + mesh.verts_packed().pow(2).mean() * 0
        opt.zero_grad(); loss.backward(); opt.step()

    print("\ncompleted without failing")
    np.savez(out_dir / "probe_history.npz",
             **{k: np.array([r[k] for r in reports]) for k in reports[0]})


if __name__ == "__main__":
    main()
