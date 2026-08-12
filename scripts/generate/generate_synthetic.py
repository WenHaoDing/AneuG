"""
Export spec-compliant synthetic-aneurysm training bundles from the
Fourier branch MLP-VAE generator.

Samples a GHD VAE + branch Fourier MLP-VAE end to end (via utils.generate_synthetic
and utils.export_synthetic) and writes, per accepted sample, the file bundle defined
in `SYNTHETIC_DATA_SPEC.md` so the existing 8-view preprocessing pipeline can run on
synthetic data unchanged. All dome-extraction, bundle-writing, self-intersection,
visualization and validation code lives in utils/export_synthetic.py; this script is
just the sampling/rejection loop and its CLI.

Per-sample folder ``{out}/synth_{NNNNN}_aneurysm1/`` contains:

    ghd_coefficients.npz     phi[144,3] f32, w_rot[3] f32, log_scale () f64, t_vec[3] f32
    ghd_fitted.obj           dome GHD mesh, NORMALIZED (canonical) frame  (full faces)
    ghd_fitted_world.obj     same mesh in SYNTH-WORLD mm  (== "cropped" dome-only mesh)
    merge_mesh_world.obj     full fused dome+vessel mesh in SYNTH-WORLD mm
    dome_face_ids.npy        (K,) int64 face indices (into the .obj face list) of the dome
    landmarks.npz            neck_pt_world, R_frame, dome_centroid_world, aneu_type
    metrics.json             {"s_can": ...}
    merged_centerline.npy    {"branches":[{group,pts,radii,arc_length}], "connected_branch_ids":[...]}

Coordinate model (see SYNTHETIC_DATA_SPEC.md "two frames"):

    verts_world = (verts_canonical * s_can) @ R_frame + neck_pt_world

For synthetics we use ``R_frame = I``, ``neck_pt_world = 0``, and
``s_can = recon.norm_canonical``. The transformer pipeline already produces
geometry in *physical GHD space* (canonical x norm_canonical), so:

    verts_world      = multi_recon._reconstruct_verts_np(phi, atype)   # physical == world
    verts_canonical  = verts_world / s_can
    branch curves    = already in physical/world space (no extra scaling)

GHD fit-corrections are left at identity (w_rot=0, log_scale=0, t_vec=0): the
new fused-mesh path does not apply the VAE scalar, so baking it into only the
dome mesh would desync it from the merged mesh. This keeps every mesh in one
frame and satisfies the spec round-trip ``world == canonical * s_can`` exactly.

Usage:
    conda activate new
    python scripts/generate/generate_synthetic.py --out v2/synthetic_v1 --n 50 --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from utils.generate_synthetic import (
    load_ghd_vae, load_branch_vae,
    _sample_ghd as sample_ghd,
    _build_conditions as build_conditions,
    _branch_centerlines as branch_centerlines,
)
from utils.export_synthetic import (
    ANEU_TYPE_STR, RealPhiGMM, reconstruct_world_dome, count_self_intersections,
    self_intersection_frac, save_training_sample, visualize_sample, validate_sample,
)

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
DEVICE = "cuda:0"

# Checkpoints for eval_branch_mlp_vae.py predate the "args" convention that
# utils.generate_synthetic's loaders require, so this uses a known-working pair
# from tr_checkpoints/v2 instead.
GHD_VAE_CKPT    = ROOT / "tr_checkpoints" / "v2" / "stage1" / "ghd_vae_h512_z16_kl2" / "epoch_01000.pth"
BRANCH_VAE_CKPT = ROOT / "tr_checkpoints" / "v2" / "stage2" / "branch_mlp_vae" / "branch_mlp_vae_gcn_h512_l4_z4_kl5" / "epoch_00500.pth"

PRESENCE_THRESH = 0.5      # branch kept (tubed) iff predicted presence > this


@torch.no_grad()
def export(out_dir, n=50, batch=12, seed=1, save_merged=True, validate=True,
           estimate_radii=False, viz=True, sample_z=True,
           min_branches=1, reject_invalid=True, max_attempts_factor=5,
           compute_selfx=True, max_selfx_frac=None, max_dome_selfx=0,
           force_type=None, start_index=0, ghd_amp=1.0, branch_amp=1.0,
           phi_source="vae", gmm_k=40, real_root="dataset/processed",
           extrude_length=3.0, min_branch_arc=3.0, fuse_smooth=True,
           min_torsion=True, smooth_n_rings=3, smooth_n_iter=10, smooth_lam=0.5,
           planarize_n_iter=10, planarize_lam=0.5):
    """Generate `n` ACCEPTED samples and write spec bundles under `out_dir`.

    sample_z:    True  → branch latent z ~ N(0,I) per sample (diverse vessels);
                 False → z = 0 (deterministic mean/most-likely branches).
    force_type:  None → random type per sample; int → all samples this aneurysm
                 type (0=bifurcation, 1/2=sidewall). Use to build per-type batches.
    start_index: folder numbering offset, so per-type runs can share one out_dir
                 without name collisions (synth_{start_index+i:05d}_aneurysm1).

    Rejection-resampling (a rejected candidate is not written and not counted;
    a new one is drawn until `n` are accepted or the attempt cap is hit):
      min_branches:   reject (cheaply, before any mesh work) candidates with
                      fewer than this many present branches — a sample with no
                      vessel reaching the dome is unusable (`connected_branch_ids`
                      would be empty, failing the spec).
      max_dome_selfx: reject (before fusion) GHD domes whose own surface self-
                      intersects beyond this many triangle pairs. Clean domes
                      score exactly 0, so the default 0 drops only broken shapes
                      with no false positives. None disables the check.
      max_selfx_frac: if set, reject samples whose self-intersecting-face fraction
                      exceeds this (calibrate against a labelled subset first).
      reject_invalid: also delete + resample any written sample that fails
                      `validate_sample` (e.g. fusion produced a non-watertight or
                      missing merged mesh). Requires `validate=True`.
      max_attempts_factor: cap total draws at `n * factor` to avoid looping
                      forever if the model rarely produces valid samples.

    Mesh fusion (forwarded to MultiCanonicalGHDReconstruct.reconstruct_fused_mesh
    for the merged mesh only; the dome itself is unaffected):
      extrude_length, min_branch_arc: stub length / minimum arc for absent or
                      short branches.
      min_torsion:    use a rotation-minimizing (parallel-transport) frame along
                      each branch tube instead of a fixed-reference frame, so the
                      cross-section doesn't pick up artificial twist.
      fuse_smooth, smooth_n_rings, smooth_n_iter, smooth_lam: post-fusion
                      smoothing at the branch/dome graft.
      planarize_n_iter, planarize_lam: opening-ring planarization before fusion.
    """
    import shutil

    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
    ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
    model = load_branch_vae(BRANCH_VAE_CKPT, multi_recon, DEVICE)
    tstr = "random" if force_type is None else f"{force_type} ({ANEU_TYPE_STR.get(force_type, '?')})"
    print(f"Branch MLP-VAE: {BRANCH_VAE_CKPT}\nGHD VAE: {GHD_VAE_CKPT}\nType: {tstr}\nphi_source: {phi_source}")
    gmm = RealPhiGMM(real_root, k=gmm_k) if phi_source == "gmm" else None
    gmm_rng = np.random.default_rng(seed)

    idx = 0                 # accepted-sample counter (drives folder names)
    attempts = 0            # total candidates drawn
    attempt_cap = max(n * max_attempts_factor, n + 50)
    rejected = {"no_branch": 0, "dome_defect": 0, "self_intersect": 0, "invalid": 0, "error": 0}
    all_issues = {}

    while idx < n and attempts < attempt_cap:
        B = min(batch, attempt_cap - attempts)
        if force_type is None:
            types = torch.randint(0, ghd_vae.num_types, (B,), device=DEVICE)
        else:
            types = torch.full((B,), int(force_type), dtype=torch.long, device=DEVICE)
        # GHD dome shape.
        if phi_source == "gmm":
            # phi sampled from the real distribution (matches real spread); scale
            # still from the VAE (unit-correct for branch conditioning).
            _, scale = sample_ghd(ghd_vae, ghd_mean, ghd_std, ghd_input_dim, types)
            phi_np = np.concatenate([gmm.sample(int(t), 1, gmm_rng) for t in types.tolist()], 0)
            phi = torch.from_numpy(phi_np).to(DEVICE)
        else:
            # ghd_amp scales the VAE latent (temperature); 1.0 == prior. NB: >1 adds
            # off-manifold shapes and lowers real-axis spread — prefer phi_source=gmm
            # for variety.
            phi, scale = sample_ghd(ghd_vae, ghd_mean, ghd_std, ghd_input_dim, types, z_amp=ghd_amp)
        cond = build_conditions(multi_recon, phi, types, scale, model.max_branches)
        # Fourier MLP-VAE: per-branch chord vector, Fourier coeffs, presence prob.
        # branch_amp scales the branch latent likewise.
        if sample_z:
            z = torch.randn(B, model.latent_dim, device=DEVICE) * branch_amp
        else:
            z = torch.zeros(B, model.latent_dim, device=DEVICE)
        vec, coeff, pres = model.sample(phi, cond, z=z)
        starts = cond.start_points.cpu().numpy()
        vec, coeff, pres = vec.cpu().numpy(), coeff.cpu().numpy(), pres.cpu().numpy()
        mask = cond.branch_mask.cpu().numpy()

        for i in range(B):
            if idx >= n:
                break
            attempts += 1
            atype = int(types[i])
            # present openings → Fourier centerline; absent (pres ≤ thresh) or not a
            # candidate opening for this type → None (3mm stub)
            curves = branch_centerlines(starts[i], vec[i], coeff[i], pres[i],
                                        mask[i], PRESENCE_THRESH)
            branch_mask = [c is not None for c in curves]

            # cheap pre-save reject: a sample with no vessel is unusable
            if sum(branch_mask) < min_branches:
                rejected["no_branch"] += 1
                continue

            # dome-quality reject: the GHD dome itself must not self-intersect
            # (a broken training target). Clean domes score exactly 0, so this is
            # a binary, false-positive-free filter. Checked before fusion to skip
            # the expensive merge on a doomed sample.
            dome_sx = None
            if max_dome_selfx is not None:
                dverts, _, dfaces, _ = reconstruct_world_dome(multi_recon, phi[i], atype)
                dome_sx = count_self_intersections(dverts, dfaces)
                if dome_sx > max_dome_selfx:
                    rejected["dome_defect"] += 1
                    continue

            name = f"synth_{start_index + idx:05d}_aneurysm1"
            sample_dir = out_dir / name
            try:
                save_training_sample(
                    sample_dir, phi[i], atype, curves, branch_mask,
                    multi_recon, save_merged=save_merged, estimate_radii=estimate_radii,
                    extrude_length=extrude_length, min_branch_arc=min_branch_arc,
                    fuse_smooth=fuse_smooth, min_torsion=min_torsion,
                    smooth_n_rings=smooth_n_rings, smooth_n_iter=smooth_n_iter,
                    smooth_lam=smooth_lam, planarize_n_iter=planarize_n_iter,
                    planarize_lam=planarize_lam,
                )
                if viz:
                    try:
                        visualize_sample(sample_dir)
                    except Exception as exc:
                        print(f"[warn] {name}: viz failed ({exc})")
                # ground-truth defect: mesh self-intersection (tube-tube crossing
                # or tube-dome poke-through). Recorded for every sample so the
                # threshold can be calibrated against a manually-labelled subset.
                selfx = None
                if compute_selfx:
                    selfx = self_intersection_frac(sample_dir / "merge_mesh_world.obj")
                if selfx is not None or dome_sx is not None:
                    meta = json.loads((sample_dir / "metrics.json").read_text())
                    if selfx is not None:
                        meta["selfx_frac"] = selfx
                    if dome_sx is not None:
                        meta["dome_selfx"] = int(dome_sx)
                    (sample_dir / "metrics.json").write_text(json.dumps(meta))
                if (max_selfx_frac is not None and selfx is not None
                        and selfx > max_selfx_frac):
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    rejected["self_intersect"] += 1
                    print(f"[reject] {name}: selfx_frac={selfx:.3f} > {max_selfx_frac}")
                    continue

                issues = validate_sample(sample_dir) if validate else []
                if issues and reject_invalid:
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    rejected["invalid"] += 1
                    print(f"[reject] {name}: {issues}")
                    continue
                sx = f"  selfx={selfx:.3f}" if selfx is not None else ""
                if issues:
                    all_issues[name] = issues
                    print(f"[FAIL] {name}: {issues}{sx}")
                else:
                    print(f"[ok]   {name}  (type {atype}){sx}")
            except Exception as exc:
                shutil.rmtree(sample_dir, ignore_errors=True)
                rejected["error"] += 1
                print(f"[reject] {name}: exception: {exc}")
                continue
            idx += 1

    print(f"\nDone: {idx}/{n} accepted in {attempts} draws → {out_dir}")
    print(f"Rejected — no-branch: {rejected['no_branch']}, "
          f"dome-defect: {rejected['dome_defect']}, "
          f"self-intersect: {rejected['self_intersect']}, "
          f"invalid: {rejected['invalid']}, error: {rejected['error']}")
    if idx < n:
        print(f"[warn] hit attempt cap ({attempt_cap}); only {idx}/{n} accepted. "
              "Lower PRESENCE_THRESH, raise max_attempts_factor, or check the model.")
    if all_issues:
        print(f"{len(all_issues)} written sample(s) still flagged (reject_invalid off).")
    return all_issues


def _parse_args():
    p = argparse.ArgumentParser(description="Export synthetic aneurysm training bundles.")
    p.add_argument("--out", type=str, default=str(ROOT / "v2" / "synthetic_v1"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-merged", action="store_true", help="skip merge_mesh_world.obj")
    p.add_argument("--no-validate", action="store_true", help="skip per-sample validation")
    p.add_argument("--no-viz", action="store_true", help="skip per-sample sanity_check.png")
    p.add_argument("--estimate-radii", action="store_true", help="store opening-ring radii")
    p.add_argument("--mean-branches", action="store_true",
                   help="z=0 deterministic branches instead of z~N(0,I)")
    p.add_argument("--min-branches", type=int, default=1,
                   help="reject+resample candidates with fewer present branches")
    p.add_argument("--keep-invalid", action="store_true",
                   help="keep (don't resample) samples that fail validation")
    p.add_argument("--max-attempts-factor", type=int, default=5,
                   help="cap total draws at n * this factor")
    p.add_argument("--type", type=int, default=None,
                   help="fix aneurysm type for all samples (0=bifurcation, 1/2=sidewall); "
                        "default random")
    p.add_argument("--start-index", type=int, default=0,
                   help="folder numbering offset (for per-type runs into one out dir)")
    p.add_argument("--ghd-amp", type=float, default=1.0,
                   help="VAE dome latent temperature (>1 adds off-manifold shapes; "
                        "prefer --phi-source gmm for variety)")
    p.add_argument("--branch-amp", type=float, default=1.0,
                   help="branch latent temperature (>1 = more diverse vessels, more defects)")
    p.add_argument("--phi-source", choices=["vae", "gmm"], default="vae",
                   help="vae = N(0,I) prior (under-covers real); gmm = Gaussian fit to "
                        "real phi (matches real spread)")
    p.add_argument("--gmm-k", type=int, default=40, help="PCA dims for the real-phi Gaussian")
    p.add_argument("--real-root", type=str, default="dataset/processed",
                   help="real phi source for --phi-source gmm")
    p.add_argument("--no-selfx", action="store_true",
                   help="skip mesh self-intersection measurement (recorded in metrics.json)")
    p.add_argument("--max-selfx-frac", type=float, default=None,
                   help="reject samples whose self-intersecting-face fraction exceeds this "
                        "(calibrate against a manually-labelled subset first; e.g. 0.005)")
    p.add_argument("--max-dome-selfx", type=int, default=0,
                   help="reject GHD domes with more than this many self-intersecting "
                        "triangle pairs (0 = drop any broken dome; -1 disables)")
    p.add_argument("--extrude-length", type=float, default=3.0,
                   help="stub length (mm) for absent branches")
    p.add_argument("--min-branch-arc", type=float, default=3.0,
                   help="minimum branch arc length (mm)")
    p.add_argument("--no-fuse-smooth", action="store_true",
                   help="skip post-fusion smoothing at the branch/dome graft")
    p.add_argument("--no-min-torsion", action="store_true",
                   help="use a fixed-reference tube frame instead of the rotation-minimizing "
                        "(parallel-transport) one; the latter avoids artificial cross-section "
                        "twist and is on by default")
    p.add_argument("--smooth-n-rings", type=int, default=3,
                   help="graft-region smoothing: ring neighborhood size")
    p.add_argument("--smooth-n-iter", type=int, default=10,
                   help="graft-region smoothing: iteration count")
    p.add_argument("--smooth-lam", type=float, default=0.5,
                   help="graft-region smoothing: lambda (step size)")
    p.add_argument("--planarize-n-iter", type=int, default=10,
                   help="opening-ring planarization: iteration count (pre-fusion)")
    p.add_argument("--planarize-lam", type=float, default=0.5,
                   help="opening-ring planarization: lambda (step size)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args.out, n=args.n, batch=args.batch, seed=args.seed,
           save_merged=not args.no_merged, validate=not args.no_validate,
           estimate_radii=args.estimate_radii, viz=not args.no_viz,
           sample_z=not args.mean_branches,
           min_branches=args.min_branches, reject_invalid=not args.keep_invalid,
           max_attempts_factor=args.max_attempts_factor,
           compute_selfx=not args.no_selfx, max_selfx_frac=args.max_selfx_frac,
           max_dome_selfx=(None if args.max_dome_selfx < 0 else args.max_dome_selfx),
           force_type=args.type, start_index=args.start_index,
           ghd_amp=args.ghd_amp, branch_amp=args.branch_amp,
           phi_source=args.phi_source, gmm_k=args.gmm_k, real_root=args.real_root,
           extrude_length=args.extrude_length, min_branch_arc=args.min_branch_arc,
           fuse_smooth=not args.no_fuse_smooth, min_torsion=not args.no_min_torsion,
           smooth_n_rings=args.smooth_n_rings, smooth_n_iter=args.smooth_n_iter,
           smooth_lam=args.smooth_lam, planarize_n_iter=args.planarize_n_iter,
           planarize_lam=args.planarize_lam)
