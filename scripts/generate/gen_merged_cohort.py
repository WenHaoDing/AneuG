"""Generate a cohort of merged aneurysm meshes, one folder per shape.

THE WHOLE PIPELINE, END TO END. Stage-1 GHD VAE draws a dome from z ~ N(0,1);
the morphology sensor finds each opening's cap and outward tangent; the branch
transformer generates a centerline per opening; each centerline is trimmed to
length and given a straight run at its far end; the sweep lofts a tube along it
and merges into the trimmed dome.

Branch 0 is the INLET and every other branch an outlet, so a bifurcated shape has
two outlets and a sidewall one. By default 7.5 mm of centerline is kept for the
inlet and 3 mm for each outlet, then 5 mm and 2.5 mm of straight run are added
past those ends, so the mesh terminates straight rather than mid-bend. An inlet
generated shorter than 5 mm has the shortfall added to its straight run, so every
inlet is at least 10 mm long. These are
short on purpose: the meshes go to CFD, where every extra millimetre costs
solver time.

Self-intersecting shapes are DISCARDED and redrawn. Such a surface has no
well-defined inside, so volume meshing and CFD both fail on it, and no repair
tool recovers it. Every other finding -- holes left unfixed, ring folding, a
curvature margin not met -- is logged and the shape is kept, on the basis that
downstream repair handles those.

Each shape gets its own subfolder:

    <case>/ghd.npy                       [144, 3] GHD coefficients
           z.npy                         [z_dim]  the stage-1 latent that produced it
           mesh.obj                      the merged surface (or .vtp with --vtp)
           dome.obj                      stage-1 only, before any branch was attached
           openings.npy                  dict: opening centroids, normals, radii, roles
           ghd_forward_fusion_info.npz   the same openings in the volume mesher's schema
           centerlines.npy               the RAW generated branches, before trimming
           meta.json                     provenance, diagnostics, warnings

OPENINGS ARE IN BRANCH ORDER: index 0 is the inlet, the rest are outlets. Each
opening is the centroid of an open boundary measured on the FINISHED mesh -- the
tube-end cap a solver imposes its boundary condition on -- and matched to its
branch by the swept centerline's endpoint.

ghd_forward_fusion_info.npz carries what cfd_mesher/generate_cfd_volume_meshes.py
reads: opening_centroids (inlet first) and cpcd_glo, whose last point per branch
is the cap reference it relabels by. The mesher reads geomagic_processed.obj, the
surface after Geomagic, so each case needs that file added before meshing:

    python -m cfd_mesher.generate_cfd_volume_meshes \
        --surface_root runtime_dataset/generation/<cohort>

    python scripts/generate/gen_merged_cohort.py --n 20 --device cuda:0

Every merged mesh is checked for ring folding (the sweep crossing itself on a
sharp turn), holes left unfixed, triangle-pair self-intersection, and whether
each opening matched its branch cleanly. Findings are printed and recorded in
meta.json.

Shapes whose caps do not close are skipped rather than written: a cap whose
boundary splits in two cannot be lofted onto, and the tube swept from it
collapses. The count and the reasons land in the cohort's manifest.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from models.morphoformer import SensorUncapper
from models.branch_transformer import MultiBranchVAE_GCNConditioner, MultiBranchConditions
from utils.generate_synthetic import load_ghd_vae
from utils.mesh_fusion import MergedMeshBuilder

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
OUT_ROOT = ROOT / "runtime_dataset" / "generation"

GHD_VAE_GAN = (ROOT / "runtime_train/ghd_vae/stage1"
               / "ghd_vae_gan_h256_z16_kl2_adv0.3" / "epoch_05000.pth")
GHD_VAE_PLAIN = (ROOT / "runtime_train/ghd_vae/stage1"
                 / "ghd_vae_h256_z16_kl2" / "epoch_05000.pth")
SENSOR = (ROOT / "runtime_train/morphoformer/morphology_sensor"
          / "h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5_prob_z8_kl0.1" / "epoch_02000.pth")
BRANCH_RUN = ROOT / "runtime_train/branch_transformer/branch_transformer_gcn_h64_z8_kl1_lr3e-4_tan1"

# branch transformer architecture (matches scripts/train/train_branch_transformer.py)
MAX_BRANCHES = 3
MAX_POINTS_PER_BRANCH = 128
NUM_TYPES = 3
HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, NHEAD = 64, 8, 4, 4
GHD_DIM, GCN_HIDDEN, GCN_POOL_RATIO = 16, 32, 0.5
NUM_TYPES_GEN = 2          # type 2 shares the sidewall canonical with type 1


def latest_checkpoint(run_dir):
    cks = sorted(Path(run_dir).glob("epoch_*.pth"))
    if not cks:
        raise FileNotFoundError(f"no epoch_*.pth in {run_dir}")
    return cks[-1]


def load_branch_model(ckpt, multi_recon, device):
    ck = torch.load(ckpt, map_location=device)
    model = MultiBranchVAE_GCNConditioner(
        multi_recon, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS, nhead=NHEAD,
        max_local_points=MAX_POINTS_PER_BRANCH - 1, num_types=NUM_TYPES,
        max_branches=MAX_BRANCHES, ghd_dim=GHD_DIM,
        gcn_hidden=GCN_HIDDEN, gcn_pool_ratio=GCN_POOL_RATIO).to(device)
    model.load_state_dict(ck["model"])
    model.set_normalization_stats(ck["point_mean"], ck["point_std"])
    model.eval()
    return model, int(ck.get("epoch", -1))


def branch_curves(offsets, lengths, starts, mask, point_mean, point_std):
    """Absolute centerline per branch: start + denormalized offsets[:n].

    Entry b is None when the branch is absent, which the builder turns into a
    short straight stub rather than a tube.
    """
    L = MAX_POINTS_PER_BRANCH - 1
    offs = offsets.view(MAX_BRANCHES, L, 3) * point_std + point_mean
    out = []
    for b in range(MAX_BRANCHES):
        if not bool(mask[b]):
            out.append(None)
            continue
        n = int(lengths[b].clamp(min=2))
        s = starts[b].detach().cpu().numpy()
        out.append(np.concatenate([s[None], s[None] + offs[b, :n].detach().cpu().numpy()]))
    return out


def order_openings_by_branch(centerlines, boundaries):
    """Assign each branch the open boundary nearest its swept centerline's end.

    Greedy with each boundary used once, in branch order, which is the same rule
    the volume mesher applies when it relabels caps. Returns (boundaries in
    branch order, match distance per branch).
    """
    if len(boundaries) < len(centerlines):
        raise RuntimeError("%d open boundaries for %d branches"
                           % (len(boundaries), len(centerlines)))
    used, ordered, dists = set(), [], []
    for cl in centerlines:
        end = np.asarray(cl)[-1]
        d, i = min((float(np.linalg.norm(np.asarray(b["centroid"]) - end)), i)
                   for i, b in enumerate(boundaries) if i not in used)
        used.add(i); ordered.append(boundaries[i]); dists.append(d)
    return ordered, dists


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=650, help="Shapes to WRITE (failures are redrawn).")
    ap.add_argument("--name", default=None, help="Cohort folder name (default: timestamped).")
    ap.add_argument("--ghd-vae", default=None,
                    help="Stage-1 checkpoint. Default is the WGAN-GP run; pass "
                         "--plain-vae for the plain one.")
    ap.add_argument("--plain-vae", action="store_true")
    ap.add_argument("--sensor", default=str(SENSOR))
    ap.add_argument("--branch-model", default=None,
                    help="Branch transformer checkpoint (default: latest in the tan1 run).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-amp", type=float, default=1.0)
    ap.add_argument("--types", default="balanced", choices=["balanced", "bifurcated", "sidewall"],
                    help="balanced writes exactly half of each type. The split is on shapes "
                         "WRITTEN, not drawn, so a discarded shape is redrawn as the same "
                         "type and the halves stay exact however many are rejected.")
    ap.add_argument("--vtp", action="store_true",
                    help="Write mesh.vtp instead of mesh.obj. OBJ is the default because "
                         "it opens in anything.")
    ap.add_argument("--max-kr", type=float, default=0.45,
                    help="Cap on curvature * tube radius. Rings fold into each other "
                         "as this approaches 1; lower is safer and straighter.")
    ap.add_argument("--seam-rings", type=int, default=6)
    ap.add_argument("--extrude-length", type=float, default=3.0,
                    help="Straight stub for a branch with no usable centerline at all.")
    ap.add_argument("--inlet-length", type=float, default=7.5,
                    help="mm of generated centerline kept for branch 0, the inlet.")
    ap.add_argument("--outlet-length", type=float, default=3.0,
                    help="mm kept for each outlet (every branch after the first).")
    ap.add_argument("--inlet-extrusion", type=float, default=5.0,
                    help="mm of straight run added past the inlet's trimmed end.")
    ap.add_argument("--inlet-min-length", type=float, default=5.0,
                    help="mm of inlet centerline guaranteed before the straight run. A "
                         "shorter generated inlet has the shortfall added to its extrusion, "
                         "so a 2 mm inlet gets 3 mm extra. 0 disables it.")
    ap.add_argument("--outlet-extrusion", type=float, default=2.5,
                    help="mm of straight run added past each outlet's trimmed end.")
    ap.add_argument("--no-refill", dest="refill", action="store_false")
    ap.add_argument("--keep-self-intersecting", action="store_true",
                    help="Write self-intersecting shapes instead of discarding them. Only "
                         "for inspecting what the detector catches: such a surface has no "
                         "well-defined inside, so volume meshing and CFD both fail on it.")
    ap.add_argument("--skip-flagged", action="store_true",
                    help="Also discard shapes that only raise advisory warnings (holes left "
                         "unfixed, ring folding). Off by default: downstream repair tools "
                         "handle those, so they are logged rather than thrown away.")
    ap.add_argument("--no-self-intersection-check", dest="check_si", action="store_false",
                    help="Skip the triangle-pair test; it is the slow part, a few seconds "
                         "per shape.")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ghd_ckpt = Path(args.ghd_vae) if args.ghd_vae else (GHD_VAE_PLAIN if args.plain_vae else GHD_VAE_GAN)
    br_ckpt = Path(args.branch_model) if args.branch_model else latest_checkpoint(BRANCH_RUN)
    out_dir = OUT_ROOT / (args.name or f"cohort_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=dev)
    vae, mean, std, gdim = load_ghd_vae(ghd_ckpt, dev)
    uncapper = SensorUncapper.from_checkpoint(Path(args.sensor), multi_recon, device=dev)
    model, br_epoch = load_branch_model(br_ckpt, multi_recon, dev)
    builder = MergedMeshBuilder(multi_recon, max_kr=args.max_kr, seam_rings=args.seam_rings,
                                extrude_length=args.extrude_length,
                                inlet_max_length=args.inlet_length,
                                outlet_max_length=args.outlet_length,
                                inlet_extrusion=args.inlet_extrusion,
                                outlet_extrusion=args.outlet_extrusion,
                                inlet_min_length=args.inlet_min_length or None,
                                check_self_intersection=args.check_si)
    with_scale = mean.numel() > gdim
    pm, ps = model.point_mean, model.point_std

    print(f"stage-1 VAE : {ghd_ckpt.parent.name}/{ghd_ckpt.name}")
    print(f"sensor      : {Path(args.sensor).parent.name} (probabilistic={uncapper.is_probabilistic})")
    print(f"centerlines : {br_ckpt.parent.name}/{br_ckpt.name} (epoch {br_epoch})")
    print(f"out         : {out_dir}\n")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    written, attempts, kept_flagged = 0, 0, []
    skipped = {"uncap_failed": 0, "fusion_error": 0, "checks_failed": 0,
               "self_intersecting": 0}
    flagged = []
    max_attempts = args.n * 5 if args.refill else args.n

    while written < args.n and attempts < max_attempts:
        attempts += 1
        if args.types == "balanced":
            # keyed on how many are already written, so a rejection redraws the
            # same type rather than eroding the balance
            t = 0 if written < (args.n + 1) // 2 else 1
        else:
            t = 0 if args.types == "bifurcated" else 1
        z = torch.randn(1, vae.latent_dim, generator=g) * args.z_amp
        types = torch.tensor([t], device=dev)
        with torch.no_grad():
            o = vae.decode(z.to(dev), types, strip_scale=True) if with_scale else vae.decode(z.to(dev), types)
            phi_n, scale_n = (o if isinstance(o, tuple) else (o, None))
            m_ = mean[..., :gdim] if with_scale else mean
            s_ = std[..., :gdim] if with_scale else std
            phi = (phi_n * s_ + m_).reshape(1, -1, 3)
            scale = (float((scale_n * std[..., gdim:] + mean[..., gdim:]).squeeze())
                     if with_scale and scale_n is not None else 1.0)

            # openings from the sensor; a shape whose caps do not close is dropped
            u = uncapper.uncap(phi[0], t)
            if not bool(np.asarray(u["rim_ok"], dtype=bool).all()):
                skipped["uncap_failed"] += 1
                continue
            faces = multi_recon.get(t).canonical_Meshes.faces_packed().cpu().numpy()
            n_open = u["cap_masks"].shape[0]
            loops = [uncapper.rim_loop(u["cap_masks"][b], faces) for b in range(n_open)]
            normals = [u["directions"][b] for b in range(n_open)]

            b_mask = torch.zeros(1, MAX_BRANCHES, dtype=torch.bool, device=dev)
            b_mask[0, :n_open] = True
            starts = torch.zeros(1, MAX_BRANCHES, 3, device=dev)
            dirs = torch.zeros(1, MAX_BRANCHES, 3, device=dev)
            starts[0, :n_open] = torch.as_tensor(u["start_points"], dtype=torch.float32, device=dev)
            dirs[0, :n_open] = torch.as_tensor(np.stack(normals), dtype=torch.float32, device=dev)
            cond = MultiBranchConditions(types, torch.ones(1, device=dev), starts, dirs, b_mask)
            offsets, lengths, _ = model.sample(phi, cond)
            curves = branch_curves(offsets[0], lengths[0], starts[0], b_mask[0], pm, ps)

        case = f"{out_dir.name}_{written:04d}_t{t}"
        cdir = out_dir / case
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            res = builder.build(
                phi[0].cpu().numpy(), t, curves, branch_mask=b_mask[0].cpu().numpy(),
                opening_indices=loops, opening_normals=normals,
                trimmed_faces=faces[~np.any([u["cap_masks"][b][faces].all(1)
                                             for b in range(n_open)], axis=0)],
                save_path=cdir / ("mesh.vtp" if args.vtp else "mesh.obj"))
        except Exception as exc:
            skipped["fusion_error"] += 1
            import shutil; shutil.rmtree(cdir, ignore_errors=True)
            print(f"  [skip] {case}: {type(exc).__name__}: {exc}")
            continue

        np.save(cdir / "ghd.npy", phi[0].cpu().numpy().astype(np.float32))
        np.save(cdir / "z.npy", z[0].numpy().astype(np.float32))
        np.save(cdir / "centerlines.npy",
                np.array([c if c is not None else np.zeros((0, 3)) for c in curves], dtype=object),
                allow_pickle=True)
        # ---- openings, ORDERED BY BRANCH ------------------------------------
        # boundary_loops() returns the open boundaries in whatever order the
        # boundary walk met them, which carries no meaning. The volume mesher
        # (cfd_mesher.volume_mesh) assumes index 0 is the INLET and the rest are
        # outlets in branch order, and relabels caps by matching them to those
        # positions. So each branch's swept centerline endpoint is matched to its
        # nearest open boundary here, and everything below is written in branch
        # order: inlet first.
        ordered, match_d = order_openings_by_branch(res["centerlines"], res["boundaries"])
        radii = np.array([b["radius"] for b in ordered])
        roles = ["inlet"] + [f"outlet{k}" for k in range(1, len(ordered))]
        # A branch whose endpoint is further from its matched boundary than that
        # boundary's own radius was not matched cleanly, and the mesher would be
        # relabelling blind on it.
        for k, (dist, rad) in enumerate(zip(match_d, radii)):
            if dist > rad:
                res["diagnostics"]["warnings"].append(
                    "OPENING MATCH UNCERTAIN: %s endpoint sits %.2f mm from its boundary "
                    "centroid, beyond that boundary's %.2f mm radius" % (roles[k], dist, rad))
        res["diagnostics"]["opening_match_distance"] = [round(float(x), 4) for x in match_d]
        res["diagnostics"]["ok"] = not res["diagnostics"]["warnings"]

        # The mesher's relabelling uses each cpcd_glo branch's LAST POINT as the
        # reference for that cap. The swept tube can stop short of its
        # centerline (rings are kept every ds_r-th, so the final ring may be
        # dropped), so the centerline is cut at the measured cap and ends exactly
        # on it -- the reference is then the true cap centroid, not an estimate.
        cpcd_glo, cpcd_tan = [], []
        for k, bnd in enumerate(ordered):
            cl = np.asarray(res["centerlines"][k], dtype=float)
            cap = np.asarray(bnd["centroid"], dtype=float)
            t_end = cl[-1] - cl[max(len(cl) - 3, 0)]
            t_end = t_end / (np.linalg.norm(t_end) + 1e-12)
            keep = ((cl - cap) @ t_end) < 0.0
            body = cl[keep] if keep.sum() >= 2 else cl[:2]
            # Tangents come from the centerline BEFORE the cap point is appended. The
            # cap centroid sits 0.1-0.2 mm to the side of the last centerline point,
            # so differencing across that jog made the final tangent nearly
            # perpendicular to the branch (median 74 deg over 1625 branches in
            # cohort_650) and the one before it 21 deg off. The cap point instead
            # inherits the last centerline tangent, which runs along the branch.
            tan = np.gradient(body, axis=0)
            tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12)
            cl = np.vstack([body, cap[None]])
            tan = np.vstack([tan, tan[-1:]])
            cpcd_glo.append(cl.astype(np.float64))
            cpcd_tan.append(tan.astype(np.float64))

        opening_centroids = np.array([b["centroid"] for b in ordered], dtype=np.float64)
        opening_normals = np.array([b["normal"] for b in ordered], dtype=np.float64)
        # orient each boundary normal OUTWARD, along its branch's terminal direction
        for k in range(len(ordered)):
            if opening_normals[k] @ (cpcd_glo[k][-1] - cpcd_glo[k][-2]) < 0:
                opening_normals[k] = -opening_normals[k]

        # the volume mesher's contract: same file name and keys as the real-data
        # surface stage writes, so it runs on this cohort unchanged
        np.savez(cdir / "ghd_forward_fusion_info.npz",
                 opening_centroids=opening_centroids,
                 opening_normals=opening_normals,
                 opening_radii=radii,
                 cpcd_glo=np.array(cpcd_glo, dtype=object),
                 cpcd_glo_tangent=np.array(cpcd_tan, dtype=object),
                 neck_centroids=np.asarray(res["neck_centroids"], dtype=np.float64),
                 roles=np.array(roles),
                 allow_pickle=True)
        np.save(cdir / "openings.npy", {
            "roles":              roles,                  # index 0 is always the inlet
            "opening_centroids":  opening_centroids,      # final tube-end cap centroids
            "opening_normals":    opening_normals,        # outward
            "opening_radii":      radii,
            "neck_centroids":     np.asarray(res["neck_centroids"]),   # where tube meets dome
            "neck_normals":       np.asarray(res["neck_normals"]),
            "aneurysm_type":      t,
        }, allow_pickle=True)
        # the dome on its own, for comparing against the merged result
        dome = multi_recon._reconstruct_verts_np(phi[0].cpu().numpy(), t)
        with open(cdir / "dome.obj", "w") as fh:
            for v in dome:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f in faces:
                fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
        (cdir / "meta.json").write_text(json.dumps({
            "case": case, "aneurysm_type": t, "scale": scale,
            "ghd_vae": str(ghd_ckpt), "sensor": str(args.sensor),
            "branch_model": str(br_ckpt), "branch_epoch": br_epoch,
            "seed": args.seed, "z_amp": args.z_amp, "attempt": attempts,
            "diagnostics": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                            for k, v in res["diagnostics"].items()},
        }, indent=2, default=float))

        d = res["diagnostics"]
        status = "ok" if d["ok"] else "FLAGGED"
        print(f"  {case}: {d['n_points']} pts, {d['n_open_boundaries']} boundaries, "
              f"fold {d['max_ring_fold_fraction']:.4f}, worst k*r {d['worst_kr']:.2f}, "
              f"drift {d['worst_exit_drift_deg']:.1f} deg, holes closed {d['holes_closed']}"
              f"  [{status}]")
        for w in d["warnings"]:
            print(f"      ! {w}")
        for nline in d.get("notes", []):
            print(f"      . {nline}")
        fatal = d.get("fatal", False) and not args.keep_self_intersecting
        if not d["ok"]:
            flagged.append(case)
            if not fatal and not args.skip_flagged:
                kept_flagged.append(case)
        if fatal or (not d["ok"] and args.skip_flagged):
            import shutil; shutil.rmtree(cdir, ignore_errors=True)
            skipped["self_intersecting" if fatal else "checks_failed"] += 1
            print("      -> discarded" + (" (self-intersection is not permitted)" if fatal else ""))
            continue
        written += 1

    (out_dir / "manifest.json").write_text(json.dumps({
        "n_written": written, "n_attempted": attempts, "skipped": skipped,
        "n_flagged": len(flagged), "flagged": flagged,
        "kept_with_warnings": kept_flagged,
        "skip_flagged": args.skip_flagged, "self_intersection_check": args.check_si,
        "ghd_vae": str(ghd_ckpt), "sensor": str(args.sensor),
        "branch_model": str(br_ckpt), "branch_epoch": br_epoch,
        "seed": args.seed, "z_amp": args.z_amp, "max_kr": args.max_kr,
        "seam_rings": args.seam_rings, "extrude_length": args.extrude_length,
    }, indent=2))
    print(f"\n{written} shape(s) -> {out_dir}   "
          f"(skipped {sum(skipped.values())}: {skipped})")
    n_types = {0: 0, 1: 0}
    for c in sorted(out_dir.iterdir()):
        if c.is_dir():
            n_types[int(c.name.rsplit("_t", 1)[1])] += 1
    print(f"types written: {n_types[0]} bifurcated / {n_types[1]} sidewall")
    if skipped["self_intersecting"]:
        print(f"{skipped['self_intersecting']} discarded for self-intersection (not permitted)")
    if kept_flagged:
        print(f"{len(kept_flagged)} written WITH warnings, logged in their meta.json: "
              + ", ".join(kept_flagged))


if __name__ == "__main__":
    main()
