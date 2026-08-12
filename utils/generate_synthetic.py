"""
Fully-generative synthetic shape generation for the v2 pipeline.

Samples two pretrained checkpoints end to end — a TypeConditionalVAE (GHD VAE,
see scripts/train/train_ghd_vae.py) and a BranchFourierVAE[_GCNConditioner] (branch MLP-VAE,
see scripts/train/train_branch_mlp_vae.py) — into fused vessel meshes with generated branch
centerlines. Extracted from scripts/evaluate/eval_branch_mlp_vae.py.

Both checkpoints must carry a saved "args" entry (added by the current training
scripts' model_args()) so the models can be reconstructed without guessing
architecture hyperparameters from folder names or state-dict shapes.

This is library code rather than a command-line entry point. Configured drivers
live in ``scripts/generate/`` and evaluation scripts reuse this module.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.skeleton_dataset_fourier import reconstruct_branch
from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy
from models.branch_transformer import MultiBranchConditions
from models.branch_mlp_vae import BranchFourierVAE, BranchFourierVAE_GCNConditioner
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from models.ghd_vae import TypeConditionalVAE


def _require_args(ckpt, ckpt_path, train_script):
    if "args" not in ckpt:
        raise KeyError(
            f"{ckpt_path} has no 'args' entry; retrain with the current "
            f"{train_script} so the checkpoint carries its own model_args()."
        )
    return ckpt["args"]


def load_ghd_vae(ckpt_path, device):
    """Load a TypeConditionalVAE checkpoint saved by scripts/train/train_ghd_vae.py.

    Returns (vae, mean, std, ghd_input_dim) — mean/std are the dataset's
    normalization stats (phi flattened + scale) and ghd_input_dim is the
    flattened phi dim (i.e. args["input_dim"], before the withscale +1).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    args = _require_args(ckpt, ckpt_path, "scripts/train/train_ghd_vae.py")
    vae = TypeConditionalVAE(**args).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae, ckpt["mean"].to(device), ckpt["std"].to(device), args["input_dim"]


def load_branch_vae(ckpt_path, multi_recon, device):
    """Load a BranchFourierVAE(_GCNConditioner) checkpoint saved by scripts/train/train_branch_mlp_vae.py."""
    ckpt = torch.load(ckpt_path, map_location=device)
    args = dict(_require_args(ckpt, ckpt_path, "scripts/train/train_branch_mlp_vae.py"))
    use_gcn = args.pop("use_gcn")
    model = (BranchFourierVAE_GCNConditioner(multi_recon, **args) if use_gcn
              else BranchFourierVAE(**args)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def _sample_ghd(ghd_vae, ghd_mean, ghd_std, ghd_input_dim, types, z_amp=1.0):
    B = types.size(0)
    device = types.device
    scale = 1.0 if z_amp is None else z_amp
    z_ghd = torch.randn(B, ghd_vae.latent_dim, device=device) * scale
    ghd_n, scale_n = ghd_vae.decode(z_ghd, types)
    phi   = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(B, -1, 3)
    scale = (scale_n * ghd_std[:, ghd_input_dim:] + ghd_mean[:, ghd_input_dim:]).squeeze(1)
    return phi, scale


@torch.no_grad()
def _build_conditions(multi_recon, phi, types, scale, max_branches):
    """Per-branch conditions (start + outward dir + candidate openings) for a mixed-type batch."""
    B = phi.size(0)
    starts = phi.new_zeros(B, max_branches, 3)
    dirs   = phi.new_zeros(B, max_branches, 3)
    mask   = torch.zeros(B, max_branches, dtype=torch.bool, device=phi.device)
    for atype in sorted(set(int(t) for t in types.tolist())):
        idx = (types == atype).nonzero(as_tuple=True)[0]
        s, d, m = multi_recon.compute_branch_conditions(phi[idx], atype, max_branches)
        starts[idx], dirs[idx], mask[idx] = s, d, m
    return MultiBranchConditions(types, scale, starts, dirs, mask)


def polydata_tris(mesh):
    """(verts, faces) numpy arrays from a pyvista PolyData (triangles)."""
    return np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:]


def _bare_ghd_mesh(phi_i, atype, obj_path):
    """Capped GHD-only mesh (no branch tubes), used when fuse=False or fusion fails."""
    verts, faces = _reconstruct_ghd_numpy(
        {"aneurysm_type": atype, "ghd": {"phi": phi_i.cpu().numpy()}}, denormalize_shape=True)
    if obj_path is not None:
        trimesh.Trimesh(verts, faces, process=False).export(obj_path)
    return verts, faces


def _branch_centerlines(start_b, vec_b, coeff_b, pres_b, mask_b, thresh):
    """Per-opening absolute centerline where present, else None (→ extrusion stub)."""
    branches = []
    for b in range(len(mask_b)):
        if mask_b[b] and pres_b[b] > thresh:
            branches.append(reconstruct_branch(start_b[b], start_b[b] + vec_b[b], coeff_b[b]))
        else:
            branches.append(None)
    return branches


@torch.no_grad()
def generate_synthetic_shapes(
    ghd_vae_ckpt,
    branch_vae_ckpt,
    canonical_root,
    n_samples=12,
    types=None,
    ghd_z_amp=5.0,
    branch_z_zero=True,
    branch_z_amp=5.0,
    presence_thresh=0.5,
    fuse=True,
    extrude_length=3.0,
    min_branch_arc=3.0,
    fuse_smooth=True,
    obj_dir=None,
    device=None,
    seed=None,
):
    """Fully-generative pipeline: sample a GHD VAE and a branch Fourier MLP-VAE
    end to end into fused vessel meshes (no ground-truth data involved).

    1. Sample z_ghd ~ N(0, ghd_z_amp^2) and decode the GHD VAE (conditioned on
       aneurysm type) into a GHD shape (phi) and scale.
    2. Reconstruct the mesh openings via a MultiCanonicalGHDReconstruct built
       from canonical_root → per-branch start points, outward directions,
       candidate-opening mask.
    3. Generate per-branch (chord vector, Fourier coefficients, presence) with
       the branch MLP-VAE, conditioned on that geometry.
    4. Reconstruct each present branch's centerline and fuse it with the
       uncapped GHD mesh (tube for present branches, short extrusion stub for
       absent/invalid ones).

    Args:
        ghd_vae_ckpt:    path to a scripts/train/train_ghd_vae.py checkpoint (needs "args").
        branch_vae_ckpt: path to a scripts/train/train_branch_mlp_vae.py checkpoint (needs "args").
        canonical_root:  dataset/canonical directory (Bifurcated/Sidewall templates).
        n_samples:       number of samples to generate.
        types:           None → random type per sample; an int → same type for
                         all samples; or a sequence of length n_samples.
        ghd_z_amp:       GHD latent sampled ~ N(0, ghd_z_amp^2); None → std 1.
        branch_z_zero:   True → branch latent z = 0 (mode); False → sample branch_z_amp.
        branch_z_amp:    branch latent sampled ~ N(0, branch_z_amp^2) when
                         branch_z_zero is False.
        presence_thresh: a candidate opening is tubed iff predicted presence > this.
        fuse:            True → fuse branches onto the GHD mesh; False → return the
                         bare (capped) GHD mesh, no branch tubes/stubs.
        extrude_length, min_branch_arc, fuse_smooth: passed to reconstruct_fused_mesh.
        obj_dir:         if given, save each sample's fused mesh as an .obj there.
        device:          torch device (or device string); defaults to cuda:0 if
                         available else cpu.
        seed:            optional RNG seed for reproducible sampling.

    Returns:
        list of dicts, one per sample:
          {
            "type": int,
            "phi": Tensor [num_coeffs, 3],
            "scale": Tensor [1],
            "branches": list (len = max_branches) of ndarray [N, 3] centerlines,
                        or None where absent / not a candidate opening,
            "presence": ndarray [max_branches] — predicted P(branch exists),
            "opening_mask": ndarray [max_branches] bool — candidate openings,
            "verts": ndarray [V, 3],
            "faces": ndarray [F, 3],
            "obj_path": Path or None,
          }
    """
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    multi_recon = MultiCanonicalGHDReconstruct(canonical_root, device=device)
    ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(ghd_vae_ckpt, device)
    branch_model = load_branch_vae(branch_vae_ckpt, multi_recon, device)

    if types is None:
        type_t = torch.randint(0, ghd_vae.num_types, (n_samples,), device=device)
    elif isinstance(types, int):
        type_t = torch.full((n_samples,), types, dtype=torch.long, device=device)
    else:
        type_t = torch.as_tensor(list(types), dtype=torch.long, device=device)
        n_samples = type_t.size(0)

    phi, scale = _sample_ghd(ghd_vae, ghd_mean, ghd_std, ghd_input_dim, type_t, z_amp=ghd_z_amp)
    cond = _build_conditions(multi_recon, phi, type_t, scale, branch_model.max_branches)

    z = (
        torch.zeros(n_samples, branch_model.latent_dim, device=device)
        if branch_z_zero
        else torch.randn(n_samples, branch_model.latent_dim, device=device) * branch_z_amp
    )
    vec, coeff, pres = branch_model.sample(phi, cond, z=z)

    starts_np = cond.start_points.cpu().numpy()
    vec_np, coeff_np, pres_np = vec.cpu().numpy(), coeff.cpu().numpy(), pres.cpu().numpy()
    mask_np = cond.branch_mask.cpu().numpy()

    if obj_dir is not None:
        obj_dir = Path(obj_dir)
        obj_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(n_samples):
        atype = int(type_t[i])
        branches = _branch_centerlines(starts_np[i], vec_np[i], coeff_np[i], pres_np[i], mask_np[i], presence_thresh)
        obj_path = (obj_dir / f"{i:03d}_type{atype}.obj") if obj_dir is not None else None

        if fuse:
            try:
                branch_mask = [b is not None for b in branches]
                merged = multi_recon.reconstruct_fused_mesh(
                    phi[i], atype, branch_points=branches, branch_mask=branch_mask,
                    extrude_length=extrude_length, min_branch_arc=min_branch_arc,
                    smooth=fuse_smooth, save_path=obj_path,
                )
                verts, faces = polydata_tris(merged)
            except Exception as exc:
                print(f"[warn] sample {i} (type {atype}): fusion failed ({exc}); falling back to GHD mesh")
                verts, faces = _bare_ghd_mesh(phi[i], atype, obj_path)
        else:
            verts, faces = _bare_ghd_mesh(phi[i], atype, obj_path)

        results.append({
            "type": atype,
            "phi": phi[i].cpu(),
            "scale": scale[i].cpu(),
            "branches": branches,
            "presence": pres_np[i],
            "opening_mask": mask_np[i],
            "verts": verts,
            "faces": faces,
            "obj_path": obj_path,
        })

    return results
