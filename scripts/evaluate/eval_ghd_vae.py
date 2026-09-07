"""
Stage-1-only evaluation for the GHD VAE — no stage-2 checkpoint required.

scripts/evaluate/eval_ablation.py pairs each stage1 config with a REFERENCE
stage2 (branch MLP-VAE) checkpoint, so it cannot run before stage 2 exists.
When the plan is to tune stage 1 first and only then train the branch VAE,
that is the wrong tool: this one loads a stage1 checkpoint alone, samples
z ~ N(0,1), decodes, reconstructs the GHD mesh, and renders it.

Two outputs per sweep:

  <out>/<config>.png     one generative panel per config (N_GEN meshes)
  <out>/_contact.png     every config side by side, same seed and same
                         sampled types, so configs differ only by the model

The contact sheet is the point: the only quality signal here is human
inspection, and judging 12 panels opened one at a time is much harder than
judging one image where the same z and the same types were used throughout.

Also prints a spread statistic per config (std of generated phi relative to the
real corpus's), because the failure mode that is hardest to see by eye is a
posterior collapse that still renders plausible individual shapes -- they are
plausible but all the same.

    python scripts/evaluate/eval_ghd_vae.py --stage1-dir runtime_train/ghd_vae/stage1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from utils.generate_synthetic import load_ghd_vae
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from dataset.ghd_datasets import ProcessedGHDDataset

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
DEFAULT_PROCESSED = ROOT / "runtime_dataset" / "AneuG_processed"
# type 2's embedding row is never trained (train_ghd_vae.MERGE_TYPE2_INTO_TYPE1),
# so sampling it would decode an untrained condition vector.
N_SAMPLED_TYPES = 2
# Shared fixed camera + surface alpha. Same view for every mesh and every
# config, so panels are directly comparable.
ELEV, AZIM, ALPHA = 18.0, 35.0, 0.45


def _set_axes_equal(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max() / 2) or 1.0
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def discover(stage1_dir, epoch_name):
    out = []
    for d in sorted(Path(stage1_dir).iterdir()):
        ck = d / epoch_name
        if d.is_dir() and ck.exists():
            out.append((d.name, ck))
    return out


@torch.no_grad()
def generate(ckpt, device, n, seed, z_amp=1.0):
    """(verts_list, faces_list, types, phi) for n samples from one checkpoint.

    z ~ N(0,1) -- the prior the KL term actually trains against. Sampling
    uniformly instead would evaluate the model off-distribution and make a
    well-behaved VAE look broken.
    """
    vae, mean, std, ghd_dim = load_ghd_vae(ckpt, device)
    recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    types = torch.randint(0, N_SAMPLED_TYPES, (n,), generator=g).to(device)
    z = (torch.randn(n, vae.latent_dim, generator=g) * z_amp).to(device)
    with_scale = mean.numel() > ghd_dim
    phi_n = vae.decode(z, types, strip_scale=True) if with_scale else vae.decode(z, types)

    m = mean[..., :ghd_dim] if mean.numel() > ghd_dim else mean
    s = std[..., :ghd_dim] if std.numel() > ghd_dim else std
    V, F = [], []
    for i in range(n):
        mesh = recon.get(int(types[i])).ghd_forward_as_Meshes(phi_n[i:i+1], mean=m, std=s)
        V.append(mesh.verts_padded()[0].cpu().numpy())
        F.append(mesh.faces_padded()[0].cpu().numpy())
    return V, F, types.cpu().numpy(), (phi_n * s + m).cpu().numpy()


def panel(V, F, types, title, out_path, ncols=6, seed=0, elev=ELEV, azim=AZIM,
          alpha=ALPHA):
    """One generative panel. NO random rotation: every mesh is a deformation of
    the same template, so holding the camera fixed means a difference on screen
    is a difference in the shape rather than in the viewpoint -- which is the
    whole point when comparing configs by eye. Surfaces are drawn semi-
    transparent so the far wall and any self-intersection show through."""
    n = len(V)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.1 * ncols, 3.1 * nrows))
    for i in range(n):
        v = V[i]
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=F[i],
                        color="lightsteelblue", edgecolor="none", alpha=alpha)
        ax.set_axis_off(); ax.set_title(f"type {int(types[i])}", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        _set_axes_equal(ax, v)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_objs(V, F, types, out_dir):
    """Write each generated mesh as .obj so they can be opened in a real viewer.

    A matplotlib panel is fine for a first pass but cannot be rotated, zoomed,
    or sliced; judging generation quality usually needs that.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (v, f, t) in enumerate(zip(V, F, types)):
        p = out_dir / f"gen_{i:02d}_type{int(t)}.obj"
        with open(p, "w") as fh:
            for x, y, z in v:
                fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in f:
                fh.write(f"f {a+1} {b+1} {c+1}\n")
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Stage-1-only GHD VAE evaluation.")
    ap.add_argument("--stage1-dir", default=str(ROOT / "runtime_train" / "ghd_vae" / "stage1"))
    ap.add_argument("--epoch", default="epoch_05000.pth")
    ap.add_argument("--contact-path", default=None,
                    help="Where the cross-config contact sheet goes "
                         "(default: <stage1-dir>/_contact.png).")
    ap.add_argument("--processed-root", default=str(DEFAULT_PROCESSED),
                    help="Real corpus, used only for the phi-spread reference.")
    ap.add_argument("--n-gen", type=int, default=12)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-amp", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--contact-per-config", type=int, default=4,
                    help="Meshes per config on the side-by-side contact sheet.")
    ap.add_argument("--elev", type=float, default=ELEV, help="Fixed camera elevation.")
    ap.add_argument("--azim", type=float, default=AZIM, help="Fixed camera azimuth.")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="Surface transparency.")
    ap.add_argument("--no-obj", dest="save_obj", action="store_false",
                    help="Skip writing the generated meshes as .obj files.")
    ap.set_defaults(save_obj=True)
    args = ap.parse_args()

    stage1 = Path(args.stage1_dir)
    configs = discover(stage1, args.epoch)
    if not configs:
        print(f"no configs with {args.epoch} under {stage1}")
        sys.exit(1)
    device = torch.device(args.device)

    real_std = None
    try:
        ds = ProcessedGHDDataset(args.processed_root, withscale=False, normalize=False)
        real_std = float(torch.stack(ds.ghd).std(0).mean())
    except Exception as exc:
        print(f"  (no real-corpus reference: {exc})")

    print(f"  {len(configs)} config(s), eval written beside each checkpoint\n")
    print(f"    {'config':<28}{'phi std':>10}{'vs real':>10}   output")
    rows = []
    for name, ck in configs:
        V, F, types, phi = generate(ck, device, args.n_gen, args.seed, args.z_amp)
        # eval lives WITH the checkpoint it describes -- <config>/eval/ -- so a
        # config dir is self-contained and stays correct if it is moved or
        # deleted, rather than leaving orphans in a shared eval folder.
        cfg_out = ck.parent / "eval"
        cfg_out.mkdir(parents=True, exist_ok=True)
        p = cfg_out / "generation.png"
        panel(V, F, types, f"{name}   z~N(0,1)  seed={args.seed}", p,
              ncols=args.ncols, seed=args.seed, elev=args.elev, azim=args.azim,
              alpha=args.alpha)
        if args.save_obj:
            save_objs(V, F, types, cfg_out / "meshes")
        gstd = float(phi.std(0).mean())
        ratio = f"{gstd/real_std:.2f}x" if real_std else "-"
        print(f"    {name:<28}{gstd:>10.4f}{ratio:>10}   {p.relative_to(stage1)}")
        rows.append((name, V[:args.contact_per_config], F[:args.contact_per_config],
                     types[:args.contact_per_config]))

    # contact sheet: one row per config, same z and types across rows
    k = args.contact_per_config
    fig = plt.figure(figsize=(3.0 * k, 3.0 * len(rows)))
    for r, (name, V, F, types) in enumerate(rows):
        for c in range(min(k, len(V))):
            v = V[c]
            ax = fig.add_subplot(len(rows), k, r * k + c + 1, projection="3d")
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=F[c],
                            color="lightsteelblue", edgecolor="none", alpha=args.alpha)
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_axis_off()
            if c == 0:
                ax.set_title(name, fontsize=9, loc="left")
            _set_axes_equal(ax, v)
    fig.suptitle(f"GHD VAE stage-1 sweep — same z and types in every row (z~N(0,1), seed={args.seed})",
                 fontsize=12)
    fig.tight_layout()
    # The one output that cannot belong to a single config: it compares them.
    contact = Path(args.contact_path) if args.contact_path else stage1 / "_contact.png"
    contact.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(contact, dpi=110)
    plt.close(fig)
    print(f"\n  contact sheet: {contact}")


if __name__ == "__main__":
    main()
