"""Generate a fixed pool of synthetic shapes for sensor fine-tuning.

WHY A POOL RATHER THAN GENERATING INSIDE THE LABELLER. Labelling is slow and
gets abandoned partway, and it is resumed days later. If the shapes are drawn
on the fly, the set being labelled depends on when the labeller ran, which
checkpoint list was passed that day and which seed. Writing the pool once makes
the corpus a fixed, inspectable artefact: it can be eyeballed before any effort
is spent on it, re-labelled by someone else, or extended without disturbing
what is already done.

Each record carries the z that produced it, so any shape can be regenerated
exactly, and the pool can later be filtered by generator or by how extreme the
draw was.

    python scripts/generate/gen_synthetic_pool.py --n-per-ckpt 40 \
        --ghd-vae runtime_train/ghd_vae/stage1/ghd_vae_gan_h512_z16_kl2_adv0.3/epoch_05000.pth \
                  runtime_train/ghd_vae/stage1/ghd_vae_h256_z16_kl2/epoch_05000.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from utils.generate_synthetic import load_ghd_vae

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
DEFAULT_OUT = ROOT / "runtime_train" / "synthetic_pool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghd-vae", nargs="+", required=True,
                    help="Stage-1 checkpoints to draw from. Pool several so the "
                         "fine-tuned sensor learns synthetic shapes in general rather "
                         "than one generator's artefacts.")
    ap.add_argument("--n-per-ckpt", type=int, default=40)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--z-amp-range", nargs=2, type=float, default=[1.0, 1.0],
                    metavar=("LOW", "HIGH"),
                    help="Each shape draws its own amplitude uniformly from this range. "
                         "Default 1.0 1.0, i.e. z ~ N(0,1) exactly -- the prior the KL term "
                         "actually trains against, and the distribution the sensor will meet "
                         "in use. Widening it (e.g. 1.0 5.0) samples the tails on purpose, "
                         "which surfaces failures faster but trains the sensor on shapes the "
                         "generator is never asked to produce.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--no-render", dest="render", action="store_false",
                    help="Skip the per-checkpoint contact sheet.")
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=dev)
    lo, hi = args.z_amp_range
    n_written = 0

    for gi, ck_path in enumerate(args.ghd_vae):
        ck_path = Path(ck_path)
        if not ck_path.exists():
            ap.error(f"not found: {ck_path}")
        vae, mean, std, gdim = load_ghd_vae(ck_path, dev)
        tag = ck_path.parent.name
        g = torch.Generator(device="cpu").manual_seed(args.seed + 1000 * gi)
        n = args.n_per_ckpt
        types = torch.randint(0, 2, (n,), generator=g)      # type 2 is merged into 1
        amps = (torch.rand(n, generator=g) * (hi - lo) + lo)
        z = torch.randn(n, vae.latent_dim, generator=g) * amps[:, None]
        with_scale = mean.numel() > gdim
        with torch.no_grad():
            o = vae.decode(z.to(dev), types.to(dev), strip_scale=True) if with_scale \
                else vae.decode(z.to(dev), types.to(dev))
            phi_n, scale_n = (o if isinstance(o, tuple) else (o, None))
            m = mean[..., :gdim] if with_scale else mean
            s = std[..., :gdim] if with_scale else std
            phi = (phi_n * s + m).reshape(n, -1, 3).cpu().numpy()
            scales = ((scale_n * std[..., gdim:] + mean[..., gdim:]).squeeze(1).cpu().numpy()
                      if with_scale and scale_n is not None else np.ones(n, dtype=np.float32))

        for i in range(n):
            case = f"synthetic_{tag}_seed{args.seed}_{i:04d}"
            np.save(out / f"{case}.npy", {
                "case": case, "aneurysm_type": int(types[i]),
                "phi": phi[i].astype(np.float32), "scale": float(scales[i]),
                "is_synthetic": True,
                "provenance": {"ghd_vae": str(ck_path), "z_amp": float(amps[i]),
                               "seed": int(args.seed), "index": int(i),
                               "z": z[i].numpy().astype(np.float32)},
            }, allow_pickle=True)
            n_written += 1
        print(f"  {tag}: {n} shape(s), z_amp {float(amps.min()):.2f}-{float(amps.max()):.2f}")

        if args.render:
            _contact(recon, phi, types.numpy(), amps.numpy(), out / f"_pool_{tag}.png", tag)

    print(f"\n{n_written} shape(s) -> {out}")


def _contact(recon, phi, types, amps, path, tag, ncols=8):
    """Eyeball sheet for the pool, so obviously broken shapes can be spotted
    before any labelling effort is spent on them."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    n = len(phi); nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(2.6 * ncols, 2.6 * nrows))
    for i in range(n):
        v = recon._reconstruct_verts_np(phi[i], int(types[i]))
        f = recon.get(int(types[i])).canonical_Meshes.faces_packed().cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                        color="lightsteelblue", edgecolor="none", alpha=0.55)
        ax.set_axis_off(); ax.view_init(elev=18, azim=35)
        ax.set_title(f"{i} t{int(types[i])} a{amps[i]:.1f}", fontsize=7)
        c, r = (v.min(0) + v.max(0)) / 2, float((v.max(0) - v.min(0)).max() / 2) or 1.0
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    fig.suptitle(tag); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"    contact sheet -> {path.name}")


if __name__ == "__main__":
    main()
