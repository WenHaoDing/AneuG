"""Stage-1 GHD VAE with a WGAN-GP critic on decoded meshes.

WHY THIS EXISTS. Scoring the 12 plain-VAE ablations with the morphology sensor
showed one dominant failure, identical across every configuration: the models
are plausible but not diverse. Precision ran 0.96-0.99 while recall ran
0.14-0.48, and generated diversity (TMD) topped out at 0.82 against the real
corpus's 1.18. Raising the KL weight helped on both axes, which says the gap is
between the aggregate posterior and the prior -- shapes decoded from z ~ N(0,1)
do not cover what real shapes occupy.

So the adversarial term is applied to PRIOR SAMPLES, not to reconstructions.
Reconstruction is already good; it is sampling that is weak, and a critic that
only ever saw reconstructions would not touch the actual problem.

    total generator loss = (the existing VAE loss, unchanged)
                         + adv_weight * ( -E[ D(decode(z ~ N(0,1))) ] )

Nothing in train_ghd_vae.py is modified. Its loss, dataset, model, mesh loss
and plotting are imported and reused; this file adds the critic, the WGAN-GP
update, and its own argument parsing and save directory.

    python scripts/train/train_ghd_vae_gan.py --hidden-dim 512 --latent-dim 16 \
        --kl-mult 4 --adv-weight 0.1 --device cuda:1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct
from models.vae_models import KL_divergence
from models.ghd_vae import TypeConditionalVAE
from models.ghd_critic import PointNetPPCritic, gradient_penalty
from models.morphoformer import RandomSO3Rotation
import scripts.train.train_ghd_vae as V


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--kl-mult", type=float, default=4.0)
    p.add_argument("--kl-anneal-epochs", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--save-root", default=str(ROOT / "runtime_train" / "ghd_vae"))
    p.add_argument("--processed-root", default=str(ROOT / "runtime_dataset" / "AneuG_processed"))
    p.add_argument("--adv-weight", type=float, default=0.1,
                   help="Adversarial pressure as a FRACTION of reconstruction pressure. "
                        "With --adaptive-adv (default) the raw adversarial gradient is "
                        "first rescaled to match the reconstruction gradient at the "
                        "decoder's output layer, so 0.1 literally means 'one tenth as "
                        "hard as reconstruction' and cannot silently explode.")
    p.add_argument("--no-adaptive-adv", dest="adaptive_adv", action="store_false",
                   help="Use adv_weight as a raw multiplier instead (classic, needs tuning).")
    p.add_argument("--no-critic-rotate", dest="critic_rotate", action="store_false", default=True,
                   help="Disable the random rotation of critic inputs. 523 real meshes is few enough "
                        "that the critic can memorise them; a fresh orientation every step "
                        "is free augmentation and costs no realism, since orientation is "
                        "fixed by the template and carries no information.")
    p.add_argument("--n-critic", type=int, default=5,
                   help="Critic updates per generator update. WGAN needs the critic "
                        "near-optimal for its output to approximate a Wasserstein distance.")
    p.add_argument("--gp-weight", type=float, default=10.0, help="Gradient-penalty weight.")
    p.add_argument("--critic-hidden", type=int, default=128)
    p.add_argument("--critic-points", type=int, default=1024,
                   help="Vertices randomly subsampled per critic forward (0 = all). "
                        "Every mesh of a type shares topology and vertex ordering, so a "
                        "critic given all 4143 vertices in a fixed order can key on "
                        "index rather than geometry; resampling each step removes that "
                        "and cuts the critic's cost about fourfold.")
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--adv-warmup", type=int, default=500,
                   help="Epochs of pure VAE training before the adversarial term is "
                        "switched on. A critic scoring an untrained decoder produces "
                        "a large, uninformative gradient that can wreck the VAE early.")
    return p.parse_args()


def plot_loss_curves(hist, save_dir, title):
    """Overwrite loss_curves.png with the history so far.

    Not train_ghd_vae.plot_loss_curves: that one hardcodes the six VAE series
    and reads a module global for its title. The adversarial run has four more
    series that are the whole point of watching it -- w_dist, lam, adv and
    adv_frac -- so it gets its own grid rather than a modified shared one.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    keys = ["loss", "recon", "mesh", "kl", "scale", "w_dist", "d_sep", "lam", "adv_frac"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, k in zip(axes.flat, keys):
        ax.plot(hist["epoch"], hist[k], lw=1.2)
        ax.set_title(k); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        if k in ("recon", "mesh", "kl"):
            ax.set_yscale("log")
    axes.flat[5].axhline(0.0, color="crimson", ls="--", lw=1)
    axes.flat[6].axhline(0.5, color="crimson", ls="--", lw=1)   # chance
    axes.flat[6].set_ylim(0.4, 1.02)
    fig.suptitle(title)
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / "loss_curves.png", dpi=130)
    plt.close(fig)


def main():
    a = parse_args()
    dev = torch.device(a.device)

    # Reuse train_ghd_vae's helpers verbatim; they read a few module globals, so
    # point those at THIS run's settings rather than their import-time defaults.
    V.DEVICE = dev
    V.KL_WEIGHT = V.KL_BASE * a.kl_mult
    V.MERGE_TYPE2_INTO_TYPE1 = True

    save_dir = (Path(a.save_root) / "stage1" /
                f"ghd_vae_gan_h{a.hidden_dim}_z{a.latent_dim}_kl{a.kl_mult:g}_adv{a.adv_weight:g}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save dir: {save_dir}")

    dataset = ProcessedGHDDataset(Path(a.processed_root), withscale=True, normalize=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    # get_mean_std() returns the GHD half only (432); dataset.mean/std are the
    # full 433 including the scale channel. The mesh loss needs the former, but
    # the checkpoint must store the LATTER, because that is what the plain VAE
    # stores and what utils.generate_synthetic.load_ghd_vae + every eval script
    # assume when they test `mean.numel() > input_dim` to detect withscale.
    mean, std = dataset.get_mean_std()
    mean, std = mean.to(dev), std.to(dev)
    full_mean, full_std = dataset.mean, dataset.std

    margs = {"input_dim": dataset.get_dim(), "hidden_dim": a.hidden_dim,
             "latent_dim": a.latent_dim, "withscale": True, "num_types": 3,
             "condition_dim": 8}
    model = TypeConditionalVAE(**margs).to(dev)
    critic = PointNetPPCritic(hidden=a.critic_hidden).to(dev)
    recon = get_ghd_reconstruct(device=dev)

    opt_g = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # betas=(0.0, 0.9) is the WGAN-GP default: low momentum keeps the critic
    # responsive to a generator that is still moving.
    opt_d = torch.optim.Adam(critic.parameters(), lr=a.critic_lr, betas=(0.0, 0.9))
    sched_g = torch.optim.lr_scheduler.LinearLR(opt_g, 1.0, 0.05, total_iters=a.epochs)

    def verts_of(ghd_n, types):
        """Differentiable normalized-GHD -> padded vertices, grouped by type.
        Yields (type_id, mask, verts) so the critic never sees two templates in
        one padded tensor."""
        for t in types.unique(sorted=True):
            m = types == t
            if m.any():
                yield int(t), m, recon.get(int(t)).ghd_forward_as_Meshes(
                    ghd_n[m], mean=mean, std=std).verts_padded()

    def pick(n_verts):
        """Shared vertex subsample. Real and fake MUST use the same indices, or
        the gradient-penalty interpolation would mix non-corresponding points."""
        if a.critic_points <= 0 or a.critic_points >= n_verts:
            return None
        return torch.randperm(n_verts, device=dev)[:a.critic_points]

    def sub(v, idx):
        return v if idx is None else v[:, idx]

    def rot(v):
        """Independent random rotation per mesh."""
        if not a.critic_rotate:
            return v
        Q = RandomSO3Rotation.sample(v.size(0), v.dtype, dev)
        return torch.einsum('bij,bnj->bni', Q, v)

    def adaptive_lambda(fidelity_loss, adv_loss):
        """Rescale the adversarial gradient to match the reconstruction gradient
        at the decoder's output layer (Esser et al., VQGAN).

        This is the whole answer to "how strong should the discriminator be".
        A fixed adversarial weight is a bet on the ratio between two losses whose
        scales drift by orders of magnitude during training: early on the critic
        is uninformative and the term is noise, later it can overwhelm
        reconstruction and destroy the VAE. Matching gradient norms makes the
        balance hold automatically, and turns --adv-weight into a readable
        fraction rather than a number to be found by trial and error.
        """
        gr = torch.autograd.grad(fidelity_loss, model.fc4.weight, retain_graph=True)[0]
        ga = torch.autograd.grad(adv_loss, model.fc4.weight, retain_graph=True)[0]
        return (gr.norm() / (ga.norm() + 1e-4)).clamp(0, 1e4).detach()

    wandb.init(project="AneuGv2_stage1", name=save_dir.name,
               config={**vars(a), **margs})
    hist = {k: [] for k in ("epoch", "loss", "recon", "mesh", "kl", "scale",
                                    "w_dist", "d_sep", "adv", "lam", "adv_frac")}

    for epoch in range(a.epochs + 1):
        model.train()
        kl_w = (V.KL_BASE * a.kl_mult * min(1.0, epoch / a.kl_anneal_epochs)
                if a.kl_anneal_epochs > 0 else V.KL_BASE * a.kl_mult)
        adv_on = epoch >= a.adv_warmup
        m = {}
        for batch, atype in loader:
            batch = batch.to(dev)
            atype = V._merge_type2(atype.to(dev))
            ghd, scale = batch[:, :-1], batch[:, -1:]

            # ---- critic: maximise E[D(real)] - E[D(fake)], fake ~ prior ----
            w_dist, sep_num, sep_den = 0.0, 0.0, 0
            if adv_on:
                n_w = 0
                for _ in range(a.n_critic):
                    with torch.no_grad():
                        z = torch.randn(ghd.size(0), a.latent_dim, device=dev)
                        out = model.decode(z, atype, strip_scale=True)
                        fake_n = out[0] if isinstance(out, tuple) else out
                    d_loss = 0.0
                    for t, mask, rv in verts_of(ghd, atype):
                        fv = recon.get(t).ghd_forward_as_Meshes(
                            fake_n[mask], mean=mean, std=std).verts_padded()
                        idx = pick(rv.size(1))
                        rv, fv = rot(sub(rv, idx)), rot(sub(fv, idx))
                        sr, sf = critic(rv, atype[mask]), critic(fv, atype[mask])
                        gp = gradient_penalty(critic, rv, fv, atype[mask])
                        d_loss = d_loss + (sf.mean() - sr.mean()) + a.gp_weight * gp
                        # w_dist was previously ASSIGNED here, inside both the
                        # per-type and the n_critic loop, so it reported only the
                        # last type of the last critic step. Accumulate instead.
                        w_dist += float((sr.mean() - sf.mean()).detach())
                        n_w += 1
                        # Separation: P(a real scores above a fake), over all
                        # real-fake pairs. This is the Mann-Whitney statistic,
                        # i.e. ROC AUC. Unlike w_dist it is bounded in [0,1] with
                        # 0.5 meaning "cannot tell them apart", so it IS
                        # comparable across runs and IS the critic's answer to
                        # "how often do I identify the fake correctly".
                        sep_num += float((sr.detach()[:, None] > sf.detach()[None, :]).float().mean())
                        sep_den += 1
                    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # ---- generator: the existing VAE loss, plus -E[D(prior sample)] ----
            ghd_recon, scale_recon, mu, logvar = model(ghd, atype, scale)
            recon_l = F.mse_loss(ghd_recon, ghd)
            mesh_l = V.mesh_reconstruction_loss(ghd, ghd_recon, atype, recon, mean, std)
            kl_l = KL_divergence(mu, logvar)
            scale_l = F.huber_loss(scale_recon, scale, delta=2.0)
            fidelity = recon_l + V.MESH_WEIGHT * mesh_l
            loss = fidelity + kl_w * kl_l + V.SCALE_WEIGHT * scale_l

            adv_l = torch.zeros((), device=dev)
            lam = 0.0
            if adv_on:
                z = torch.randn(ghd.size(0), a.latent_dim, device=dev)
                out = model.decode(z, atype, strip_scale=True)
                fake_n = out[0] if isinstance(out, tuple) else out
                for t, mask, _rv in verts_of(ghd, atype):
                    fv = recon.get(t).ghd_forward_as_Meshes(
                        fake_n[mask], mean=mean, std=std).verts_padded()
                    adv_l = adv_l - critic(rot(sub(fv, pick(fv.size(1)))), atype[mask]).mean()
                lam = float(adaptive_lambda(fidelity, adv_l)) if a.adaptive_adv else 1.0
                loss = loss + a.adv_weight * lam * adv_l

            opt_g.zero_grad(); loss.backward(); opt_g.step()
            m = {"loss": float(loss), "recon": float(recon_l), "mesh": float(mesh_l),
                 "kl": float(kl_l), "scale": float(scale_l),
                 "w_dist": w_dist / max(n_w, 1) if adv_on else 0.0,
                 "d_sep": sep_num / sep_den if sep_den else 0.5,
                 "adv": float(adv_l), "lam": lam,
                 # the number to watch: adversarial pressure actually applied,
                 # relative to reconstruction. Should sit near --adv-weight.
                 "adv_frac": a.adv_weight * lam * abs(float(adv_l)) / (float(fidelity) + 1e-8)}

        sched_g.step()
        if epoch % 10 == 0:
            print({"epoch": epoch, "adv_on": adv_on, **{k: round(v, 5) for k, v in m.items()}})
            wandb.log({"epoch": epoch, "lr": sched_g.get_last_lr()[0], "kl_weight": kl_w, **m},
                      step=epoch)
            hist["epoch"].append(epoch)
            for k in ("loss", "recon", "mesh", "kl", "scale", "w_dist", "d_sep",
                      "adv", "lam", "adv_frac"):
                hist[k].append(m[k])
            plot_loss_curves(hist, save_dir, save_dir.name)
        if epoch % 500 == 0:
            V.sanity_synthetic(model, mean, std, recon, epoch, save_dir)
        if epoch % 1000 == 0 or epoch == a.epochs:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "critic": critic.state_dict(), "args": margs,
                        "hparams": vars(a), "mean": full_mean, "std": full_std,
                        "withscale": True, "cases": dataset.cases},
                       save_dir / f"epoch_{epoch:05d}.pth")
    wandb.finish()


if __name__ == "__main__":
    main()
