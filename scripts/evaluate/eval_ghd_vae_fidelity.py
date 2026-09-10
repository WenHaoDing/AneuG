"""FPD / KPD / TMD for every stage-1 GHD VAE, scored by the MorphoFormer sensor.

WHAT THE THREE NUMBERS ARE (AneuG paper, Table 1):
  FPD  Frechet distance between real and generated feature sets. Lower better.
  KPD  unbiased polynomial-kernel MMD^2 on the same features. Lower better.
  TMD  average pairwise distance WITHIN the generated set: diversity.

The paper uses a pre-trained PointNet++ for FPD/KPD. We substitute the
morphology_sensor's pooled embedding, which is supervised on dome + phi +
rotation and so is forced to encode whole-shape morphology. Values are
therefore NOT comparable to the paper's, only to each other.

TMD is marked "higher is better" in the paper, which is only safe alongside a
good KPD -- a generator emitting noise wins TMD outright. So the REAL corpus's
own TMD is printed as the target: you want to match it, not beat it.

TMD costs nothing here. Vertices correspond across meshes of a type and the GHD
eigenvectors are orthonormal (verified, max|U'U - I| = 5e-6), so
    ||V_a - V_b||_F = norm_canonical * ||phi_a - phi_b||_F
exactly. Reported as RMS vertex displacement, so the two types are comparable.

TWO REFERENCE SETS, deliberately:
  unseen-16   the cases the sensor never trained on. Unbiased, but 16 samples
              makes KPD noisy.
  all-523     stable, but the sensor trained on these, so generated shapes are
              out-of-distribution to it in a way real ones are not.
If the two disagree on the RANKING, neither should be trusted and the sensor
needs k-fold retraining. Agreement is the evidence that the ranking is real.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from models.morphoformer import MorphoFormer
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from dataset.morpho_dataset import MorphoDataset
from utils.generate_synthetic import load_ghd_vae

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
MORPHO_ROOT = ROOT / "runtime_dataset" / "AneuG_morpho"


def kpd(x, y, degree=3, coef0=1.0, n_subsets=100, seed=0):
    """Unbiased MMD^2, polynomial kernel. Zero in expectation for two samples of
    the same distribution at ANY sample size, which is why it beats Frechet on a
    few hundred meshes."""
    rng = np.random.default_rng(seed)
    gamma = 1.0 / x.shape[1]
    m = min(len(x), len(y))
    vals = []
    for _ in range(n_subsets):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        kxx = (gamma * a @ a.T + coef0) ** degree
        kyy = (gamma * b @ b.T + coef0) ** degree
        kxy = (gamma * a @ b.T + coef0) ** degree
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals))


def fpd(x, y, pca_dim=16, eps=1e-6):
    """Frechet distance, after PCA. A covariance from tens of samples in 128
    dimensions is near-singular and the distance would be dominated by noise in
    its smallest eigenvalues."""
    from scipy import linalg
    k = min(pca_dim, min(len(x), len(y)) - 2, x.shape[1])
    if k > 0 and k < x.shape[1]:
        both = np.vstack([x, y]); mu = both.mean(0)
        _, _, vt = np.linalg.svd(both - mu, full_matrices=False)
        w = vt[:k].T
        x, y = (x - mu) @ w, (y - mu) @ w
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1])
    cy = np.cov(y, rowvar=False) + eps * np.eye(y.shape[1])
    cm, _ = linalg.sqrtm(cx @ cy, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mx - my) ** 2).sum() + np.trace(cx + cy - 2 * cm))


def _cd(a, b):
    return np.sqrt(np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2*a@b.T, 0))


def one_nna(x, y, n_rep=20, seed=0):
    """1-NN two-sample accuracy. 0.5 ideal. Above: separable. BELOW 0.5 is not
    better -- generated samples sitting closer to real ones than real ones do to
    each other is the signature of memorisation.

    The two sets MUST be equal size. With 16 real against 523 generated, almost
    every nearest neighbour is generated simply because generated points are 33x
    more numerous, and the statistic reads ~0.97 regardless of the generator.
    So the larger set is subsampled to match, and averaged over repeats."""
    rng = np.random.default_rng(seed)
    m = min(len(x), len(y))
    acc = []
    for _ in range(n_rep if m < max(len(x), len(y)) else 1):
        a = x[rng.choice(len(x), m, replace=False)]
        b = y[rng.choice(len(y), m, replace=False)]
        f = np.vstack([a, b]); lab = np.r_[np.zeros(m), np.ones(m)]
        d = _cd(f, f); np.fill_diagonal(d, np.inf)
        acc.append((lab[d.argmin(1)] == lab).mean())
    return float(np.mean(acc))


def precision_recall(real, gen, k=3):
    """precision = generated samples inside the real manifold (plausibility),
    recall = real samples inside the generated manifold (coverage)."""
    def radii(f):
        d = _cd(f, f); np.fill_diagonal(d, np.inf)
        return np.sort(d, 1)[:, min(k, len(f) - 1) - 1]
    d = _cd(gen, real)
    return (float((d <= radii(real)[None, :]).any(1).mean()),
            float((d.T <= radii(gen)[None, :]).any(1).mean()))


def tmd(phi, types, norms, nverts):
    """Mean pairwise RMS vertex displacement within a set. Exact, via the
    orthonormal GHD basis. Computed per type, then averaged by type share, since
    the two templates have different vertex counts."""
    out, w = [], []
    for t in sorted(set(int(x) for x in types)):
        P = phi[types == t].reshape((types == t).sum(), -1)
        if len(P) < 2:
            continue
        D = _cd(P, P) * norms[t] / np.sqrt(nverts[t])
        iu = np.triu_indices(len(P), 1)
        out.append(D[iu].mean()); w.append(len(P))
    return float(np.average(out, weights=w)) if out else float("nan")


@torch.no_grad()
def embed(model, phi, types, device, batch=8):
    out = []
    for i in range(0, len(phi), batch):
        p = torch.as_tensor(np.asarray(phi[i:i+batch]), dtype=torch.float32, device=device)
        t = torch.as_tensor(np.asarray(types[i:i+batch]), dtype=torch.long, device=device)
        out.append(model.embed(p, t).float().cpu().numpy())
    return np.concatenate(out, 0)


@torch.no_grad()
def generate(ckpt, n, types, device, z_amp=1.0, seed=0):
    vae, mean, std, gdim = load_ghd_vae(ckpt, device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.as_tensor(types, dtype=torch.long, device=device)
    z = (torch.randn(n, vae.latent_dim, generator=g) * z_amp).to(device)
    with_scale = mean.numel() > gdim
    out = vae.decode(z, t, strip_scale=True) if with_scale else vae.decode(z, t)
    phi_n = out[0] if isinstance(out, tuple) else out
    m = mean[..., :gdim] if with_scale else mean
    s = std[..., :gdim] if with_scale else std
    return (phi_n * s + m).reshape(n, -1, 3).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--stage1-dir", default=str(ROOT / "runtime_train" / "ghd_vae" / "stage1"))
    ap.add_argument("--epoch-name", default="epoch_05000.pth")
    ap.add_argument("--n-gen", type=int, default=523)
    ap.add_argument("--test-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(args.device)

    ck = torch.load(args.sensor, map_location=dev, weights_only=False)
    recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=dev)
    model = MorphoFormer(recon, **ck["args"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    norms = {t: float(recon.get(t).norm_canonical) for t in (0, 1)}
    nverts = {0: 4143, 1: 2590}

    ds = MorphoDataset(MORPHO_ROOT)
    all_cases = [ds.samples[i]["case"] for i in range(len(ds))]
    r_phi = np.stack([ds[i]["phi"].numpy() for i in range(len(ds))])
    r_types = np.array([int(ds[i]["aneurysm_type"]) for i in range(len(ds))])
    unseen = set(sorted(sorted(all_cases, key=lambda c: hashlib.md5(c.encode()).hexdigest())
                        [:args.test_size]))
    umask = np.array([c in unseen for c in all_cases])
    print(f"corpus {len(ds)}   unseen-by-sensor {int(umask.sum())}   "
          f"types {np.bincount(r_types).tolist()}")

    feats = {"unseen-16": embed(model, r_phi[umask], r_types[umask], dev),
             "all-523":   embed(model, r_phi, r_types, dev)}
    real_tmd = tmd(r_phi, r_types, norms, nverts)

    # real-vs-real floor: nothing below this is resolvable
    rng = np.random.default_rng(args.seed)
    floors = {}
    for k, f in feats.items():
        idx = rng.permutation(len(f)); h = len(f) // 2
        floors[k] = kpd(f[idx[:h]], f[idx[h:2*h]], seed=args.seed)

    runs = sorted(d for d in Path(args.stage1_dir).iterdir()
                  if d.is_dir() and (d / args.epoch_name).exists())
    print(f"\n{len(runs)} run(s) with {args.epoch_name}. "
          f"real TMD = {real_tmd:.4f}  (target, not a ceiling)")
    print(f"KPD floor: unseen-16 {floors['unseen-16']:+.5f}   all-523 {floors['all-523']:+.5f}\n")

    gtypes = r_types[rng.integers(0, len(r_types), args.n_gen)]
    rows = []
    for d in runs:
        gphi = generate(d / args.epoch_name, args.n_gen, gtypes, dev, seed=args.seed)
        gf = embed(model, gphi, gtypes, dev)
        r = {"run": d.name, "tmd": tmd(gphi, gtypes, norms, nverts)}
        for k, f in feats.items():
            p, rc = precision_recall(f, gf)
            r[k] = (kpd(f, gf, seed=args.seed), fpd(f, gf), one_nna(f, gf), p, rc)
        rows.append(r)
        print(f"  scored {d.name}")

    for key in ("unseen-16", "all-523"):
        print(f"\n=== reference: {key} (KPD floor {floors[key]:+.5f}) ===")
        hdr = (f"{'run':<34}{'KPD':>10}{'FPD':>9}{'TMD':>9}"
               f"{'1-NNA':>8}{'prec':>7}{'rec':>7}")
        print(hdr); print("-" * len(hdr))
        for r in sorted(rows, key=lambda r: r[key][0]):
            k_, f_, n_, p_, c_ = r[key]
            # full name, no truncation: the adv weight lives at the END of a
            # GAN run's name, so clipping made three different runs identical
            name = r["run"].replace("ghd_vae_", "")
            print(f"{name:<34}{k_:>10.5f}{f_:>9.3f}{r['tmd']:>9.4f}"
                  f"{n_:>8.3f}{p_:>7.3f}{c_:>7.3f}")
        print(f"{'REAL CORPUS':<34}{'':>10}{'':>9}{real_tmd:>9.4f}{0.5:>8.3f}")

    a = [r["run"] for r in sorted(rows, key=lambda r: r["unseen-16"][0])]
    b = [r["run"] for r in sorted(rows, key=lambda r: r["all-523"][0])]
    from scipy.stats import spearmanr
    rho = spearmanr([a.index(x) for x in a], [b.index(x) for x in a]).correlation
    print(f"\nranking agreement between the two reference sets: Spearman {rho:+.3f}")
    print("Low agreement would mean the sensor's bias, not generator quality, is "
          "driving the ordering.")


if __name__ == "__main__":
    main()
