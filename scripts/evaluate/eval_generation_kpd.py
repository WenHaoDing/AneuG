"""
FPD / KPD for GHD-VAE generation, using MorphoFormer as the feature extractor.

WHY THIS EXTRACTOR. FID-style metrics need a feature space in which distance
means "different shape". Training a classifier on ~520 meshes with 2 classes
would give about one bit of signal; a marker regressor would collapse to the
markers themselves. The morphology_sensor is neither: it is supervised on the
dome (sac shape, not just openings) and on phi + affine (the whole geometry),
so its pooled embedding has to carry global morphology. Same argument as using
Inception for FID, with the extractor domain-matched instead of borrowed.

THREE THINGS THAT MAKE THE NUMBER MEAN ANYTHING, each easy to get wrong:

  1. The real reference features must come from meshes the extractor NEVER
     TRAINED ON. Otherwise the distance mixes the generator's error with how
     out-of-distribution the generated shapes are to the extractor, and that
     second term varies per generator, so it can reorder a ranking.

  2. A real-vs-real FLOOR. Split the real reference in half and measure the
     same distance. If generated-vs-real is not clearly above the floor, the
     metric cannot resolve what is being asked of it and its absolute value is
     noise.

  3. A DEGRADATION LADDER. Deliberately worsen the generator (shrink or inflate
     z, add noise to phi, emit the prior mean) and check that the metric rises
     monotonically. A metric that cannot separate the mean shape from real data
     is decoration, however principled its derivation.

WHY FOLDS, AND WHY THE METRIC IS AVERAGED RATHER THAN POOLED. Requirement 1
caps the reference set at the held-out split -- 16 cases for the sweep runs,
far too few for a stable estimate. k-fold fixes that: every case is held out by
exactly one fold. But features from two folds come from two separately trained
networks, so they live in DIFFERENT embedding spaces and must never be
concatenated into one feature matrix. The metric is therefore computed WITHIN
each fold and averaged across folds, with the spread reported. Pass every fold
checkpoint to --sensor.

KPD IS THE PRIMARY NUMBER, not FPD. Frechet distance is biased at small sample
size -- which is why KID was introduced -- and each fold contributes only tens
of real meshes. KPD is an unbiased MMD^2 estimator and assumes no Gaussianity.
FPD is reported alongside for comparability, after PCA, because a covariance
estimated from tens of samples in 32-128 dimensions is badly conditioned.

    python scripts/evaluate/eval_generation_kpd.py \
        --sensor runtime_train/morphoformer/morphology_sensor/*/epoch_01400.pth \
        --ghd-vae runtime_train/ghd_vae/stage1/<cfg>/epoch_05000.pth
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


# ── distribution metrics ─────────────────────────────────────────────────────

def kpd(x, y, degree=3, gamma=None, coef0=1.0, n_subsets=100, seed=0):
    """Unbiased MMD^2 with a polynomial kernel -- the KID estimator.

    Unbiased means the self-similarity diagonals are excluded, so the expected
    value is 0 for two samples of the SAME distribution at any sample size.
    That is the whole reason to prefer it here over Frechet. Averaged over
    random subsets to expose the estimator's own variance.
    """
    rng = np.random.default_rng(seed)
    gamma = 1.0 / x.shape[1] if gamma is None else gamma
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


def fpd(x, y, pca_dim=None, eps=1e-6):
    """Frechet distance between Gaussians fitted to the two feature sets.

    PCA first: a covariance estimated from tens of samples in the full feature
    dimension is near-singular, and the distance is then dominated by noise in
    its smallest eigenvalues rather than by shape.
    """
    from scipy import linalg
    if pca_dim and pca_dim < x.shape[1]:
        both = np.vstack([x, y])
        mu = both.mean(0)
        _, _, vt = np.linalg.svd(both - mu, full_matrices=False)
        w = vt[:pca_dim].T
        x, y = (x - mu) @ w, (y - mu) @ w
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1])
    cy = np.cov(y, rowvar=False) + eps * np.eye(y.shape[1])
    covmean, _ = linalg.sqrtm(cx @ cy, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(cx + cy - 2 * covmean))


def _cdist(a, b):
    return np.sqrt(np.maximum(
        (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0))


def one_nna(x, y):
    """1-nearest-neighbour two-sample accuracy. 0.5 is ideal.

    Above 0.5 the two sets are separable (generated shapes are distinguishable
    from real). BELOW 0.5 is not better -- it means generated samples sit
    closer to real ones than real ones do to each other, the signature of
    over-fitting or mode collapse onto the training shapes.
    """
    f = np.vstack([x, y])
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    d = _cdist(f, f)
    np.fill_diagonal(d, np.inf)
    return float((lab[d.argmin(1)] == lab).mean())


def precision_recall(real, gen, k=3):
    """Kynkaanniemi k-NN manifold precision/recall.

    precision = fraction of GENERATED samples inside the real manifold (are
    they plausible), recall = fraction of REAL samples inside the generated
    manifold (is the diversity covered). They separate the two failure modes
    that a single distance number conflates.
    """
    def radii(f):
        d = _cdist(f, f)
        np.fill_diagonal(d, np.inf)
        return np.sort(d, 1)[:, k - 1]
    rr, rg = radii(real), radii(gen)
    d = _cdist(gen, real)
    precision = float((d <= rr[None, :]).any(1).mean())
    recall = float((d.T <= rg[None, :]).any(1).mean())
    return precision, recall


# ── features ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed(model, phi, types, device, batch=8):
    out = []
    for i in range(0, len(phi), batch):
        p = torch.as_tensor(np.asarray(phi[i:i + batch]), dtype=torch.float32, device=device)
        t = torch.as_tensor(np.asarray(types[i:i + batch]), dtype=torch.long, device=device)
        out.append(model.embed(p, t).float().cpu().numpy())
    return np.concatenate(out, 0)


def load_real(cases):
    """phi and (type-2-merged) aneurysm types, via the training dataset class
    so every convention matches what the extractor was trained on."""
    ds = MorphoDataset(MORPHO_ROOT, cases=list(cases))
    phi = np.stack([ds[i]["phi"].numpy() for i in range(len(ds))])
    types = np.array([int(ds[i]["aneurysm_type"]) for i in range(len(ds))])
    return phi, types


@torch.no_grad()
def generate(vae, mean, std, ghd_dim, types, device, z_amp=1.0, phi_noise=0.0,
             mean_shape=False, seed=0):
    """Sample the generator. z ~ N(0,1) is the prior the KL term actually
    trains against; the extra knobs exist only to build the degradation ladder."""
    n = len(types)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.as_tensor(types, dtype=torch.long, device=device)
    z = torch.zeros(n, vae.latent_dim) if mean_shape else \
        torch.randn(n, vae.latent_dim, generator=g) * z_amp
    z = z.to(device)
    with_scale = mean.numel() > ghd_dim
    phi_n = vae.decode(z, t, strip_scale=True) if with_scale else vae.decode(z, t)
    m = mean[..., :ghd_dim] if with_scale else mean
    s = std[..., :ghd_dim] if with_scale else std
    phi = (phi_n * s + m).reshape(n, -1, 3)
    if phi_noise > 0:
        phi = phi + phi_noise * phi.std() * torch.randn(
            phi.shape, generator=g).to(device)
    return phi.cpu().numpy()


LADDER = [("generator  z~N(0,1)", dict()),
          ("z_amp 0.5  under-dispersed", dict(z_amp=0.5)),
          ("z_amp 2.0  over-dispersed", dict(z_amp=2.0)),
          ("phi + 25% noise", dict(phi_noise=0.25)),
          ("prior mean shape only", dict(mean_shape=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", nargs="+", required=True,
                    help="morphology_sensor checkpoint(s); pass every fold.")
    ap.add_argument("--ghd-vae", required=True, help="stage-1 GHD VAE checkpoint.")
    ap.add_argument("--test-size", type=int, default=16,
                    help="Fallback held-out size when a checkpoint records none.")
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--per-type", action="store_true",
                    help="Also report each aneurysm type separately. The two "
                         "types use different canonical templates and therefore "
                         "different eigenbases, so a pooled number can hide one "
                         "type being generated badly.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)

    vae, gmean, gstd, gdim = load_ghd_vae(args.ghd_vae, device)
    recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=device)
    rows = {name: [] for name, _ in LADDER}
    floor = []

    for ci, ckpt_path in enumerate(args.sensor):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not ck["args"].get("predict_dome"):
            print(f"WARNING: {Path(ckpt_path).name} has no dome head -- that is an "
                  f"uncapper, not a morphology sensor. Its features were never "
                  f"pushed to encode sac shape; the metric will be weak.")
        model = MorphoFormer(recon, **ck["args"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        held = list(ck.get("held_out_cases") or [])
        if not held:
            allc = sorted(p.stem for p in MORPHO_ROOT.glob("*.npy"))
            held = sorted(sorted(allc, key=lambda c: hashlib.md5(c.encode()).hexdigest())
                          [:args.test_size])
            print(f"  {Path(ckpt_path).parent.name}: no held-out list recorded, "
                  f"reproduced {len(held)} by hash")
        rphi, rtypes = load_real(held)
        real = embed(model, rphi, rtypes, device)
        print(f"fold {ci}: {Path(ckpt_path).parent.name}  "
              f"{len(real)} unseen real case(s), feature dim {real.shape[1]}")

        # real-vs-real floor: the metric resolves nothing below this
        idx = np.random.default_rng(args.seed).permutation(len(real))
        h = len(real) // 2
        floor.append(kpd(real[idx[:h]], real[idx[h:2 * h]], seed=args.seed))

        # match the fold's own type composition, so the comparison never
        # mixes a difference in shape with a difference in type mix
        gtypes = rtypes[np.random.default_rng(args.seed).integers(0, len(rtypes), len(rtypes))]
        for name, kw in LADDER:
            gphi = generate(vae, gmean, gstd, gdim, gtypes, device, seed=args.seed, **kw)
            gen = embed(model, gphi, gtypes, device)
            p, r = precision_recall(real, gen)
            rows[name].append((kpd(real, gen, seed=args.seed),
                               fpd(real, gen, pca_dim=min(args.pca_dim, len(real) - 2)),
                               one_nna(real, gen), p, r))

    def fmt(v):
        a = np.array(v)
        return a.mean(0), (a.std(0) if len(a) > 1 else np.zeros(a.shape[1]))

    n_f = len(args.sensor)
    print(f"\n{'':<30}{'KPD':>10}{'FPD':>10}{'1-NNA':>9}{'prec':>8}{'rec':>8}   "
          f"(mean over {n_f} fold{'s' if n_f > 1 else ''})")
    print("-" * 82)
    print(f"{'real vs real  FLOOR':<30}{np.mean(floor):>10.5f}{'':>10}{'0.500':>9}"
          f"{'':>8}{'':>8}")
    for name, _ in LADDER:
        m, s = fmt(rows[name])
        pm = f" +-{s[0]:.4f}" if n_f > 1 else ""
        print(f"{name:<30}{m[0]:>10.5f}{m[1]:>10.3f}{m[2]:>9.3f}{m[3]:>8.3f}"
              f"{m[4]:>8.3f}{pm}")

    print(f"\nRead the ladder before the top row. The generator's KPD is only "
          f"meaningful if it sits clearly above the floor ({np.mean(floor):.5f}) "
          f"and clearly below the degraded settings.")
    if n_f == 1:
        print("SINGLE FOLD: the reference set is just this checkpoint's held-out "
              "cases, so these numbers carry large variance and no spread can be "
              "reported. Pass every fold checkpoint once k-fold training is done.")


if __name__ == "__main__":
    main()
