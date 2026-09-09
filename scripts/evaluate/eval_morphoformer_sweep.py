"""
Held-out evaluation for MorphoFormer checkpoints, as a comparison table.

Reads the training loss curves only to locate runs; every number here is
recomputed on the HELD-OUT cases the run never trained on (the split is a hash
of the case name, so it is reproduced exactly rather than trusted from a log).

Reports interpretable quantities, not the raw training losses:

  endpoint    mean L2 distance, prediction to ground-truth cap centroid
  tangent     mean ANGLE in degrees -- (1 - cos) hides how big an error is;
              0.13 sounds small but is 29 degrees
  cap IoU     intersection-over-union of the predicted cap region against the
              hand-labelled patch. This is the number that says whether the
              model can actually open a mesh, which endpoint/tangent do not.
  cap Dice    same regions, Dice -- more forgiving of boundary disagreement,
              which matters because the boundaries were brushed by hand
  dome IoU    same, for the dome mask (morphology_sensor only)
  phi MSE, and rotation-recovery error in degrees (morphology_sensor only)

    python scripts/evaluate/eval_morphoformer_sweep.py --test-size 16
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

from dataset.morpho_dataset import MorphoDataset, collate_morpho
from models.morphoformer import MorphoFormer, RandomSO3Rotation
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
MORPHO_ROOT = ROOT / "runtime_dataset" / "AneuG_morpho"
ASSEMBLED = ROOT / "runtime_dataset" / "assembled"
PROCESSED = ROOT / "runtime_dataset" / "AneuG_processed"


def region_from_probs(p, frac=0.2, min_ratio=3.0, max_k=400):
    """Same rule the sanity panels use, so the table and the pictures agree."""
    n = p.size
    peak = float(p.max())
    if peak <= 0:
        return np.zeros(n, dtype=bool)
    idx = np.flatnonzero(p >= max(frac * peak, min_ratio / n))
    if idx.size > max_k:
        idx = idx[np.argsort(p[idx])[-max_k:]]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def iou_dice(a, b):
    inter = float(np.logical_and(a, b).sum())
    if a.sum() + b.sum() == 0:
        return float("nan"), float("nan")
    union = float(np.logical_or(a, b).sum())
    return (inter / union if union else float("nan"),
            2 * inter / float(a.sum() + b.sum()))


@torch.no_grad()
def evaluate(ckpt_path, test_cases, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args, hp = ck["args"], ck.get("hparams", {})
    recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=device)
    model = MorphoFormer(recon, **args).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = MorphoDataset(MORPHO_ROOT, cases=test_cases,
                       max_branches=args.get("max_branches", 3),
                       with_dome=args.get("predict_dome", False),
                       assembled_root=ASSEMBLED,
                       with_affine=False)

    ep_err, tg_deg, ious, dices, d_iou, d_dice, phi_e, aff_e = [], [], [], [], [], [], [], []
    for i in range(len(ds)):
        item = ds[i]
        phi = item["phi"].unsqueeze(0).to(device)
        at = item["aneurysm_type"].unsqueeze(0).to(device)
        eppred, tgpred, loc, tok, extra = model(phi, at)
        n_v = int(tok[0].sum())
        for b in range(args.get("max_branches", 3)):
            if not bool(item["branch_mask"][b]):
                continue
            ep_err.append(float(torch.linalg.norm(
                eppred[0, b].cpu() - item["endpoints"][b])))
            cos = float(torch.dot(torch.nn.functional.normalize(tgpred[0, b].cpu(), dim=0),
                                  torch.nn.functional.normalize(item["tangents"][b], dim=0)))
            tg_deg.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
            pred = region_from_probs(loc[0, b, :n_v].cpu().numpy())
            gt = np.asarray(item["in_patch"][b])[:n_v]
            i_, d_ = iou_dice(pred, gt)
            ious.append(i_); dices.append(d_)
        if "dome_logits" in extra and "dome" in item:
            pd = (torch.sigmoid(extra["dome_logits"][0, :n_v]).cpu().numpy() > 0.5)
            gd = np.asarray(item["dome"])[:n_v]
            i_, d_ = iou_dice(pd, gd)
            d_iou.append(i_); d_dice.append(d_)
        if "phi_pred" in extra:
            phi_e.append(float(torch.nn.functional.mse_loss(
                extra["phi_pred"][0].cpu(), item["phi"].flatten())))
        if args.get("predict_rotation"):
            # Give the model a mesh turned by a KNOWN random rotation and ask how
            # far off its answer is, in degrees. Scoring it on the unrotated mesh
            # would only ask whether it can say "no rotation", and the previous
            # version scored it against the stored Stage-2 pose, which the input
            # never carried at all.
            pyg = model.multi_recon.to_pyg_batch(
                phi, at, point_std=None, include_normals=model.use_normals)
            Q = RandomSO3Rotation.sample(1, phi.dtype, device)
            pyg = RandomSO3Rotation.apply_to_data(pyg, Q)
            pred6 = model(pyg_batch=pyg)[4]["rotation_pred"][0]
            # 6D -> rotation matrix by Gram-Schmidt, the standard decoding.
            a1, a2 = pred6[:3], pred6[3:]
            b1 = torch.nn.functional.normalize(a1, dim=0)
            b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
            R_pred = torch.stack([b1, b2, torch.cross(b1, b2, dim=0)], dim=1)
            cos = ((R_pred.T @ Q[0]).diagonal().sum() - 1) / 2
            aff_e.append(float(np.degrees(np.arccos(np.clip(float(cos), -1, 1)))))

    m = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    return {"hidden": args.get("hidden"), "variant": hp.get("variant", "?"),
            "epoch": ck.get("epoch"), "n_cases": len(ds),
            "endpoint": m(ep_err), "tangent_deg": m(tg_deg),
            "cap_iou": m(ious), "cap_dice": m(dices),
            "dome_iou": m(d_iou), "dome_dice": m(d_dice),
            "phi_mse": m(phi_e), "rot_deg": m(aff_e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runtime_train" / "morphoformer"))
    ap.add_argument("--epoch", default=None, help="Checkpoint to use (default: latest per run).")
    ap.add_argument("--test-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Reproduce the split rather than trusting the log: same hash, same order.
    all_cases = sorted(p.stem for p in MORPHO_ROOT.glob("*.npy"))
    order = sorted(all_cases, key=lambda c: hashlib.md5(c.encode()).hexdigest())
    test_cases = sorted(order[:args.test_size])

    ckpts = []
    for d in sorted(Path(args.root).glob("*/*/")):
        found = sorted(d.glob("epoch_*.pth"))
        if not found:
            continue
        ckpts.append(found[-1] if args.epoch is None else d / f"epoch_{int(args.epoch):05d}.pth")

    device = torch.device(args.device)
    rows = []
    for c in ckpts:
        if not c.exists():
            print(f"  (missing {c})"); continue
        try:
            rows.append((c, evaluate(c, test_cases, device)))
        except Exception as exc:
            print(f"  FAILED {c.parent.name}: {type(exc).__name__}: {exc}")

    print(f"\nHeld-out evaluation on {len(test_cases)} cases "
          f"(never trained on; split reproduced by case-name hash)\n")
    hdr = (f"{'run':<46}{'ep':>6}{'endpoint':>10}{'tang deg':>10}{'cap IoU':>9}"
           f"{'cap Dice':>10}{'dome IoU':>10}{'phi MSE':>10}{'rot deg':>10}")
    print(hdr); print("-" * len(hdr))
    for c, r in rows:
        print(f"{c.parent.name[:45]:<46}{r['epoch']:>6}{r['endpoint']:>10.3f}"
              f"{r['tangent_deg']:>10.1f}{r['cap_iou']:>9.3f}{r['cap_dice']:>10.3f}"
              f"{r['dome_iou']:>10.3f}{r['phi_mse']:>10.4f}{r['rot_deg']:>10.2f}")


if __name__ == "__main__":
    main()
