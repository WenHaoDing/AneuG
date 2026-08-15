"""
Standalone evaluation for the endcap predictor (models.endcap_predictor).
Loads a trained checkpoint and renders, per branch slot: the predicted CAP
region (loc_head's per-vertex probability distribution, top-K vertices —
same visual convention as dataset/preprocess_endcaps.py's ground-truth
in_patch scatter), the predicted ENDPOINT (soft-argmax point), and the
predicted TANGENT (arrow). Two panels:
  - real cases: ground truth (black patch/endpoint/tangent) vs. prediction
    (colored) — note there's no held-out split for this model (see
    scripts/train/train_endcap_predictor.py's module docstring: the real
    target is unseen SYNTHETIC shapes, not held-out real ones), so these
    cases were seen during training — useful as a "did it learn anything at
    all" check, not a generalization check.
  - synthetic cases: phi sampled fresh from the frozen stage-1 GHD VAE, no
    ground truth to compare against — this panel is the one that actually
    matters for this model's purpose.

conda activate new
python scripts/evaluate/eval_endcap_predictor.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.endcap_dataset import EndcapDataset
from models.endcap_predictor import EndcapPredictor
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from utils.generate_synthetic import load_ghd_vae

# ── eval config ───────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "tr_checkpoints" / "v2_1" / "endcap_predictor" / "h32_gps2_tw1_pw0.1" / "epoch_02000.pth"
GHD_VAE_CKPT = ROOT / "tr_checkpoints" / "v2_1" / "stage1" / "ghd_vae_h512_z16_kl0.5" / "epoch_05000.pth"
PROCESSED_ROOT = ROOT / "dataset" / "processed_endcaps"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"

N_REAL      = 8
N_SYNTHETIC = 8
NCOLS       = 4
TOP_K       = 30   # predicted cap = top-K highest-probability vertices per branch (matches typical real patch size)
SEED        = 0
DEVICE      = "cuda:0"
NUM_TYPES   = 3   # type 2 merged into type 1 everywhere — never sampled

# Candidate branch count per type (bifurcated=3, sidewall=2) — for synthetic
# samples there's no real branch_mask to gate on (unlike eval_real, which
# uses item["branch_mask"]), so the redundant slot(s) beyond a type's real
# opening count must be masked out here instead. Slot 2 never gets gradient
# from any loss for a type-1/2 sample during training (branch_mask[2]=False
# there), so its prediction is untrained noise, not a real branch — plotting
# it unmasked would be actively misleading.
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}

SAVE_DIR = CHECKPOINT.parent / "eval"
BRIGHT = ["#e6194B", "#3cb44b", "#4363d8"]


def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device)
    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=device)
    model = EndcapPredictor(multi_recon, **ckpt["args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded endcap predictor: {CHECKPOINT} (epoch {ckpt['epoch']})")
    return model, multi_recon


@torch.no_grad()
def predict(model, phi, aneurysm_type):
    """Runs the model and additionally recovers dense vertex positions
    (needed to plot the cap region — not part of the model's own return
    value) by redoing the same to_dense_batch step the model does internally,
    on the same mesh reconstruction — deterministic, so this is guaranteed to
    line up with the model's own pos_dense without touching its interface."""
    pyg_batch = model.multi_recon.to_pyg_batch(
        phi, aneurysm_type, point_std=None, include_normals=model.use_normals)
    pos_dense, _ = to_dense_batch(pyg_batch.x[:, :3], pyg_batch.batch)
    endpoint, tangent, loc_probs, token_mask = model(pyg_batch=pyg_batch)
    return endpoint, tangent, loc_probs, token_mask, pos_dense


def predicted_cap_points(loc_probs_row, pos_row, token_mask_row, top_k=TOP_K):
    """loc_probs_row/token_mask_row: [N]; pos_row: [N,3]. Returns (points
    [k,3], weights [k] normalized within the top-k set) — the branch's
    predicted cap region, mirroring how preprocess_endcaps.py visualizes the
    ground-truth in_patch region as a scatter of nearby vertices."""
    probs = loc_probs_row.clone()
    probs[~token_mask_row] = -1.0
    k = min(top_k, int(token_mask_row.sum().item()))
    top_vals, top_idx = torch.topk(probs, k)
    pts = pos_row[top_idx].cpu().numpy()
    weights = (top_vals / top_vals.sum().clamp(min=1e-8)).cpu().numpy()
    return pts, weights


def plot_case(ax, verts, faces, panels, title):
    """panels: list of dicts, one per branch, each with keys:
    color, cap_pts, cap_weights, endpoint, tangent, and optionally
    gt_patch_pts, gt_endpoint, gt_tangent (real-case comparison only)."""
    from dataset.preprocess_ImperialNHS import _set_axes_equal

    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.15)
    all_pts = [verts]

    for p in panels:
        c = p["color"]
        if p.get("gt_patch_pts") is not None and len(p["gt_patch_pts"]) > 0:
            ax.scatter(*p["gt_patch_pts"].T, s=8, color="black", alpha=0.4)
            all_pts.append(p["gt_patch_pts"])
        if p.get("gt_endpoint") is not None:
            ax.scatter(*p["gt_endpoint"], s=60, marker="o", color="black")
            all_pts.append(p["gt_endpoint"][None])
        if p.get("gt_tangent") is not None:
            ax.quiver(*p["gt_endpoint"], *(p["gt_tangent"] * 2.0), color="black",
                      linewidth=1.5, arrow_length_ratio=0.3)

        if len(p["cap_pts"]) > 0:
            ax.scatter(p["cap_pts"][:, 0], p["cap_pts"][:, 1], p["cap_pts"][:, 2],
                      s=10 + 40 * p["cap_weights"] / p["cap_weights"].max(),
                      color=c, alpha=0.6)
            all_pts.append(p["cap_pts"])
        ax.scatter(*p["endpoint"], s=60, marker="x", color=c)
        ax.quiver(*p["endpoint"], *(p["tangent"] * 2.0), color="yellow",
                  linewidth=1.5, arrow_length_ratio=0.3)
        all_pts += [p["endpoint"][None]]

    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    _set_axes_equal(ax, *all_pts)


def eval_real(model, dataset, save_path):
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy

    n = min(N_REAL, len(dataset))
    nrows = (n + NCOLS - 1) // NCOLS
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4 * NCOLS, 4 * nrows))
    for i in range(n):
        item = dataset[i]
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        atype_t = item["aneurysm_type"].unsqueeze(0).to(DEVICE)
        endpoint, tangent, loc_probs, token_mask, pos_dense = predict(model, phi, atype_t)

        atype = int(item["aneurysm_type"])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": item["phi"].numpy()}}, denormalize_shape=True)

        panels = []
        for b in range(model.max_branches):
            if not item["branch_mask"][b]:
                continue
            cap_pts, cap_w = predicted_cap_points(loc_probs[0, b], pos_dense[0], token_mask[0])
            gt_in_patch = item["in_patch"][b].numpy().astype(bool)
            gt_verts = pos_dense[0].cpu().numpy()[:gt_in_patch.shape[0]]
            panels.append({
                "color": BRIGHT[b % len(BRIGHT)],
                "cap_pts": cap_pts, "cap_weights": cap_w,
                "endpoint": endpoint[0, b].cpu().numpy(),
                "tangent": tangent[0, b].cpu().numpy(),
                "gt_patch_pts": gt_verts[gt_in_patch] if gt_in_patch.any() else None,
                "gt_endpoint": item["endpoints"][b].numpy(),
                "gt_tangent": item["tangents"][b].numpy(),
            })

        ax = fig.add_subplot(nrows, NCOLS, i + 1, projection="3d")
        plot_case(ax, verts, faces, panels, f"{item['case']} (type {atype})")

    fig.suptitle("real cases (black=ground truth, colored=predicted cap/endpoint, yellow=predicted tangent)\n"
                "seen during training — no held-out split for this model", fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"Saved → {save_path}")


def eval_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, save_path):
    from dataset.preprocess_ImperialNHS import _reconstruct_ghd_numpy
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    g = torch.Generator(device="cpu").manual_seed(SEED)
    types = torch.randint(0, NUM_TYPES - 1, (N_SYNTHETIC,), generator=g).to(DEVICE)
    z_ghd = torch.randn(N_SYNTHETIC, ghd_vae.latent_dim, generator=g).to(DEVICE)
    ghd_n, _ = ghd_vae.decode(z_ghd, types)
    phi = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(N_SYNTHETIC, -1, 3)

    endpoint, tangent, loc_probs, token_mask, pos_dense = predict(model, phi, types)

    nrows = (N_SYNTHETIC + NCOLS - 1) // NCOLS
    fig = plt.figure(figsize=(4 * NCOLS, 4 * nrows))
    for i in range(N_SYNTHETIC):
        atype = int(types[i])
        verts, faces = _reconstruct_ghd_numpy(
            {"aneurysm_type": atype, "ghd": {"phi": phi[i].cpu().numpy()}}, denormalize_shape=True)

        panels = []
        n_open = TYPE_N_OPEN.get(atype, model.max_branches)
        for b in range(n_open):
            cap_pts, cap_w = predicted_cap_points(loc_probs[i, b], pos_dense[i], token_mask[i])
            panels.append({
                "color": BRIGHT[b % len(BRIGHT)],
                "cap_pts": cap_pts, "cap_weights": cap_w,
                "endpoint": endpoint[i, b].cpu().numpy(),
                "tangent": tangent[i, b].cpu().numpy(),
            })

        ax = fig.add_subplot(nrows, NCOLS, i + 1, projection="3d")
        plot_case(ax, verts, faces, panels, f"type {atype}")

    fig.suptitle("synthetic generation (colored=predicted cap/endpoint, yellow=predicted tangent) — "
                "no ground truth, this is the case that actually matters", fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    model, multi_recon = load_model(DEVICE)
    dataset = EndcapDataset(PROCESSED_ROOT, max_branches=model.max_branches)

    eval_real(model, dataset, SAVE_DIR / "eval_real.png")

    ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
    for p in ghd_vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")
    eval_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, SAVE_DIR / "eval_synthetic.png")


if __name__ == "__main__":
    main()

"""
conda activate new
python scripts/evaluate/eval_endcap_predictor.py
"""
