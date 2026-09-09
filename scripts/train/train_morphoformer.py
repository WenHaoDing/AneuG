"""
Train the endcap predictor (models/morphoformer.py) on
runtime_dataset/AneuG_morpho (see dataset/preprocess_endcaps.py).

Predicts, per branch slot, a point on the mesh surface where a parent vessel
should be capped (soft-argmax over real mesh-vertex positions, weighted by a
per-vertex classifier head) and the tangent direction there — see
models/morphoformer.py's module docstring for the full design rationale
(GCN stem + GPSConv, no SAGPooling/FPS/attention modules, decoupled
localization vs. tangent heads, soft-argmax grounding, graph-distance patch
auxiliary loss).

Three losses, no self-supervised/synthetic term (same reasoning as the
claydoll scripts: nothing here depends on real-vs-generated conditioning
mismatch, since the model only ever sees mesh geometry reconstructed from phi
— identical at train and generation time):
  - endpoint_loss: masked MSE, predicted vs. real endpoint (mm^2).
  - tangent_loss:  masked (1 - cosine similarity), predicted vs. real tangent.
  - patch_loss:    cross-entropy of loc_head's per-vertex distribution against
                   a uniform target over each branch's in_patch vertices —
                   auxiliary signal so that distribution has something to
                   learn from directly, not just indirectly through the
                   endpoint regression loss.

Random SO(3) rotation augmentation (models.morphoformer.RandomSO3Rotation)
is applied here, in prepare_batch, not inside the model — rotating only the
mesh input would leave it inconsistent with the ground-truth endpoint/tangent
targets (only ever known in the canonical frame), so the SAME per-sample
rotations are applied to both the mesh and those targets before either
reaches the model, keeping model.forward() itself augmentation-agnostic.

conda activate new
python scripts/train/train_morphoformer.py
python scripts/train/train_morphoformer.py --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch

ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

from dataset.morpho_dataset import MorphoDataset, collate_morpho
from models.morphoformer import MorphoFormer, RandomSO3Rotation
from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
from utils.generate_synthetic import load_ghd_vae


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu (default: cuda:1).")
    p.add_argument("--variant", choices=("uncapper", "morphology_sensor"),
                   default="morphology_sensor",
                   help="uncapper: caps only -- the practical tool that opens a mesh. "
                        "morphology_sensor: caps + dome + the phi/rotation auxiliary branch "
                        "-- the feature extractor behind the FPD/KPD generation metrics. "
                        "Exactly two models are needed, so this is the only switch: the "
                        "heads are not individually toggleable.")
    p.add_argument("--rotation-weight", type=float, default=0.5)
    p.add_argument("--tangent-weight", type=float, default=1.0)
    p.add_argument("--patch-weight", type=float, default=2.0,
                   help="Weight on the cap-REGION term. Above the tangent term's 1.0: the "
                        "region is what actually opens a mesh, and its cross-entropy floor "
                        "is the target entropy (~4.85), not zero, so only the excess above "
                        "that carries gradient.")
    p.add_argument("--dome-weight", type=float, default=1.0)
    p.add_argument("--phi-weight", type=float, default=0.5)
    p.add_argument("--test-size", type=int, default=0,
                   help="Hold out this many cases from training and render a SECOND sanity "
                        "panel on them each time. Independent of --folds: this is a quick "
                        "look at generalisation during a sweep, not a CV protocol.")
    p.add_argument("--folds", type=int, default=0,
                   help="k-fold cross-validation. 0 (default) trains on everything. Use k>0 "
                        "for the DESCRIPTOR: the FPD/KPD reference features must come from "
                        "meshes the extractor never trained on, or the metric is biased by "
                        "the extractor's own overfitting and its noise floor is unmeasurable.")
    p.add_argument("--fold", type=int, default=0, help="Which fold to hold out (0..folds-1).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None,
                   help="Encoder width. 32 is the historical default and looks like the "
                        "binding constraint: that embedding has to serve three cap "
                        "heatmaps, a dome mask, 432 phi coefficients and a 7-d pose.")
    p.add_argument("--no-rotation-augment", dest="rotation_augment", action="store_false",
                   help="Disable the random SO(3) augmentation.")
    p.add_argument("--save-root", default=None)
    p.add_argument("--no-sanity-synthetic", dest="sanity_synthetic", action="store_false",
                   help="Skip the synthetic sanity panel (needs a stage-1 GHD VAE checkpoint).")
    p.add_argument("--ghd-vae-ckpt", default=None)
    p.set_defaults(sanity_synthetic=None, rotation_augment=True)
    return p.parse_args() if __name__ == "__main__" else p.parse_args([])


_args = _parse_args()

PROCESSED_ROOT = ROOT / "runtime_dataset" / "AneuG_morpho"
CANONICAL_ROOT = ROOT / "dataset" / "canonical"

DEVICE     = torch.device(_args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
# `if _args.epochs` would treat --epochs 0 as "unset" and silently train the
# full 2000 -- 0 is a legitimate request (render the sanity panels and stop).
EPOCHS     = 2000 if _args.epochs is None else _args.epochs
BATCH_SIZE = 64  # GPSConv's global attention is O(N^2) per graph, up to N=4143 (Bifurcated).
                 # An earlier note here claimed batch 32 OOM'd; that was a shared,
                 # contended card. Measured on an idle 24 GB card, peak allocation
                 # for an all-Bifurcated batch is 1.1 GB at 16 and 1.6 GB at 24, so
                 # memory is not the binding constraint. 24 over 32 keeps a
                 # reasonable number of optimizer steps per epoch: the corpus is
                 # only ~507 training cases, so batch 24 gives ~21 steps/epoch.
LR         = 5e-4
LR_MIN     = 1e-6   # linear-decay floor (1/500 of LR), reached at the final epoch

MAX_BRANCHES = 3
PHI_DIM      = 432   # 144 GHD coefficients x 3 axes, flattened -- the phi head's target
ROTATION_DIM   = 6     # 6D rotation: first two columns of the augmentation rotation
NUM_TYPES    = 3
# Openings per aneurysm type: bifurcated has three, sidewall two. The model
# always emits max_branches slots, and get_loss masks the unused one via
# branch_mask -- so for a sidewall shape slot 2 is never trained and its output
# is arbitrary. Real cases carry their own branch_mask; GENERATED ones do not,
# so the synthetic panel has to derive the count from the type or it will draw
# a third cap that means nothing.
TYPE_N_OPEN = {0: 3, 1: 2, 2: 2}

# Irreducible floor of patch_loss: it is -mean log p(v) over the patch, so a
# perfect prediction (mass spread uniformly over the patch) scores
# mean(log |patch|), measured at 4.846 over this corpus. Drawn on the curve so
# the term is read against what is achievable rather than against zero.
PATCH_FLOOR = 4.846
HIDDEN       = _args.hidden if _args.hidden else 32
STEM_LAYERS  = 2   # plain GCNConv layers before GPS attention (see models/morphoformer.py)
GPS_LAYERS   = 4   # GPSConv layers (local MPNN + global self-attention), on top of the stem
GPS_HEADS    = 4
USE_NORMALS  = True   # default on for this model — see MultiCanonicalGHDReconstruct.to_pyg_batch
ROTATION_AUGMENT = _args.rotation_augment   # random SO(3) of mesh + targets, in prepare_batch

TANGENT_WEIGHT = _args.tangent_weight
# Was 0.1, on the assumption that the patch term was a mere auxiliary nudging
# the soft-argmax toward the right place. That undersells it: the cap REGION is
# a first-class output -- it is what actually opens a mesh -- so it carries at
# least as much weight as the tangent. Its raw value also looks larger than it
# is: patch is a cross-entropy whose floor is the target's entropy
# (mean log|patch| = 4.85), not zero, so only the excess above that carries
# gradient.
PATCH_WEIGHT   = _args.patch_weight

# Which of the two models this run trains. Both share one encoder, so their
# features stay comparable; only the heads and the supervision differ.
_VARIANT = _args.variant
# One switch, two models. The dome head and the phi/rotation auxiliary branch
# stand or fall together: the branch exists only to shape the embedding the
# FPD/KPD metrics consume, and dome supervision is what makes that embedding a
# morphology sensor rather than a cap detector. The uncapper gets neither --
# extra heads would only add gradient noise to the task that has to work at
# inference.
_SENSOR = _args.variant == "morphology_sensor"
PREDICT_DOME = PREDICT_PHI = PREDICT_ROTATION = _SENSOR
DOME_WEIGHT  = _args.dome_weight
PHI_WEIGHT   = _args.phi_weight
ROTATION_WEIGHT = _args.rotation_weight
FOLDS, FOLD  = _args.folds, _args.fold

_tag = f"h{HIDDEN}_gps{GPS_LAYERS}_tw{TANGENT_WEIGHT:g}_pw{PATCH_WEIGHT:g}"
if PREDICT_DOME:
    _tag += f"_dome{DOME_WEIGHT:g}"
if PREDICT_PHI:
    _tag += f"_phi{PHI_WEIGHT:g}"
if PREDICT_ROTATION:
    _tag += f"_rot{ROTATION_WEIGHT:g}"
if FOLDS > 0:
    _tag += f"_fold{FOLD}of{FOLDS}"
SAVE_ROOT = Path(_args.save_root) if _args.save_root else ROOT / "runtime_train" / "morphoformer"
SAVE_DIR = SAVE_ROOT / _VARIANT / _tag

# Frozen stage-1 GHD VAE — only used to sample synthetic phi for the fully-
# generative sanity panel. Post type2->type1 merge (confirmed via its own
# saved hparams["merge_type2_into_type1"] == True).
GHD_VAE_CKPT = (Path(_args.ghd_vae_ckpt) if _args.ghd_vae_ckpt else
                ROOT / "runtime_train" / "ghd_vae" / "stage1" / "ghd_vae_h512_z16_kl0.5"
                     / "epoch_05000.pth")

SAVE_EVERY   = 200
LOG_EVERY    = 10
SANITY_EVERY = 200
SANITY_SYNTHETIC = True if _args.sanity_synthetic is None else _args.sanity_synthetic

WANDB_PROJECT = "AneuGv2_endcap"
WANDB_RUN_NAME = SAVE_DIR.name


def model_args():
    return {
        "max_branches": MAX_BRANCHES,
        "hidden": HIDDEN,
        "stem_layers": STEM_LAYERS,
        "gps_layers": GPS_LAYERS,
        "gps_heads": GPS_HEADS,
        "use_normals": USE_NORMALS,
        "predict_dome": PREDICT_DOME,
        "predict_phi": PREDICT_PHI,
        "predict_rotation": PREDICT_ROTATION,
        "phi_dim": PHI_DIM,
        "rotation_dim": ROTATION_DIM,
    }


def all_hparams():
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_min": LR_MIN,
        "max_branches": MAX_BRANCHES,
        "num_types": NUM_TYPES,
        "hidden": HIDDEN,
        "stem_layers": STEM_LAYERS,
        "gps_layers": GPS_LAYERS,
        "gps_heads": GPS_HEADS,
        "use_normals": USE_NORMALS,
        "rotation_augment": ROTATION_AUGMENT,
        "tangent_weight": TANGENT_WEIGHT,
        "patch_weight": PATCH_WEIGHT,
        "ghd_vae_ckpt": str(GHD_VAE_CKPT),
        "variant": _VARIANT,
        "predict_dome": PREDICT_DOME,
        "predict_phi": PREDICT_PHI,
        "dome_weight": DOME_WEIGHT,
        "phi_weight": PHI_WEIGHT,
        "rotation_weight": ROTATION_WEIGHT,
        "folds": FOLDS,
        "fold": FOLD,
        "save_every": SAVE_EVERY,
        "log_every": LOG_EVERY,
        "sanity_every": SANITY_EVERY,
        "sanity_synthetic": SANITY_SYNTHETIC,
    }


def save_checkpoint(model, optimizer, epoch, held_out=()):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": model_args(),
            "hparams": all_hparams(),
            # The cases this run never saw. The FPD/KPD reference set must be
            # drawn from exactly these, so it travels with the weights rather
            # than having to be re-derived (and re-derived identically) later.
            "held_out_cases": list(held_out),
        },
        SAVE_DIR / f"epoch_{epoch:05d}.pth",
    )


def prepare_batch(batch, multi_recon, augment):
    """Builds the PyG mesh batch here (not inside the model — see module
    docstring) and, when augmenting, rotates it AND the ground-truth
    endpoint/tangent targets by the exact same per-sample rotations, so the
    model (which stays augmentation-agnostic) is always training against a
    self-consistent (mesh, targets) pair, whichever frame that happens to be in."""
    phi           = batch["phi"].to(DEVICE)
    aneurysm_type = batch["aneurysm_type"].to(DEVICE)
    endpoints     = batch["endpoints"].to(DEVICE)
    tangents      = batch["tangents"].to(DEVICE)

    pyg_batch = multi_recon.to_pyg_batch(phi, aneurysm_type, point_std=None, include_normals=USE_NORMALS)

    if augment:
        rot = RandomSO3Rotation.sample(phi.size(0), phi.dtype, DEVICE)   # [B, 3, 3]
        pyg_batch = RandomSO3Rotation.apply_to_data(pyg_batch, rot)
        endpoints = RandomSO3Rotation.apply_to_points(rot, endpoints)
        tangents  = RandomSO3Rotation.apply_to_points(rot, tangents)

    out = {
        "pyg_batch":     pyg_batch,
        "endpoints":     endpoints,
        "tangents":      tangents,
        "branch_mask":   batch["branch_mask"].to(DEVICE),
        "in_patch":      batch["in_patch"].to(DEVICE),
        "node_batch":    batch["node_batch"].to(DEVICE),
    }
    # Sensor targets under augmentation.
    #   dome  per-VERTEX mask -- moving the vertices does not change which ones
    #         are dome, so it is never rotated.
    #   phi   the shape in the CANONICAL frame -- invariant, so the head must
    #         return the same answer whichever way the mesh is turned.
    #   rotation  the AUGMENTATION ROTATION Q and nothing else, as 6D.
    #
    #         The mesh handed to the network is rebuilt from phi alone and then
    #         turned by Q, so Q is the only pose information actually present in
    #         the input, and it is fully recoverable. This target used to be
    #         Q @ R_fitted, where R_fitted is the GHD Stage-2 pose that carries
    #         the warped canonical onto the real patient complex
    #         (preprocess_assembled.reconstruct). That transform is not part of
    #         the input at all, so the head was being asked for something no
    #         function of the mesh could produce, and the resulting noise went
    #         straight into the embedding that FPD/KPD read.
    #
    #         With augment off the input is the unrotated canonical-frame mesh,
    #         so the target is the identity.
    if "dome" in batch:
        out["dome"] = batch["dome"].to(DEVICE)
    out["phi"] = phi
    if PREDICT_ROTATION:
        R = rot if augment else torch.eye(3, device=DEVICE).expand(phi.size(0), 3, 3)
        # 6D representation: the first two columns of R, concatenated. The third
        # is recoverable by cross product, so nothing is lost, and unlike
        # axis-angle it is continuous everywhere.
        #
        # cat, NOT R[:, :, :2].reshape(B, 6) -- reshape walks row-major and would
        # interleave the two columns as [R00,R01,R10,...], which decodes back to
        # a different rotation. Verified to round-trip to 0 through the
        # Gram-Schmidt decoding used at eval.
        out["rotation"] = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1).contiguous()
    return out


def plot_loss_curves(history, save_dir):
    """Overwrites SAVE_DIR/loss_curves.png each call with the full history so
    far — lets you check progress on a run without opening wandb. Log-scale
    y-axis since endpoint/tangent/patch losses span orders of magnitude over
    training."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # dome/phi/rotation only exist for the morphology_sensor variant, so include
    # whichever the run actually recorded rather than a fixed list.
    keys = [k for k in ("loss", "endpoint", "tangent", "patch", "dome", "phi", "rotation")
            if history.get(k)]
    ncols = min(len(keys), 4)
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    epochs = history["epoch"]
    for ax, key in zip(axes, keys):
        ax.plot(epochs, history[key])
        ax.set_yscale("log")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
        if key == "patch":
            # Cross-entropy against a uniform-over-patch target: its floor is
            # that target's entropy, not zero. Without the line the curve looks
            # stuck at ~5.3 when it is actually within 0.4 of optimal.
            ax.axhline(PATCH_FLOOR, color="crimson", ls="--", lw=1.2)
            ax.text(0.02, 0.04, f"floor {PATCH_FLOOR:.2f}", color="crimson",
                    transform=ax.transAxes, fontsize=8)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(SAVE_DIR.name)
    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "loss_curves.png", dpi=130)
    plt.close(fig)


class _Subset:
    """Minimal view over chosen dataset rows.

    sanity_check only ever does len() and [i], so a full torch Subset would add
    nothing -- and this keeps `.samples` reachable, which sanity_check needs for
    the ground-truth patch and branch mask.
    """

    def __init__(self, base, idx):
        self.samples = [base.samples[i] for i in idx]
        self._base = base
        self._idx = list(idx)

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        return self._base[self._idx[i]]


def _region_from_probs(probs_b, frac=0.2, min_ratio=3.0, max_k=400):
    """Vertices the model actually calls this cap.

    loc_probs is a softmax over ALL vertices, so absolute values are tiny and a
    fixed threshold is meaningless -- hence `frac` of the branch's own peak.
    But that alone is badly behaved when the model is untrained: the
    distribution is then nearly uniform, peak ~= 1/N, and 0.2*peak selects
    almost EVERY vertex, blanketing the mesh in dots that look like a
    prediction but carry no information.

    Two guards. `min_ratio` demands a vertex be at least that many times the
    uniform probability, so an untrained head draws nothing rather than
    everything -- an honest picture of "knows nothing yet". `max_k` caps the
    count at roughly a real cap's size, keeping the marker set readable once
    the head sharpens.
    """
    if probs_b is None or probs_b.size == 0:
        return np.zeros(0, dtype=int)
    n = probs_b.size
    peak = float(probs_b.max())
    if peak <= 0:
        return np.zeros(0, dtype=int)
    thr = max(frac * peak, min_ratio / n)
    idx = np.flatnonzero(probs_b >= thr)
    if idx.size > max_k:
        idx = idx[np.argsort(probs_b[idx])[-max_k:]]
    return idx


# Region colours. Red / black / blue for the three branch slots -- maximum
# separation from each other and from the grey body -- and a dark grey dome.
CAP_COLORS = ["#111111", "#d62728", "#1f4fd8"]   # cap 0 black, 1 red, 2 blue
DOME_COLOR = "#4d4d4d"                           # dark grey


def _faces_in(faces, vmask, min_verts=2):
    """Faces of a region, drawn as SURFACE rather than scattered points.

    min_verts=2 (majority), NOT all three. Requiring every vertex means one
    vertex dipping below threshold removes a whole triangle, so a slightly
    noisy per-vertex mask renders as a patch riddled with holes -- an artefact
    of the drawing rule, not of the prediction. A majority rule closes those
    pinholes while still following the region's real boundary.
    """
    if vmask is None or not np.any(vmask):
        return None
    sel = vmask[faces].sum(axis=1) >= min_verts
    return faces[sel] if sel.any() else None


def _close_mask(faces, vmask, n_verts):
    """One dilate-then-erode pass over the mesh graph.

    Fills isolated interior dropouts -- a vertex the head happened to score
    just under threshold while all its neighbours are well over -- without
    growing the outer boundary, since erode undoes the dilation everywhere the
    region was already solid.
    """
    if vmask is None or not np.any(vmask):
        return vmask
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    nbr_in = np.zeros(n_verts, dtype=np.int32)
    np.add.at(nbr_in, e[:, 0], vmask[e[:, 1]].astype(np.int32))
    deg = np.zeros(n_verts, dtype=np.int32)
    np.add.at(deg, e[:, 0], 1)
    dilated = vmask | (nbr_in > 0)
    nbr_out = np.zeros(n_verts, dtype=np.int32)
    np.add.at(nbr_out, e[:, 0], (~dilated[e[:, 1]]).astype(np.int32))
    return dilated & (nbr_out == 0) | vmask


def _draw_region(ax, verts, faces, vmask, color, alpha=1.0):
    vmask = _close_mask(faces, np.asarray(vmask, dtype=bool), verts.shape[0])
    f = _faces_in(faces, vmask)
    if f is not None:
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=f,
                        color=color, edgecolor="none", alpha=alpha)
    return f is not None


def _recon_numpy(model, phi_1, atype):
    """Vertices/faces for one shape, via the model's OWN reconstructor.

    NOT preprocess_ImperialNHS._reconstruct_ghd_numpy, which normalises by
    ||V_can||max * 1.10 * 2.50 -- the OLD project's convention. AneuG's
    norm_canonical is plain ||V_can||max (5.375418 bifurcated / 6.037298
    sidewall), so that function divides the base shape by 2.75x too much while
    adding an undivided eigvec @ phi on top: the deformation then dominates and
    every mesh comes out visibly warped, for both aneurysm types.
    """
    recon = model.multi_recon.get(int(atype))
    mesh = recon.ghd_forward_as_Meshes(phi_1, denormalize_shape=True)
    return (mesh.verts_padded()[0].detach().cpu().numpy(),
            mesh.faces_padded()[0].detach().cpu().numpy())


@torch.no_grad()
def sanity_check(model, dataset, epoch, save_dir, n=8, ncols=4, prefix=""):
    """Ground truth (black) vs. prediction (dashed color) on a few real
    (TRAINING — no held-out split, see main()) cases — the gap between a
    black and dashed marker visualizes endpoint_loss directly; yellow arrows
    are the tangents (real vs. predicted). Real-data fit is a sanity check on
    whether the model is learning at all; sanity_synthetic (generated shapes)
    is the one that actually matters for this model's purpose."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _set_axes_equal

    model.eval()
    n = min(n, len(dataset))
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    bright = CAP_COLORS

    for i in range(n):
        item = dataset[i]
        phi = item["phi"].unsqueeze(0).to(DEVICE)
        atype_t = item["aneurysm_type"].unsqueeze(0).to(DEVICE)
        endpoint_pred, tangent_pred, loc_probs, _tok, extra = model(phi, atype_t)
        endpoint_pred = endpoint_pred[0].cpu().numpy()
        tangent_pred = tangent_pred[0].cpu().numpy()
        loc_np = loc_probs[0].detach().cpu().numpy()
        dome_np = (torch.sigmoid(extra["dome_logits"][0]).detach().cpu().numpy()
                   if "dome_logits" in extra else None)

        atype = int(item["aneurysm_type"])
        verts, faces = _recon_numpy(model, phi, atype)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        # Ghost the surface so the region dots below carry the image: solid
        # markers on a near-transparent mesh read far better than the reverse.
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="#b8b8bd", edgecolor="none", alpha=0.55)
        all_pts = [verts]
        if dome_np is not None and dome_np.shape[0] >= verts.shape[0]:
            _draw_region(ax, verts, faces, dome_np[:verts.shape[0]] > 0.5, DOME_COLOR)
        for b in range(model.max_branches):
            if not item["branch_mask"][b]:
                continue
            pr = _region_from_probs(loc_np[b, :verts.shape[0]])
            if pr.size:
                m = np.zeros(verts.shape[0], dtype=bool); m[pr] = True
                _draw_region(ax, verts, faces, m, CAP_COLORS[b % len(CAP_COLORS)])
            c = bright[b % len(bright)]
            ep_real = item["endpoints"][b].numpy()
            tg_real = item["tangents"][b].numpy()
            ep_pred = endpoint_pred[b]
            tg_pred = tangent_pred[b]

            # No endpoint markers: the two arrows already start at the two
            # endpoints, and the dotted connector below shows the gap between
            # them, so extra dots only add clutter.
            ax.plot(*zip(ep_real, ep_pred), color=c, lw=1.0, ls=":")
            # Same treatment as the synthetic panel: length scaled to the mesh,
            # and the predicted arrow in the branch colour rather than yellow,
            # which is illegible on the ghosted surface. Ground truth stays black.
            L = 0.22 * float((verts.max(0) - verts.min(0)).max())
            ax.quiver(*ep_real, *tg_real, length=L, color="black",
                      linewidth=2.0, arrow_length_ratio=0.28)
            ax.quiver(*ep_pred, *tg_pred, length=L, color=c,
                      linewidth=2.2, arrow_length_ratio=0.28)
            all_pts += [ep_real[None], ep_pred[None]]

        ax.set_axis_off()
        ax.set_title(f"{item['case']} (type {atype})", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  {'HELD-OUT TEST' if prefix else 'training'} cases"
                 f"  (black=real, colored=predicted)", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{prefix}epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


@torch.no_grad()
def sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim,
                     epoch, save_dir, n=8, ncols=4, seed=0):
    """Fully-generative sanity: synthetic GHD phi -> predicted endpoints/
    tangents. No ground truth exists for these, so this just checks the
    prediction looks like a sane vessel-cap location — the real point of this
    whole redesign was making that hold for GENERATED, not just real, shapes."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess_ImperialNHS import _set_axes_equal

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Type 2 is merged into type 1 everywhere in this pipeline (see
    # MorphoDataset.__getitem__) — never sample it here either, regardless of
    # whether the loaded GHD_VAE_CKPT happens to have its own type-2 row
    # trained (an old, pre-merge checkpoint would).
    types = torch.randint(0, NUM_TYPES - 1, (n,), generator=g).to(DEVICE)
    z_ghd = torch.randn(n, ghd_vae.latent_dim, generator=g).to(DEVICE)
    ghd_n, _ = ghd_vae.decode(z_ghd, types)
    phi = (ghd_n * ghd_std[:, :ghd_input_dim] + ghd_mean[:, :ghd_input_dim]).reshape(n, -1, 3)

    endpoint_pred, tangent_pred, loc_probs, _tok, extra = model(phi, types)
    endpoint_pred, tangent_pred = endpoint_pred.cpu().numpy(), tangent_pred.cpu().numpy()
    loc_all = loc_probs.detach().cpu().numpy()
    dome_all = (torch.sigmoid(extra["dome_logits"]).detach().cpu().numpy()
                if "dome_logits" in extra else None)

    bright = CAP_COLORS
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i in range(n):
        atype = int(types[i])
        verts, faces = _recon_numpy(model, phi[i:i + 1], atype)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        # Ghost the surface so the region dots below carry the image: solid
        # markers on a near-transparent mesh read far better than the reverse.
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                        color="#b8b8bd", edgecolor="none", alpha=0.55)
        all_pts = [verts]
        if dome_all is not None and dome_all.shape[1] >= verts.shape[0]:
            _draw_region(ax, verts, faces, dome_all[i, :verts.shape[0]] > 0.5, DOME_COLOR)
        n_open = TYPE_N_OPEN.get(atype, model.max_branches)
        for b in range(min(n_open, model.max_branches)):
            c = CAP_COLORS[b % len(CAP_COLORS)]
            # no ground truth for generated shapes -- predicted region only
            pr = _region_from_probs(loc_all[i, b, :verts.shape[0]])
            if pr.size:
                m = np.zeros(verts.shape[0], dtype=bool); m[pr] = True
                _draw_region(ax, verts, faces, m, c)
            ep, tg = endpoint_pred[i, b], tangent_pred[i, b]
            # Arrow length relative to the mesh, not a fixed 2.0: these shapes
            # span ~15-25 units, so a constant stub is nearly invisible. Colour
            # is the branch's own -- yellow was illegible against the ghosted
            # near-white surface.
            L = 0.22 * float((verts.max(0) - verts.min(0)).max())
            ax.quiver(*ep, *tg, length=L, color=c, linewidth=2.2,
                      arrow_length_ratio=0.28)
            all_pts.append(ep[None])
        ax.set_axis_off()
        ax.set_title(f"type {atype}  ({n_open} caps)", fontsize=8)
        _set_axes_equal(ax, *all_pts)

    fig.suptitle(f"epoch {epoch}  synthetic generation, predicted endcaps", fontsize=10)
    fig.tight_layout()
    save_dir = Path(save_dir) / "sanity"; save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"sanity_synth_epoch_{epoch:05d}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    model.train()
    return out


def main():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config=all_hparams())

    # No held-out split — this model's actual target is unseen SYNTHETIC
    # shapes (sanity_synthetic), not held-out real cases, so a real-data val
    # split wouldn't measure the thing that matters; use all real data to train.
    dataset = MorphoDataset(PROCESSED_ROOT, max_branches=MAX_BRANCHES,
                            with_dome=PREDICT_DOME,
                            assembled_root=ROOT / "runtime_dataset" / "assembled",
                            # The rotation target is built from the
                            # augmentation rotation in prepare_batch, so the
                            # stored Stage-2 affine is not needed. Loading it
                            # would only drop every case that lacks one,
                            # synthetic cases included.
                            with_affine=False)
    test_set = None
    if _args.test_size > 0:
        # Hash of the case name, not a shuffle: every run in the sweep must hold
        # out the SAME cases or their test panels are not comparable, and that
        # has to survive re-runs and directory-order changes.
        import hashlib
        order = sorted(range(len(dataset.samples)),
                       key=lambda i: hashlib.md5(
                           dataset.samples[i]["case"].encode()).hexdigest())
        keep_idx, test_idx = order[_args.test_size:], order[:_args.test_size]
        test_set = _Subset(dataset, sorted(test_idx))
        names = [dataset.samples[i]["case"] for i in sorted(test_idx)]
        dataset.samples = [dataset.samples[i] for i in sorted(keep_idx)]
        print(f"held-out test set: {len(test_idx)} case(s) -> {', '.join(names)}")
        print(f"training on {len(dataset.samples)}")

    held_out = []
    if FOLDS > 0:
        # Deterministic split on the case name, so every fold of every run sees
        # the same partition regardless of directory listing order -- the
        # descriptor's metric depends on knowing exactly which meshes it never
        # saw, and that has to survive a re-run.
        import hashlib
        keep, held = [], []
        for i, rec in enumerate(dataset.samples):
            h = int(hashlib.md5(rec["case"].encode()).hexdigest()[:8], 16)
            (held if h % FOLDS == FOLD else keep).append(i)
        held_out = [dataset.samples[i]["case"] for i in held]
        dataset.samples = [dataset.samples[i] for i in keep]
        print(f"fold {FOLD}/{FOLDS}: training on {len(keep)}, holding out {len(held)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, drop_last=True, collate_fn=collate_morpho)

    multi_recon = MultiCanonicalGHDReconstruct(CANONICAL_ROOT, device=DEVICE)
    model = MorphoFormer(multi_recon, **model_args()).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Linear decay rather than cosine: a plain ramp spends less of the run at a
    # very low rate than cosine's long tail, which matters at LR=5e-5 where
    # there is little headroom to begin with.
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=LR_MIN / LR, total_iters=max(EPOCHS, 1))

    ghd_vae = ghd_mean = ghd_std = ghd_input_dim = None
    if SANITY_SYNTHETIC:
        ghd_vae, ghd_mean, ghd_std, ghd_input_dim = load_ghd_vae(GHD_VAE_CKPT, DEVICE)
        for p in ghd_vae.parameters():
            p.requires_grad_(False)
        print(f"Loaded GHD VAE: {GHD_VAE_CKPT.name}")

    print(f"{_VARIANT}: {len(dataset)} samples (no held-out split)")

    history = {"epoch": [], "loss": [], "endpoint": [], "tangent": [], "patch": [],
               "dome": [], "phi": [], "rotation": []}

    for epoch in range(EPOCHS + 1):
        model.train()
        metrics = {}
        for batch in loader:
            data = prepare_batch(batch, multi_recon, augment=ROTATION_AUGMENT)
            endpoint_pred, tangent_pred, loc_weights, token_mask, extra = model(pyg_batch=data["pyg_batch"])
            endpoint_loss, tangent_loss, patch_loss = model.get_loss(
                endpoint_pred, tangent_pred, loc_weights, token_mask,
                data["endpoints"], data["tangents"], data["branch_mask"],
                data["in_patch"], data["node_batch"])
            loss = endpoint_loss + TANGENT_WEIGHT * tangent_loss + PATCH_WEIGHT * patch_loss

            dome_loss = phi_loss = rotation_loss = None
            if PREDICT_DOME and "dome_logits" in extra:
                # Per-vertex BCE over REAL vertices only: token_mask excludes the
                # dense-batch padding, which would otherwise dominate the mean
                # (the two canonicals differ in vertex count, so every sidewall
                # mesh in a mixed batch is padded).
                tgt = to_dense_batch(data["dome"].float(), data["node_batch"])[0]
                m = token_mask
                dome_loss = F.binary_cross_entropy_with_logits(
                    extra["dome_logits"][m], tgt[m])
                loss = loss + DOME_WEIGHT * dome_loss
            if PREDICT_PHI and "phi_pred" in extra:
                phi_loss = F.mse_loss(extra["phi_pred"], data["phi"].flatten(1))
                loss = loss + PHI_WEIGHT * phi_loss
            if PREDICT_ROTATION and "rotation_pred" in extra:
                rotation_loss = F.mse_loss(extra["rotation_pred"], data["rotation"])
                loss = loss + ROTATION_WEIGHT * rotation_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            metrics = {
                "loss": float(loss), "endpoint": float(endpoint_loss),
                "tangent": float(tangent_loss), "patch": float(patch_loss),
            }
            if dome_loss is not None:
                metrics["dome"] = float(dome_loss)
            if phi_loss is not None:
                metrics["phi"] = float(phi_loss)
            if rotation_loss is not None:
                metrics["rotation"] = float(rotation_loss)

        scheduler.step()
        if epoch % LOG_EVERY == 0:
            print({"epoch": epoch, "lr": round(scheduler.get_last_lr()[0], 6), **metrics})
            wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], **metrics}, step=epoch)
            history["epoch"].append(epoch)
            # dome/phi only exist for the descriptor variant
            for key in (k for k in ("loss", "endpoint", "tangent", "patch", "dome", "phi", "rotation")
                        if k in metrics):
                history[key].append(metrics[key])
            plot_loss_curves(history, SAVE_DIR)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, held_out)
        if epoch % SANITY_EVERY == 0:
            sanity_check(model, dataset, epoch, SAVE_DIR)
            if test_set is not None and len(test_set):
                # Same renderer, different cases and a different filename: the
                # train panel shows fit, this one shows whether any of it
                # transfers to meshes the model never saw.
                sanity_check(model, test_set, epoch, SAVE_DIR, prefix="test_")
            if SANITY_SYNTHETIC and ghd_vae is not None:
                sanity_synthetic(model, ghd_vae, ghd_mean, ghd_std, ghd_input_dim, epoch, SAVE_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()


"""
conda activate new
python scripts/train/train_morphoformer.py
"""
