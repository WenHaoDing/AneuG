"""
Sanity-check visualization for MultiBranchVAE sampling.

Samples one case from val_dataset, runs model.sample, then overlays the
generated branch curves on the GHD-reconstructed mesh and saves a figure.
"""

import numpy as np
import torch

from pathlib import Path


@torch.no_grad()
def sanity_check(model, val_dataset, epoch, save_dir, device, cond_cls, max_branches, max_local_points):
    """Sample one random val case and visualise generated branches vs GHD mesh.

    Args:
        model:             MultiBranchVAE (switched to eval inside, restored after)
        val_dataset:       torch Subset wrapping VesselSkeletonDataset
        epoch:             current training epoch (used in filename / title)
        save_dir:          root checkpoint dir; figures go to save_dir/sanity/
        device:            torch.device
        cond_cls:          MultiBranchConditions namedtuple class
        max_branches:      int
        max_local_points:  int  (= MAX_POINTS_PER_BRANCH - 1)
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from dataset.preprocess import _reconstruct_ghd_numpy, _set_axes_equal

    base_dataset = val_dataset.dataset   # unwrap Subset → VesselSkeletonDataset

    idx  = torch.randint(len(val_dataset), (1,)).item()
    item = val_dataset[idx]

    # build batch-of-1 condition
    phi  = item["phi"].unsqueeze(0).to(device)                             # [1, 144, 3]
    cond = cond_cls(
        aneurysm_type    = item["aneurysm_type"].unsqueeze(0).to(device),  # [1]
        scale            = item["scale"].unsqueeze(0).to(device),           # [1]
        start_points     = item["start_points"].unsqueeze(0).to(device),   # [1, max_branches, 3]
        branch_direction = item["branch_direction"].unsqueeze(0).to(device),# [1, max_branches, 3]
        branch_mask      = item["branch_mask"].unsqueeze(0).to(device),    # [1, max_branches]
    )

    model.eval()
    outputs, lengths, _ = model.sample(phi, cond)  # [1, seq_len, 3], [1, max_branches]
    model.train()

    # denormalize and reshape to [max_branches, max_local_points, 3]
    outputs = outputs[0].cpu()                                              # [seq_len, 3]
    outputs = base_dataset.denormalize_local_points(outputs)               # [seq_len, 3]
    outputs = outputs.view(max_branches, max_local_points, 3)              # [max_branches, max_local_points, 3]
    lengths = lengths[0].cpu()                                             # [max_branches]
    start_points = item["start_points"]                                    # [max_branches, 3]

    # reconstruct GHD mesh (numpy-only, no pytorch3d)
    checkpoint = {
        "aneurysm_type": int(item["aneurysm_type"]),
        "ghd": {"phi": item["phi"].numpy()},
    }
    verts, faces = _reconstruct_ghd_numpy(checkpoint, denormalize_shape=True)

    # build absolute branch curves from sampled output
    branch_points = []
    for b in range(max_branches):
        if not item["branch_mask"][b]:
            continue
        n     = int(lengths[b].clamp(min=1))
        start = start_points[b].numpy()                                    # [3]
        offs  = outputs[b, :n].numpy()                                     # [n, 3]
        pts   = np.concatenate([start[None], start[None] + offs], axis=0) # [n+1, 3]
        branch_points.append(pts)

    # plot
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                    color="lightgray", edgecolor="none", alpha=0.25)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branch_points), 1)))
    for i, pts in enumerate(branch_points):
        c = colors[i % len(colors)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, color=c, label=f"branch {i}")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, color=c)
        ax.scatter(*pts[0], s=40, marker="x", color=c)

    ax.set_title(f"epoch {epoch} | {item['case']} (type {int(item['aneurysm_type'])}: {item['canonical_type']})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_axes_equal(ax, verts, *branch_points)
    ax.legend(loc="upper right")
    fig.tight_layout()

    save_path = Path(save_dir) / "sanity" / f"epoch_{epoch:05d}_{item['case']}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved sanity figure → {save_path}")
