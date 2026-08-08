"""
Batch driver to reproduce Wenhao's visualize_ghd_generation.ipynb logic across
all (model_kind, mode, aneurysm_type) combinations for the baseline checkpoints.

Does NOT modify any of Wenhao's source files. Imports his modules and follows
the same pattern the notebook uses, just inside a loop, with figures saved as
PNGs instead of displayed inline.

Run from anywhere; ROOT is resolved from this file's location.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.ghd_datasets import ProcessedGHDDataset
from dataset.preprocess_ImperialNHS import get_ghd_reconstruct
from models.ghd_vae import TypeConditionalVAE
from models.ghd_vqvae import ConditionalGHDVQVAE

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PROCESSED_ROOT = ROOT / "dataset" / "processed"
OUT_DIR = ROOT / "checkpoints" / "v2" / "renders_baseline"

CHECKPOINTS = [
    ("vae_baseline",   "vae",   ROOT / "checkpoints" / "v2" / "ghd_vae"   / "epoch_10000.pth"),
    ("vqvae_baseline", "vqvae", ROOT / "checkpoints" / "v2" / "ghd_vqvae" / "epoch_10000.pth"),
]
MODES = ["reconstruct", "random"]
TYPES = [0, 1, 2]
NUM_SAMPLES = 4


def build_model(model_kind, checkpoint):
    state = checkpoint["model"]
    withscale = bool(checkpoint.get("withscale", True))
    if model_kind == "vae":
        input_total = state["fc1.weight"].shape[1]
        input_dim = input_total - 1 if withscale else input_total
        hidden_dim = state["fc1.weight"].shape[0]
        latent_dim = state["fc21.weight"].shape[0]
        num_types, condition_dim = state["type_embedding.weight"].shape
        model = TypeConditionalVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            withscale=withscale,
            num_types=num_types,
            condition_dim=condition_dim,
        )
    elif model_kind == "vqvae":
        model = ConditionalGHDVQVAE(
            seq_len=int(checkpoint.get("seq_len", 144)),
            channels=int(checkpoint.get("channels", 3)),
            num_types=int(checkpoint.get("num_types", 3)),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            embedding_dim=int(checkpoint.get("embedding_dim", 64)),
            num_codes=int(checkpoint.get("num_codes", 128)),
        )
    else:
        raise ValueError(model_kind)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, withscale


@torch.no_grad()
def generate_vae(model, withscale, aneurysm_type, n):
    type_tensor = torch.full((n,), int(aneurysm_type), dtype=torch.long, device=DEVICE)
    z = torch.randn(n, model.latent_dim, device=DEVICE)
    if withscale:
        ghd, scale = model.decode(z, type_tensor)
    else:
        ghd = model.decode(z, type_tensor)
        scale = torch.zeros(n, 1, device=DEVICE)
    return ghd, scale, type_tensor


@torch.no_grad()
def generate_vqvae(model, aneurysm_type, n):
    type_tensor = torch.full((n,), int(aneurysm_type), dtype=torch.long, device=DEVICE)
    latent_len = model.seq_len // 4
    indices = torch.randint(model.quantizer.num_codes, (n, latent_len), device=DEVICE)
    z_q = model.quantizer.embedding(indices).view(n, latent_len, model.embedding_dim)
    x, scale = model.decode(z_q, type_tensor)
    return x.reshape(n, -1), scale, type_tensor


@torch.no_grad()
def reconstruct_from_dataset(model_kind, model, withscale, dataset, aneurysm_type, n):
    idxs = [i for i, t in enumerate(dataset.aneurysm_type) if int(t) == int(aneurysm_type)][:n]
    if not idxs:
        return None, None, None
    samples = [dataset[i] for i in idxs]
    batch = torch.stack([x for x, _ in samples]).to(DEVICE)
    type_tensor = torch.stack([t for _, t in samples]).to(DEVICE)
    if withscale:
        ghd, scale = batch[:, :-1], batch[:, -1:]
    else:
        ghd = batch
        scale = batch.new_zeros((batch.shape[0], 1))
    if model_kind == "vae":
        if withscale:
            ghd, scale, _, _ = model(ghd, type_tensor, scale)
        else:
            ghd, _, _ = model(ghd, type_tensor, scale)
    else:
        out = model(ghd.view(ghd.shape[0], 144, 3), scale, type_tensor)
        ghd = out["x_recon"].reshape(ghd.shape[0], -1)
        scale = out["scale_recon"]
    return ghd, scale, type_tensor


def denormalize_scale(scale, mean, std, withscale):
    if not withscale:
        return None
    return scale * std[:, -1:] + mean[:, -1:]


def mesh_from_ghd(ghd_reconstruct, ghd, scale, aneurysm_type, mean, std, withscale):
    reconstructor = ghd_reconstruct.get(int(aneurysm_type))
    scale_dn = denormalize_scale(scale, mean, std, withscale)
    return reconstructor.ghd_forward_as_Meshes(ghd, mean=mean[:, :432], std=std[:, :432], scale=scale_dn)


def plot_mesh(mesh, title, save_path):
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    faces = mesh.faces_list()[0].detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=faces, color="lightsteelblue", edgecolor="none", alpha=0.85,
    )
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    radius = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ghd_reconstruct = get_ghd_reconstruct(device=DEVICE)
    dataset = ProcessedGHDDataset(PROCESSED_ROOT, withscale=True, normalize=True)

    for label, model_kind, ckpt_path in CHECKPOINTS:
        print(f"\n=== {label} ({model_kind}) from {ckpt_path.name} ===", flush=True)
        if not ckpt_path.exists():
            print(f"  skip: checkpoint missing at {ckpt_path}", flush=True)
            continue
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model, withscale = build_model(model_kind, checkpoint)
        mean = checkpoint["mean"].to(DEVICE)
        std = checkpoint["std"].to(DEVICE)
        out_subdir = OUT_DIR / label
        out_subdir.mkdir(parents=True, exist_ok=True)

        for mode in MODES:
            for aneurysm_type in TYPES:
                if mode == "reconstruct":
                    ghd, scale, type_tensor = reconstruct_from_dataset(
                        model_kind, model, withscale, dataset, aneurysm_type, NUM_SAMPLES
                    )
                elif model_kind == "vae":
                    ghd, scale, type_tensor = generate_vae(model, withscale, aneurysm_type, NUM_SAMPLES)
                else:
                    ghd, scale, type_tensor = generate_vqvae(model, aneurysm_type, NUM_SAMPLES)

                if ghd is None:
                    print(f"  skip {mode} type={aneurysm_type}: no samples", flush=True)
                    continue

                for i in range(ghd.shape[0]):
                    mesh = mesh_from_ghd(
                        ghd_reconstruct, ghd[i:i+1], scale[i:i+1],
                        int(type_tensor[i]), mean, std, withscale,
                    )
                    fname = out_subdir / f"{mode}_type{aneurysm_type}_sample{i}.png"
                    plot_mesh(mesh, f"{label} | {mode} | type={aneurysm_type} | sample={i}", fname)
                    print(f"  saved {fname.name}", flush=True)

    print(f"\nDone. All renders in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
