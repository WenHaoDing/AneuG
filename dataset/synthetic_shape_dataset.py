"""Pre-computed pool of synthetic shapes with their sensor-derived openings.

WHY THIS EXISTS. Stage-2 training needs, for every synthetic shape it sees, the
cap regions and the tangent at each opening. Those come from the morphology
sensor, and calling it inside the training loop means keeping a second network
and a second canonical reconstructor resident on the GPU, running an encoder
forward per step, and then doing the connected-component and boundary-walk work
on CPU while the optimiser waits. None of it depends on the stage-2 weights --
the sensor is frozen and the shapes come from a frozen stage-1 generator -- so
all of it can be done once, ahead of time, and read back as plain arrays.

TWO HALVES, AND ONLY ONE OF THEM TOUCHES A NETWORK.

  generation (once, offline)   `python dataset/synthetic_shape_dataset.py`
      loads the stage-1 GHD VAE and the sensor, draws shapes, uncaps them,
      writes pool.npz.

  use (every run)              `SyntheticShapeDataset(path)`
      reads pool.npz. No VAE, no sensor, no canonical reconstructor, no GPU.

The dataset class never generates. If the pool is missing it raises and tells
you the command to run, rather than quietly spending twenty minutes of GPU time
inside what a caller expected to be a file read.

WHAT COUNTS AS A GOOD SHAPE. A record is written only when every opening the
shape's type is supposed to have produced a usable cut: as many caps and tangent
vectors as the canonical has openings (3 for bifurcated, 2 for sidewall), each
cap's boundary closing into one simple cycle. A cap whose boundary splits in two
cannot be lofted onto, and a tube swept from it collapses, so a shape carrying
one is dropped rather than recorded with a hole in its branch mask. The
acceptance rate is reported and stored in the manifest.

    python dataset/synthetic_shape_dataset.py --n 1000 --device cuda:0

    from dataset.synthetic_shape_dataset import SyntheticShapeDataset
    pool = SyntheticShapeDataset("runtime_dataset/synthetic_shapes/default")
    b = pool.sample_batch(32, device="cuda:0")
    b["start_points"], b["branch_direction"], b["branch_mask"]   # sensor output
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_ROOT = ROOT / "dataset" / "canonical"
DEFAULT_POOL = ROOT / "runtime_dataset" / "synthetic_shapes" / "default"
DEFAULT_GHD_VAE = (ROOT / "runtime_train" / "ghd_vae" / "stage1"
                   / "ghd_vae_gan_h256_z16_kl2_adv0.3" / "epoch_05000.pth")
DEFAULT_SENSOR = (ROOT / "runtime_train" / "morphoformer" / "morphology_sensor"
                  / "h128_gps4_tw1_pw2_dome1_phi0.5_rot0.5_prob_z8_kl0.1"
                  / "epoch_02000.pth")
MAX_BRANCHES = 3
NUM_TYPES = 2          # type 2 shares the sidewall canonical with type 1


# ─────────────────────────────────────────────────────────────────────────────
# the dataset: loading only
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticShapeDataset(Dataset):
    """Synthetic shapes with their openings already measured.

    Each item carries what stage 2 needs to condition on an opening:

        phi               [144, 3]   GHD coefficients from the stage-1 generator
        aneurysm_type     scalar     0 bifurcated, 1 sidewall
        scale             scalar
        start_points      [MB, 3]    sensor endpoint, pushed outside the cut plane
        branch_direction  [MB, 3]    sensor tangent, unit length
        branch_mask       [MB]       True for openings this type has

    NOTE ON `start_points`. The sensor's raw endpoint lands INSIDE the cut plane
    on roughly a fifth of branches, and a tube lofted from there begins under the
    dome and self-intersects, so the stored point is the one nudged out past the
    rim by `SensorUncapper.push_outside`. The old in-training path called
    `branch_conditions`, which does not do that nudge, so conditioning on this
    pool is not bit-identical to conditioning on a live sensor call: the tangents
    match to float noise but a minority of start points sit slightly further out.
    The raw endpoints are kept as `start_points_raw` for anyone who wants the old
    behaviour exactly.

    The cap regions and ordered rim loops are kept too, so a merged mesh can be
    built later without the sensor: see `cap_mask`, `rim_loop`, `trimmed_faces`.
    Those are stored packed and are unpacked on demand rather than per item,
    because training reads only the six fields above.
    """

    def __init__(self, root=DEFAULT_POOL, max_branches=MAX_BRANCHES):
        self.root = Path(root)
        path = self.root / "pool.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"no synthetic pool at {path}. This class loads a pre-computed "
                f"pool, it does not build one. Generate it once with:\n"
                f"    python dataset/synthetic_shape_dataset.py --out {self.root} "
                f"--n 1000 --device cuda:0"
            )
        d = np.load(path, allow_pickle=False)
        self._d = {k: d[k] for k in d.files}
        self.manifest = json.loads((self.root / "manifest.json").read_text()) \
            if (self.root / "manifest.json").exists() else {}

        self.cases = [str(c) for c in self._d["case"]]
        self.aneurysm_type = torch.as_tensor(self._d["aneurysm_type"], dtype=torch.long)
        self.scale = torch.as_tensor(self._d["scale"], dtype=torch.float32)
        self.phi = torch.as_tensor(self._d["phi"], dtype=torch.float32)
        self.start_points = torch.as_tensor(self._d["start_points"], dtype=torch.float32)
        self.branch_direction = torch.as_tensor(self._d["directions"], dtype=torch.float32)
        self.branch_mask = torch.as_tensor(self._d["branch_mask"], dtype=torch.bool)
        self.start_points_raw = torch.as_tensor(self._d["start_points_raw"],
                                                dtype=torch.float32)

        if max_branches != self.branch_mask.shape[1]:
            raise ValueError(
                f"pool was built with max_branches={self.branch_mask.shape[1]}, "
                f"asked for {max_branches}")
        self.max_branches = max_branches

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return {
            "case":             self.cases[idx],
            "phi":              self.phi[idx],
            "aneurysm_type":    self.aneurysm_type[idx],
            "scale":            self.scale[idx],
            "start_points":     self.start_points[idx],
            "branch_direction": self.branch_direction[idx],
            "branch_mask":      self.branch_mask[idx],
        }

    # ---- batch access -------------------------------------------------------

    def sample_batch(self, n, generator=None, device=None, balanced=True):
        """Random shapes, stacked. Drop-in for a `branch_conditions` call.

        `balanced` draws equally from each type, matching what the stage-2
        direction loss did when it generated its own shapes: an unbalanced draw
        would weight the loss toward whichever canonical happened to come up.
        """
        if balanced:
            idx = []
            per = n // NUM_TYPES
            for t in range(NUM_TYPES):
                pool = (self.aneurysm_type == t).nonzero(as_tuple=True)[0]
                if len(pool) == 0:
                    continue
                take = per + (1 if t < n - per * NUM_TYPES else 0)
                idx.append(pool[torch.randint(len(pool), (take,), generator=generator)])
            idx = torch.cat(idx)[:n]
        else:
            idx = torch.randint(len(self), (n,), generator=generator)
        return self.batch(idx, device=device)

    def batch(self, idx, device=None):
        idx = torch.as_tensor(idx, dtype=torch.long)
        out = {
            "case":             [self.cases[int(i)] for i in idx],
            "index":            idx,
            "phi":              self.phi[idx],
            "aneurysm_type":    self.aneurysm_type[idx],
            "scale":            self.scale[idx],
            "start_points":     self.start_points[idx],
            "branch_direction": self.branch_direction[idx],
            "branch_mask":      self.branch_mask[idx],
        }
        if device is not None:
            out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out

    # ---- geometry kept for mesh fusion --------------------------------------

    def n_verts(self, idx):
        return int(self._d["n_verts"][int(idx)])

    def cap_mask(self, idx, branch):
        """Boolean per-vertex mask of one predicted cap."""
        nv = self.n_verts(idx)
        bits = self._d["cap_bits"][int(idx), int(branch)]
        return np.unpackbits(bits)[:nv].astype(bool)

    def dome_mask(self, idx):
        if "dome_bits" not in self._d:
            return None
        nv = self.n_verts(idx)
        return np.unpackbits(self._d["dome_bits"][int(idx)])[:nv].astype(bool)

    def rim_loop(self, idx, branch):
        """Rim as an ORDERED cycle, the form the tube sweeper needs."""
        off, ln = self._d["rim_offsets"][int(idx), int(branch)]
        return self._d["rim_flat"][off:off + ln].astype(np.int64)

    def trimmed_faces(self, idx, multi_recon):
        """Canonical faces with the capped triangles deleted.

        Rebuilt from the stored cap masks rather than stored directly: the face
        array is the same for every shape of a type, so keeping a copy per shape
        would be the largest thing in the file by far and would carry no
        information the masks do not already have.
        """
        faces = (multi_recon.get(int(self.aneurysm_type[idx]))
                 .canonical_Meshes.faces_packed().cpu().numpy())
        cut = np.zeros(self.n_verts(idx), dtype=bool)
        for b in range(self.max_branches):
            if bool(self.branch_mask[idx, b]):
                cut |= self.cap_mask(idx, b)
        return faces[~cut[faces].all(1)]

    def fused_mesh(self, idx, multi_recon, branch_points, branch_mask=None, **kwargs):
        """Stage-1 + stage-2 mesh, built from the stored openings.

        Mirrors `SensorUncapper.fused_mesh` and takes the same arguments, but
        reads the cut from this pool instead of running the sensor, so a
        training run can render a merged mesh without either network resident.

        Returns None when this shape has no usable opening, which cannot happen
        for a pool record -- every one passed the cap check at generation -- but
        is kept so the two call signatures behave identically.
        """
        idx = int(idx)
        atype = int(self.aneurysm_type[idx])
        loops = [self.rim_loop(idx, b) for b in range(self.max_branches)
                 if bool(self.branch_mask[idx, b])]
        if not loops:
            return None
        normals = [self.branch_direction[idx, b].numpy()
                   for b in range(self.max_branches) if bool(self.branch_mask[idx, b])]
        return multi_recon.reconstruct_fused_mesh(
            self.phi[idx].numpy(), atype, branch_points, branch_mask,
            opening_indices=loops, trimmed_faces=self.trimmed_faces(idx, multi_recon),
            opening_normals=normals, **kwargs)

    # ---- provenance ---------------------------------------------------------

    def health(self, idx):
        """Per-branch flags recorded at generation: rim flatness, how far the
        sensor's plane fit disagreed with its tangent, and how far the endpoint
        had to be pushed to clear the cut plane. Health signals, not answers."""
        i = int(idx)
        return {
            "rim_flatness":         self._d["rim_flatness"][i],
            "svd_disagreement_deg": self._d["svd_disagreement_deg"][i],
            "endpoint_pushed":      self._d["endpoint_pushed"][i],
        }

    def __repr__(self):
        n0 = int((self.aneurysm_type == 0).sum())
        return (f"SyntheticShapeDataset({len(self)} shapes, "
                f"{n0} bifurcated / {len(self) - n0} sidewall, {self.root})")


# ─────────────────────────────────────────────────────────────────────────────
# generation: run once, offline
# ─────────────────────────────────────────────────────────────────────────────
def generate_pool(out_dir=DEFAULT_POOL, n=1000, ghd_vae_ckpt=DEFAULT_GHD_VAE,
                  sensor_ckpt=DEFAULT_SENSOR, device="cuda:0", seed=0, z_amp=1.0,
                  max_branches=MAX_BRANCHES, sample_sensor=False, batch_size=32,
                  refill=True, max_attempts_mult=5, render=24, verbose=True):
    """Draw shapes from the stage-1 VAE, uncap them with the sensor, keep the good ones.

    With `refill` the draw continues until `n` shapes have been ACCEPTED, so the
    pool size is the number asked for rather than whatever survived. Without it,
    exactly `n` shapes are attempted and the rejects simply reduce the total.
    """
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
    from models.morphoformer import SensorUncapper
    from utils.generate_synthetic import load_ghd_vae

    dev = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multi_recon = MultiCanonicalGHDReconstruct(canonical_root=CANONICAL_ROOT, device=dev)
    vae, mean, std, gdim = load_ghd_vae(Path(ghd_vae_ckpt), dev)
    uncapper = SensorUncapper.from_checkpoint(Path(sensor_ckpt), multi_recon, device=dev)
    with_scale = mean.numel() > gdim
    if verbose:
        print(f"stage-1 VAE : {Path(ghd_vae_ckpt).parent.name}/{Path(ghd_vae_ckpt).name}")
        print(f"sensor      : {Path(sensor_ckpt).parent.name}/{Path(sensor_ckpt).name} "
              f"(probabilistic={uncapper.is_probabilistic})")
        print(f"target      : {n} accepted shape(s), z ~ N(0,{z_amp}^2), seed {seed}\n")

    nv_max = max(int(multi_recon.get(t).canonical_Meshes.verts_packed().shape[0])
                 for t in range(NUM_TYPES))
    g = torch.Generator(device="cpu").manual_seed(seed)
    rec = _Accumulator(max_branches, nv_max)
    attempts = 0
    max_attempts = n * max_attempts_mult if refill else n
    rejected = {"rim_not_simple": 0}

    while rec.n < n and attempts < max_attempts:
        want = min(batch_size, (n - rec.n) if refill else (n - attempts))
        # Balanced across types, in whole halves, so neither canonical is
        # over-represented in the pool the stage-2 loss averages over.
        per = max(1, want // NUM_TYPES)
        types = torch.arange(NUM_TYPES).repeat_interleave(per)
        b = types.numel()
        z = torch.randn(b, vae.latent_dim, generator=g) * z_amp
        with torch.no_grad():
            o = vae.decode(z.to(dev), types.to(dev), strip_scale=True) if with_scale \
                else vae.decode(z.to(dev), types.to(dev))
            phi_n, scale_n = (o if isinstance(o, tuple) else (o, None))
            m = mean[..., :gdim] if with_scale else mean
            s = std[..., :gdim] if with_scale else std
            phi = (phi_n * s + m).reshape(b, -1, 3)
            scales = ((scale_n * std[..., gdim:] + mean[..., gdim:]).squeeze(1).cpu().numpy()
                      if with_scale and scale_n is not None
                      else np.ones(b, dtype=np.float32))

        for i in range(b):
            attempts += 1
            t = int(types[i])
            u = uncapper.uncap(phi[i], t, sample=sample_sensor)
            # The whole acceptance test: as many usable caps as this type has
            # openings. `rim_ok` is per branch and already encodes "one simple
            # cycle", which is what the sweeper needs and what a split cap fails.
            if not bool(np.asarray(u["rim_ok"], dtype=bool).all()):
                rejected["rim_not_simple"] += 1
                continue
            case = f"syn_{seed}_{rec.n:05d}_t{t}"
            rec.add(case, t, float(scales[i]), phi[i].detach().cpu().numpy(),
                    z[i].numpy(), u, uncapper, multi_recon)
            if refill and rec.n >= n:
                break
        if verbose:
            print(f"  {rec.n:5d} accepted / {attempts:5d} attempted "
                  f"({100.0 * rec.n / max(attempts, 1):.1f}%)", flush=True)

    if rec.n == 0:
        raise RuntimeError("no shape passed the cap check -- is the sensor checkpoint right?")

    arrays = rec.finish()
    np.savez_compressed(out_dir / "pool.npz", **arrays)
    manifest = {
        "n_shapes": int(rec.n),
        "n_attempted": int(attempts),
        "acceptance_rate": round(rec.n / max(attempts, 1), 4),
        "rejected": rejected,
        "n_bifurcated": int((arrays["aneurysm_type"] == 0).sum()),
        "n_sidewall": int((arrays["aneurysm_type"] == 1).sum()),
        "ghd_vae": str(ghd_vae_ckpt),
        "sensor": str(sensor_ckpt),
        "sensor_probabilistic": bool(uncapper.is_probabilistic),
        "sensor_sampled": bool(sample_sensor),
        "seed": int(seed), "z_amp": float(z_amp),
        "max_branches": int(max_branches),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(f"\n{rec.n} shape(s) -> {out_dir / 'pool.npz'}  "
              f"({(out_dir / 'pool.npz').stat().st_size / 1e6:.1f} MB)")
        print(f"acceptance {100 * manifest['acceptance_rate']:.1f}% "
              f"({rejected['rim_not_simple']} rejected for a cap that does not close)")

    if render:
        _preview(out_dir, multi_recon, min(render, rec.n))
    return out_dir


class _Accumulator:
    """Collects per-shape arrays and packs them into one npz at the end.

    Caps are bit-packed and rim loops are stored flat with offsets, because the
    two vary in length: caps across types (4143 vs 2590 vertices) and rims from
    shape to shape. Padding either to a rectangle would waste most of the file.
    """

    def __init__(self, max_branches, nv_max):
        self.mb, self.nv_max = max_branches, nv_max
        self.n = 0
        self.case, self.atype, self.scale, self.phi, self.z = [], [], [], [], []
        self.starts, self.starts_raw = [], []
        self.dirs, self.mask, self.nverts = [], [], []
        self.cap_bits, self.dome_bits = [], []
        self.rim_flat, self.rim_off = [], []
        self.flat, self.disag, self.pushed = [], [], []

    def add(self, case, atype, scale, phi, z, u, uncapper, multi_recon):
        mb, nvm = self.mb, self.nv_max
        nv = u["verts"].shape[0]
        n_open = u["cap_masks"].shape[0]
        faces = multi_recon.get(atype).canonical_Meshes.faces_packed().cpu().numpy()

        caps = np.zeros((mb, nvm), dtype=bool)
        caps[:n_open, :nv] = u["cap_masks"]
        starts = np.zeros((mb, 3), np.float32); starts[:n_open] = u["start_points"]
        dirs = np.zeros((mb, 3), np.float32); dirs[:n_open] = u["directions"]
        # undo the nudge rather than re-running the sensor: push_outside moved
        # the endpoint exactly `endpoint_pushed` along the tangent.
        raw = starts - dirs * np.pad(np.asarray(u["endpoint_pushed"], np.float32),
                                     (0, mb - n_open))[:, None]
        mask = np.zeros(mb, bool); mask[:n_open] = True

        offs = np.zeros((mb, 2), np.int64)
        for b in range(n_open):
            loop = np.asarray(uncapper.rim_loop(u["cap_masks"][b], faces), dtype=np.int32)
            offs[b] = (sum(len(x) for x in self.rim_flat), len(loop))
            self.rim_flat.append(loop)

        pad = lambda a: np.pad(np.asarray(a, np.float32), (0, mb - n_open))
        self.case.append(case); self.atype.append(atype); self.scale.append(scale)
        self.phi.append(phi.reshape(-1, 3).astype(np.float32)); self.z.append(z.astype(np.float32))
        self.starts.append(starts); self.starts_raw.append(raw)
        self.dirs.append(dirs); self.mask.append(mask)
        self.nverts.append(nv)
        self.cap_bits.append(np.packbits(caps, axis=-1))
        dome = u.get("dome")
        self.dome_bits.append(np.packbits(
            np.pad(np.asarray(dome, bool), (0, nvm - nv)) if dome is not None
            else np.zeros(nvm, bool)))
        self.rim_off.append(offs)
        self.flat.append(pad(u["rim_flatness"])); self.disag.append(pad(u["svd_disagreement_deg"]))
        self.pushed.append(pad(u["endpoint_pushed"]))
        self.n += 1

    def finish(self):
        return {
            "case":                 np.array(self.case),
            "aneurysm_type":        np.array(self.atype, np.int64),
            "scale":                np.array(self.scale, np.float32),
            "phi":                  np.stack(self.phi),
            "z":                    np.stack(self.z),
            "start_points":         np.stack(self.starts),
            "start_points_raw":     np.stack(self.starts_raw),
            "directions":           np.stack(self.dirs),
            "branch_mask":          np.stack(self.mask),
            "n_verts":              np.array(self.nverts, np.int64),
            "cap_bits":             np.stack(self.cap_bits),
            "dome_bits":            np.stack(self.dome_bits),
            "rim_flat":             np.concatenate(self.rim_flat) if self.rim_flat
                                    else np.zeros(0, np.int32),
            "rim_offsets":          np.stack(self.rim_off),
            "rim_flatness":         np.stack(self.flat),
            "svd_disagreement_deg": np.stack(self.disag),
            "endpoint_pushed":      np.stack(self.pushed),
        }


def _preview(out_dir, multi_recon, n, ncols=6):
    """Contact sheet of the finished pool: mesh, predicted caps, tangents.

    Written so the pool can be eyeballed before a run is built on it -- a sensor
    pointed the wrong way is obvious here and invisible in the loss curves.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ds = SyntheticShapeDataset(out_dir)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.0 * ncols, 3.0 * nrows))
    colors = ["#e6194B", "#3cb44b", "#4363d8"]
    for i in range(n):
        t = int(ds.aneurysm_type[i])
        v = multi_recon._reconstruct_verts_np(ds.phi[i].numpy(), t)
        f = multi_recon.get(t).canonical_Meshes.faces_packed().cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                        color="lightsteelblue", edgecolor="none", alpha=0.35)
        span = float((v.max(0) - v.min(0)).max())
        for b in range(ds.max_branches):
            if not bool(ds.branch_mask[i, b]):
                continue
            cap = ds.cap_mask(i, b)
            ax.scatter(*v[cap].T, s=3, color=colors[b % 3], depthshade=False)
            st = ds.start_points[i, b].numpy(); dr = ds.branch_direction[i, b].numpy()
            ax.quiver(*st, *(dr * 0.22 * span), color=colors[b % 3], lw=2)
        ax.set_axis_off(); ax.view_init(elev=18, azim=35)
        ax.set_title(f"{ds.cases[i]}", fontsize=6)
        c, r = (v.min(0) + v.max(0)) / 2, span / 2 or 1.0
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    fig.suptitle(f"{out_dir.name}: caps and tangents from the sensor")
    fig.tight_layout()
    path = out_dir / "pool_preview.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"preview -> {path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_POOL))
    ap.add_argument("--n", type=int, default=1000,
                    help="Shapes to ACCEPT (rejects are redrawn unless --no-refill).")
    ap.add_argument("--ghd-vae", default=str(DEFAULT_GHD_VAE))
    ap.add_argument("--sensor", default=str(DEFAULT_SENSOR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-amp", type=float, default=1.0,
                    help="z ~ N(0, amp^2). Leave at 1.0: that is the prior the KL "
                         "term trained against and the distribution stage 2 will meet.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sample-sensor", action="store_true",
                    help="Probabilistic sensor only: draw a labelling from the prior "
                         "instead of taking its mean. Off by default so the pool is "
                         "reproducible from the seed alone.")
    ap.add_argument("--no-refill", dest="refill", action="store_false",
                    help="Attempt exactly --n shapes instead of accepting --n.")
    ap.add_argument("--render", type=int, default=24,
                    help="Shapes on the preview sheet; 0 to skip.")
    a = ap.parse_args()
    generate_pool(out_dir=a.out, n=a.n, ghd_vae_ckpt=a.ghd_vae, sensor_ckpt=a.sensor,
                  device=a.device, seed=a.seed, z_amp=a.z_amp, batch_size=a.batch_size,
                  sample_sensor=a.sample_sensor, refill=a.refill, render=a.render)


if __name__ == "__main__":
    main()
