"""Dataset for the ostium model (1D graph -> site + ostium) and the complex model (-> GHD latent
+ affine). Design: TASK_vessel_conditioned_aneurysm.md section 5.6.

Reads runtime_dataset/assembled/<dataset>/<case>/vessel_graph/vessel_graph.npz (written by
dataset/preprocess_vessel_graph.py) and, for the complex targets, the case's GHD fit and the
Stage-1 GHD-VAE.

Per sample (training mode):
  1. trim off-centre: random crop centre Q near (not at) the site, random geodesic radius from
     `crop_min_mm` up to the whole graph (probability `p_whole` of keeping all of it); every graph
     point within `keep_mm` (Euclidean) of the ostium rim and within 15 mm along the graph of the
     site is always kept. The crop is a geodesic ball, so it is one connected piece.
  2. random SO(3) rotation of everything; random sigma (tangent sign)
  3. site jitter: P' = a random graph point within `jitter_mm` (along the graph) of the true P;
     the rim is re-encoded at P' (teacher forcing with a slightly-off site)
  4. normalise: centre on the crop's centroid, positions in units of `unit_mm` (10 mm)
  5. hierarchy: farthest-point sampling to ~1 mm (level 1) and ~4 mm (level 2); every finer
     point is pooled into its nearest seed ALONG THE GRAPH; coarse edges join seeds whose
     regions touch; Laplacian eigenvectors (+ eigenvalues, random sign flips) and random-walk
     encodings per coarse level.
Validation mode: whole graph, no rotation, sigma = +1, no jitter.

Model inputs never include branch_id / the aneurysm mask / geo_dist_from_site / the true site
beyond the conditioning site P' (they reveal the answer).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[1]
ASSEMBLED = REPO / "runtime_dataset" / "assembled"
VAE_CKPT = REPO / "runtime_train/ghd_vae/stage1/ghd_vae_gan_h256_z16_kl2_adv0.3/epoch_05000.pth"
MORPH_REFERENCE = REPO / "dataset" / "ostium_morph_reference.json"   # condition labels, see scripts/ostium/morph_reference.py


# ---------------------------------------------------------------------- batching

class OstiumData(Data):
    """PyG Data with the coarse levels' index fields offset correctly when batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("site", "site_true"):
            return self.num_nodes
        if key == "edge_index_1":
            return int(self.n1)
        if key == "edge_index_2":
            return int(self.n2)
        if key == "assign_01":
            return int(self.n1)
        if key == "assign_12":
            return int(self.n2)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("edge_index_1", "edge_index_2"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


FOLLOW_BATCH = ["pos_1", "pos_2"]


# ---------------------------------------------------------------------- helpers

def patient_id(dataset, case):
    """Approximate patient key for the train/val split (cases of one scan stay together)."""
    if dataset == "ImperialNHS":
        return case.split("_aneurysm")[0]
    c = re.sub(r"_cut\d+$", "", case)
    c = re.sub(r"_(L|R)?ICA.*$", "", c)
    c = re.sub(r"_ID\d+$", "", c)
    c = re.sub(r"\.\d+$", "", c)
    if c.startswith("ANSYS_"):
        c = "_".join(c.split("_")[:3])
    return f"AneuX:{c}"


def in_val(pid, val_frac, seed):
    h = int(hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < val_frac


def adjacency(pos, ei):
    import scipy.sparse as sp
    w = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1) + 1e-9
    n = len(pos)
    return sp.csr_matrix((w, (ei[0], ei[1])), shape=(n, n))


def geodesic(A, src):
    from scipy.sparse.csgraph import dijkstra
    return dijkstra(A, directed=False, indices=src)


def voronoi(A, seeds):
    """Nearest seed along the graph for every point: (index into seeds)."""
    from scipy.sparse.csgraph import dijkstra
    _, _, src = dijkstra(A, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    lut = -np.ones(A.shape[0], int)
    lut[seeds] = np.arange(len(seeds))
    return lut[src]


def coarse_edges(ei, lab):
    a, b = lab[ei[0]], lab[ei[1]]
    m = a != b
    e = np.unique(np.sort(np.stack([a[m], b[m]], 1), axis=1), axis=0)
    return np.concatenate([e.T, e.T[::-1]], axis=1) if len(e) else np.zeros((2, 0), int)


def fps(pos, ratio, rng):
    from torch_cluster import fps as _fps
    if len(pos) <= 2:
        return np.arange(len(pos))
    idx = _fps(torch.as_tensor(pos, dtype=torch.float32), ratio=float(ratio), random_start=True)
    return np.sort(idx.numpy())


def positional_encodings(n, ei, k, rw_steps, rng, flip):
    """Laplacian eigenvectors (k smallest non-trivial) + their eigenvalues, random-walk PE."""
    A = np.zeros((n, n))
    if ei.shape[1]:
        A[ei[0], ei[1]] = 1.0
    deg = A.sum(1)
    L = np.diag(deg) - A
    from scipy.linalg import eigh
    hi = min(k, n - 1)                                     # only the k+1 smallest are needed
    vals, vecs = eigh(L, subset_by_index=[0, hi], driver="evr")
    vecs, vals = vecs[:, 1:k + 1], vals[1:k + 1]
    if flip:
        vecs = vecs * rng.choice([-1.0, 1.0], size=vecs.shape[1])[None]
    ev = np.zeros((n, k))
    ew = np.zeros((n, k))
    ev[:, :vecs.shape[1]] = vecs
    ew[:, :vals.shape[0]] = vals[None]
    P = A / np.maximum(deg, 1)[:, None]
    rw, M = np.zeros((n, rw_steps)), np.eye(n)
    for s_ in range(rw_steps):
        M = M @ P
        rw[:, s_] = np.diag(M)
    return ev, ew, rw


# ---------------------------------------------------------------------- dataset

def graph_inputs(pos, radius, ei, rng, unit=10.0, l1_mm=1.0, l2_mm=4.0, pe_k=8, rw_steps=8, flip_pe=False, A=None):
    """Model inputs of a point graph in mm: node features, the two coarse levels and their positional
    encodings. Shared by training (OstiumGraphDataset) and inference, so both build exactly the same
    inputs. rng draws the farthest-point seeds and (flip_pe) the eigenvector signs.
    Returns (OstiumData fields, centre in mm); positions in the fields are (pos - centre) / unit."""
    n = len(pos)
    A = adjacency(pos, ei) if A is None else A
    deg = np.bincount(ei[0], minlength=n)
    tang = np.zeros((n, 3))
    dv = pos[ei[1]] - pos[ei[0]]
    dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
    dv *= np.sign(dv[:, :1] + 1e-12)                            # sign-free direction
    np.add.at(tang, ei[0], dv)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    tt = np.stack([tang[:, 0] ** 2, tang[:, 1] ** 2, tang[:, 2] ** 2,
                   tang[:, 0] * tang[:, 1], tang[:, 0] * tang[:, 2], tang[:, 1] * tang[:, 2]], 1)
    centre = pos.mean(0)
    posn = (pos - centre) / unit
    feat = np.concatenate([posn, radius[:, None] / unit, tt,
                           (deg >= 3)[:, None].astype(float), (deg == 1)[:, None].astype(float)], 1)
    # hierarchy
    mean_edge = float(np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1).mean())
    seeds1 = fps(pos, min(1.0, mean_edge / l1_mm), rng)           # ~1 point per l1_mm
    lab1 = voronoi(A, seeds1)
    ei1 = coarse_edges(ei, lab1)
    seeds2_in1 = fps(pos[seeds1], min(1.0, l1_mm / l2_mm), rng)
    lab2_full = voronoi(A, seeds1[seeds2_in1])
    assign12 = lab2_full[seeds1]
    ei2 = coarse_edges(ei, lab2_full)
    pe1 = positional_encodings(len(seeds1), ei1, pe_k, rw_steps, rng, flip_pe)
    pe2 = positional_encodings(len(seeds2_in1), ei2, pe_k, rw_steps, rng, flip_pe)
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    i64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
    fields = dict(
        x=f32(feat), pos=f32(posn), edge_index=i64(ei), radius=f32(radius / unit),
        pos_1=f32(posn[seeds1]), edge_index_1=i64(ei1), assign_01=i64(lab1), n1=len(seeds1),
        pe_1=f32(pe1[0]), eig_1=f32(pe1[1]), rw_1=f32(pe1[2]),
        pos_2=f32(posn[seeds1[seeds2_in1]]), edge_index_2=i64(ei2), assign_12=i64(assign12), n2=len(seeds2_in1),
        pe_2=f32(pe2[0]), eig_2=f32(pe2[1]), rw_2=f32(pe2[2]))
    return fields, centre


class OstiumGraphDataset(Dataset):
    def __init__(self, split="train", datasets=("ImperialNHS", "AneuX"), root=ASSEMBLED,
                 val_frac=0.15, split_seed=0, augment=None, H=5, M=128, unit_mm=10.0,
                 keep_mm=6.0, crop_min_mm=8.0, p_whole=0.2, centre_frac=0.8, jitter_mm=0.3,
                 site_sigma_mm=1.0, l1_mm=1.0, l2_mm=4.0, pe_k=8, rw_steps=8, n_ring=64,
                 complex_targets=True, vae_ckpt=VAE_CKPT, cases=None, seed=None, val_cases=None,
                 point_frames=False, rotate=True, crop=True, rot_max_deg=180.0, morph=False):
        """split: "train" | "val" | "all". augment defaults to (split == "train"): pass it
        explicitly for split "all" (training on every case needs augment=True). The split is by patient: with `val_cases` (case names)
        every case of those cases' patients is validation; otherwise a hash of the patient id
        (val_frac, split_seed). point_frames: also store the signed tangent and the local scale at
        every graph point (tangent_pts, scale_pts), so the ostium head can be run at a sampled site
        (evaluation only; not needed for training). rotate: with augment, a random SO(3) rotation per
        sample (False keeps the scan orientation; trimming, sigma and jitter are unchanged). crop: with
        augment, trim off-centre (False always keeps the whole graph; the rest unchanged).
        rot_max_deg: with rotate, below 180 the rotation is about a uniformly random axis by an angle
        uniform in [0, rot_max_deg] (180: a uniformly random rotation, as before). morph: also store the
        case's morphology condition [log dome volume (mm3), aspect ratio] (morph, raw) from
        MORPH_REFERENCE (scripts/ostium/morph_reference.py); it does not change with augmentation."""
        self.split, self.H, self.M, self.unit = split, H, M, unit_mm
        self.augment = (split == "train") if augment is None else augment
        self.keep_mm, self.crop_min_mm, self.p_whole = keep_mm, crop_min_mm, p_whole
        self.centre_frac, self.jitter_mm, self.site_sigma = centre_frac, jitter_mm, site_sigma_mm
        self.l1_mm, self.l2_mm, self.pe_k, self.rw_steps, self.n_ring = l1_mm, l2_mm, pe_k, rw_steps, n_ring
        self.complex_targets, self.point_frames, self.rotate, self.crop = complex_targets, point_frames, rotate, crop
        self.rot_max_deg = rot_max_deg
        self.rng = np.random.default_rng(seed)
        found = [(ds, f.parent.parent.name, f) for ds in datasets
                 for f in sorted((Path(root) / ds).glob("*/vessel_graph/vessel_graph.npz"))]
        val_pids = None
        if val_cases is not None:
            val_cases = set(val_cases)
            val_pids = {(ds, patient_id(ds, c)) for ds, c, _ in found if c in val_cases}
        items = []
        for ds, case, f in found:
            if cases is not None and case not in cases:
                continue
            if val_pids is not None:
                v = (ds, patient_id(ds, case)) in val_pids
            else:
                v = in_val(patient_id(ds, case), val_frac, split_seed)
            if split == "all" or (split == "val") == v:
                items.append((ds, case, f))
        self.items = items
        self.morph = None
        if morph:
            ref = json.loads(Path(MORPH_REFERENCE).read_text())
            self.morph = {(r["ds"], r["case"]): [float(np.log(r["volume_mm3"])), r["aspect_ratio"]] for r in ref["cases"]}
        self._cache = {}
        self._vae = None
        if complex_targets:
            self._load_vae(vae_ckpt)

    # ---- static per-case data ------------------------------------------------
    def _load_vae(self, ckpt):
        from models.ghd_vae import TypeConditionalVAE
        from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        h = ck["hparams"]
        dim = int(ck["mean"].numel()) - 1
        vae = TypeConditionalVAE(dim, h["hidden_dim"], h["latent_dim"], withscale=True, num_types=3)
        vae.load_state_dict(ck["model"])
        self._vae = vae.eval()
        self._vae_mean, self._vae_std = ck["mean"].reshape(-1), ck["std"].reshape(-1)
        mc = MultiCanonicalGHDReconstruct(str(REPO / "dataset" / "canonical"))
        self._norm_can = {t: float(mc.get(t).norm_canonical) for t in (0, 1, 2)}

    @torch.no_grad()
    def vae_encode(self, phi, log_scale):
        x = torch.as_tensor(np.r_[np.asarray(phi).ravel(), np.exp(log_scale)], dtype=torch.float32)
        x = (x - self._vae_mean) / self._vae_std
        mu, _ = self._vae.encode(x[None, :-1], x[None, -1:])
        return mu[0].numpy()

    def case_data(self, i):
        if i in self._cache:
            return self._cache[i]
        from geometry.ostium import _arc_resample
        ds, case, f = self.items[i]
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        d = dict(ds=ds, case=case, pos=z["pos"].astype(float), radius=z["radius"].astype(float),
                 ei=z["edge_index"].astype(int), rim=z["rim"].astype(float), site=int(z["site_point"]),
                 atype=int(meta["aneurysm_type"]))
        d["C"] = _arc_resample(d["rim"], self.M).mean(0)
        if self.complex_targets:
            from geometry.complex_target import fit_parts
            fp = fit_parts(case, ds, self._norm_can[d["atype"]])
            d.update(fp)
            d["z"] = self.vae_encode(fp["phi"], fp["log_scale"])
        self._cache[i] = d
        return d

    def __len__(self):
        return len(self.items)

    # ---- augmentation ---------------------------------------------------------
    def trim(self, d, rng):
        pos, ei, P = d["pos"], d["ei"], d["site"]
        A = adjacency(pos, ei)
        n = len(pos)
        if not self.augment or not self.crop or rng.random() < self.p_whole:
            return np.arange(n), A
        gP = geodesic(A, P)
        from scipy.spatial import cKDTree
        near_rim = cKDTree(d["rim"]).query(pos)[0] < self.keep_mm
        keep = near_rim & (gP < 15.0)
        keep[P] = True
        R = float(np.exp(rng.uniform(np.log(self.crop_min_mm), np.log(max(self.crop_min_mm * 1.01, gP.max())))))
        dqp = rng.uniform(0, self.centre_frac * R)
        cand = np.flatnonzero(np.abs(gP - dqp) < 0.5)
        Q = int(rng.choice(cand)) if len(cand) else P
        gQ = geodesic(A, Q)
        R2 = max(R, float(gQ[keep].max()))
        sel = np.flatnonzero(gQ <= R2 + 1e-6)
        return sel, A

    # ---- one sample -------------------------------------------------------------
    def __getitem__(self, i):
        from scipy.spatial.transform import Rotation
        from geometry.ostium import code_to_vector, encode_ostium, local_scale, graph_frame_axes
        from geometry.complex_target import affine_in_frame, rotate_scene
        rng = self.rng
        d = self.case_data(i)
        sel, _ = self.trim(d, rng)
        remap = -np.ones(len(d["pos"]), int)
        remap[sel] = np.arange(len(sel))
        m = (remap[d["ei"][0]] >= 0) & (remap[d["ei"][1]] >= 0)
        ei = remap[d["ei"][:, m]]
        pos = d["pos"][sel]
        radius = d["radius"][sel]
        P = int(remap[d["site"]])
        if not (self.augment and self.rotate):
            G = np.eye(3)
        elif self.rot_max_deg >= 180:
            G = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
        else:                                               # small rotations about a random axis
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            G = Rotation.from_rotvec(ax * np.radians(rng.uniform(0, self.rot_max_deg))).as_matrix()
        pos = pos @ G.T
        rim = d["rim"] @ G.T
        n = len(pos)
        A = adjacency(pos, ei)
        deg = np.bincount(ei[0], minlength=n)
        gP = geodesic(A, P)
        # site jitter (teacher forcing with a slightly-off site)
        if self.augment and self.jitter_mm > 0:
            cand = np.flatnonzero(gP <= self.jitter_mm)
            Pj = int(rng.choice(cand))
        else:
            Pj = P
        sigma = int(rng.choice([-1, 1])) if self.augment else 1
        root = int(np.flatnonzero(deg == 1)[0]) if (deg == 1).any() else 0
        pg = dict(pos=pos, edge_index=ei, radius=radius, meta=dict(root_point=root))
        code = encode_ostium(rim, pg, H=self.H, M=self.M, sign=sigma, site=Pj)
        y = code_to_vector(code)
        t = graph_frame_axes(pg, Pj, sigma)
        s = code["scale"]
        C = pos[Pj] + code["offset"] * s
        Rf = code["frame"]
        # smooth target ring, in the ostium frame (units of s) and in image space
        from geometry.ostium import decode_ostium
        ring_img = decode_ostium(code, pg, M=self.n_ring)
        ring_loc = (ring_img - C) @ Rf.T / s
        # site soft label
        q = np.exp(-0.5 * (gP / self.site_sigma) ** 2)
        q /= q.sum()
        fields, centre = graph_inputs(pos, radius, ei, rng, unit=self.unit, l1_mm=self.l1_mm, l2_mm=self.l2_mm,
                                      pe_k=self.pe_k, rw_steps=self.rw_steps, flip_pe=self.augment, A=A)
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        i64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
        data = OstiumData(
            **fields, site_q=f32(q), site=i64([Pj]), site_true=i64([P]),
            y_ost=f32(y)[None], tangent=f32(t)[None], scale=f32([[s / self.unit]]),
            sigma=f32([sigma]), ring_loc=f32(ring_loc)[None], ring_img=f32((ring_img - centre) / self.unit)[None],
            frame=f32(Rf)[None], C=f32((C - centre) / self.unit)[None],
            atype=i64([d["atype"]]), case=d["case"], ds=d["ds"])
        if self.point_frames:
            from geometry.ostium import graph_tangents, local_scales
            data.tangent_pts = f32(graph_tangents(pg, sigma))
            data.scale_pts = f32(local_scales(pg)[:, None] / self.unit)
        if self.complex_targets:
            Qr, br = rotate_scene(d["Q"], d["b"], G)
            aff = affine_in_frame(Qr, d["k"], br, C, Rf, s)
            data.x_cx = f32(np.r_[d["z"], aff])[None]
            data.phi = f32(d["phi"])[None]
            data.log_scale = f32([d["log_scale"]])
        if self.morph is not None:
            data.morph = f32([self.morph[(d["ds"], d["case"])]])
        return data


def _worker_init(_):
    import os
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (1 << 32))   # different augmentations per worker


def make_loader(ds, batch_size, shuffle, num_workers=4):
    from torch_geometric.loader import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      follow_batch=FOLLOW_BATCH, persistent_workers=num_workers > 0,
                      worker_init_fn=_worker_init if num_workers > 0 else None)
