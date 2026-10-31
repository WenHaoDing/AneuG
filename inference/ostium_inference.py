"""Inference for the vessel-graph ostium model.

From a 1D vessel graph (e.g. a healthy 96^3 patch exported by the image-generation repo, see
HANDOFF_inference_1d_graph.md) to sampled aneurysms:

  graph -> training-like crops -> site score per point (averaged over crops) -> sampled sites
        -> per site: a training-like crop around it -> ostium code y -> ring
        -> aneurysm type -> complex x = [VAE latent, affine] -> frozen VAE decoder -> GHD mesh
        -> placed in the graph's mm frame -> frozen morphology sensor -> dome patch

Inference graphs are larger and branchier than what the model sees in training (whole vessel
pieces of a patch against off-centre training crops), so every forward pass runs on a crop drawn
like a training crop. Inputs are built by dataset.ostium_dataset.graph_inputs, the same function
training uses. Every sample gets plausibility checks (ring against the skeleton, dome against the
vessel tubes).

Prompts (optional): a prompt is a graph point where the user wants the aneurysm. By default the
site is the prompt itself (run_inference.py --prompt_sigma 0). With a width sigma > 0 it is drawn
instead from p(site | prompt) ~ p(site) * exp(-d(site, prompt)^2 / 2 sigma^2), d along the graph:
the model's site score times a Gaussian proximity kernel. This is the posterior a prompt-conditioned
site head would learn if trained with prompts sampled within ~sigma of the true site, so no model
is retrained and every trained checkpoint works with it. sigma sets how strictly the prompt binds;
within that reach the model's own score picks the site. Random prompts avoid the vessel ends:
at least min_end_mm along the graph from any end (default: the training sites' 5th percentile).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))
REFERENCE = REPO / "dataset" / "ostium_crop_reference.json"
PROMPT_REFERENCE = REPO / "dataset" / "ostium_prompt_reference.json"
MORPH_REFERENCE = REPO / "dataset" / "ostium_morph_reference.json"


# ------------------------------------------------------------------------------ graphs

def load_graph(path):
    """A point graph .npz (save_point_graph format): pos (mm), radius (mm), edge_index, plus the
    optional in_core mask and meta of the image-model test graphs."""
    z = np.load(path, allow_pickle=True)
    g = dict(pos=z["pos"].astype(float), radius=z["radius"].astype(float), edge_index=z["edge_index"].astype(int))
    g["in_core"] = z["in_core"].astype(bool) if "in_core" in z.files else np.ones(len(g["pos"]), bool)
    g["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return g


def graph_stats(pos, ei, keep=None):
    """Centerline length (mm), ends and junctions of a graph, or of the subgraph `keep` (mask)."""
    n = len(pos)
    keep = np.ones(n, bool) if keep is None else keep
    m = keep[ei[0]] & keep[ei[1]]
    deg = np.bincount(ei[0][m], minlength=n)[keep]
    length = float(np.linalg.norm(pos[ei[0][m]] - pos[ei[1][m]], axis=1).sum() / 2)
    return dict(length_mm=round(length, 1), ends=int((deg == 1).sum()), junctions=int((deg >= 3).sum()))


def subgraph(g, sel):
    """The graph restricted to the points `sel` (indices), with edges renumbered."""
    remap = -np.ones(len(g["pos"]), int)
    remap[sel] = np.arange(len(sel))
    ei = g["edge_index"]
    m = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    return dict(pos=g["pos"][sel], radius=g["radius"][sel], edge_index=remap[ei[:, m]]), remap


def cut_ends(g, sel):
    """Points of the crop `sel` that lost a neighbour to the crop (a vessel cut by the crop)."""
    ei = g["edge_index"]
    n = len(g["pos"])
    keep = np.zeros(n, bool)
    keep[sel] = True
    m = keep[ei[0]] & keep[ei[1]]
    deg_full = np.bincount(ei[0], minlength=n)
    deg_crop = np.bincount(ei[0][m], minlength=n)
    return np.flatnonzero(keep & (deg_crop < deg_full))


# ------------------------------------------------------------------------------ training reference

def training_reference(path=REFERENCE, n_crops=3, seed=0):
    """Distributions the cropper imitates, from the training cases (cached as JSON):
    extent of each whole training graph from its site (along the graph), and the length / ends /
    junctions of whole graphs and of training crops. Quantiles 0..100."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    from dataset.ostium_dataset import OstiumGraphDataset, adjacency, geodesic
    vn = [r["case"] for r in json.load(open(REPO / "dataset" / "ostium_val_cases.json"))["cases"]]
    ds = OstiumGraphDataset("train", augment=True, complex_targets=False, val_cases=vn, seed=seed)
    rng = np.random.default_rng(seed)
    extent, whole, crops = [], [], []
    for i in range(len(ds)):
        d = ds.case_data(i)
        g = geodesic(adjacency(d["pos"], d["ei"]), d["site"])
        extent.append(float(g.max()))
        whole.append(graph_stats(d["pos"], d["ei"]))
        for _ in range(n_crops):
            sel, _ = ds.trim(d, rng)
            keep = np.zeros(len(d["pos"]), bool)
            keep[sel] = True
            crops.append(graph_stats(d["pos"], d["ei"], keep))
    q = lambda v: [float(x) for x in np.percentile(v, np.arange(101))]
    ref = dict(
        note="ostium model training reference: quantiles 0..100 (see inference/ostium_inference.py)",
        whole_extent_from_site_mm=q(extent),
        whole={k: q([r[k] for r in whole]) for k in ("length_mm", "ends", "junctions")},
        crops={k: q([r[k] for r in crops]) for k in ("length_mm", "ends", "junctions")},
        trim=dict(p_whole=ds.p_whole, crop_min_mm=ds.crop_min_mm, centre_frac=ds.centre_frac, keep_mm=ds.keep_mm))
    path.write_text(json.dumps(ref, indent=1))
    return ref


def sample_quantile(quantiles, rng):
    """One draw from a distribution given by its quantiles 0..100 (linear interpolation)."""
    return float(np.interp(rng.uniform(0, 100), np.arange(101), quantiles))


def training_like_crop(g, A, site, rng, ref, protect_mm=7.0):
    """Points of a crop drawn like a training crop around `site` (dataset trim rule):
      - an extent E is drawn from the training graphs' own extents from their sites;
      - with p_whole the crop is everything within E of the site (a "whole" training graph);
      - otherwise a ball along the graph, radius R log-uniform in [crop_min, E], centred on a
        point Q at distance U(0, centre_frac R) from the site;
      - everything within protect_mm of the site is always kept (training keeps everything near
        the ostium).
    Returns the sorted point indices."""
    from dataset.ostium_dataset import geodesic
    t = ref["trim"]
    gP = geodesic(A, site)
    E = sample_quantile(ref["whole_extent_from_site_mm"], rng)
    keep = gP <= protect_mm
    if rng.random() < t["p_whole"]:
        return np.flatnonzero(gP <= max(E, protect_mm))
    lo = t["crop_min_mm"]
    R = float(np.exp(rng.uniform(np.log(lo), np.log(max(lo * 1.01, E)))))
    dqp = rng.uniform(0, t["centre_frac"] * R)
    cand = np.flatnonzero(np.abs(gP - dqp) < 0.5)
    Q = int(rng.choice(cand)) if len(cand) else site
    gQ = geodesic(A, Q)
    R2 = max(R, float(gQ[keep].max()))
    return np.flatnonzero(gQ <= R2 + 1e-6)


# ------------------------------------------------------------------------------ model

def load_model(ckpt, device="cpu"):
    """Trained OstiumModel, its target standardisation and the frozen decoder chain (VAE decoder,
    GHD reconstruction, morphology sensor)."""
    import train_ostium_model as T
    from dataset.ostium_dataset import VAE_CKPT, OstiumGraphDataset
    from models.ostium_model import MorphOstiumModel, OstiumModel
    dev = torch.device(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    kw = dict(d=a["d"], mode=a["mode"], with_complex=not a.get("no_complex", False),
              flow_h=a.get("flow_h", 256), flow_blocks=a.get("flow_blocks", 4), norm_tokens=bool(a.get("norm_tokens", 0)))
    # --morph_cond 1 runs: conditioned on the dome's [log volume, aspect ratio] (stats are in the weights)
    model = (MorphOstiumModel(**kw) if a.get("morph_cond", 0) else OstiumModel(**kw)).to(dev).eval()
    model.load_state_dict(ck["model"])
    ys = T.Stats(torch.zeros(2, 33))
    ys.mean, ys.std = ck["y_stats"]
    ys.to(dev)
    xs = T.Stats(torch.zeros(2, 26))
    xs.mean, xs.std = ck["x_stats"]
    xs.to(dev)
    holder = SimpleNamespace()
    OstiumGraphDataset._load_vae(holder, VAE_CKPT)          # the dataset's own loader, without a dataset
    dec = T.Decoder(holder, dev, with_sensor=True)
    return SimpleNamespace(model=model, ys=ys, xs=xs, dec=dec, args=a, epoch=ck["epoch"], dev=dev, T=T)


def make_batch(gc, rng, args, site=None, sigma=1):
    """Model input for the crop gc (pos/radius/edge_index in mm), as one batch on CPU.
    With a site: also its signed tangent and local scale (the ostium head's inputs)."""
    from torch_geometric.data import Batch
    from dataset.ostium_dataset import FOLLOW_BATCH, OstiumData, adjacency, graph_inputs
    from geometry.ostium import graph_tangents, local_scales
    unit = 10.0
    pos, radius, ei = gc["pos"], gc["radius"], gc["edge_index"]
    fields, centre = graph_inputs(pos, radius, ei, rng, unit=unit, flip_pe=False, A=adjacency(pos, ei))
    deg = np.bincount(ei[0], minlength=len(pos))
    root = int(np.flatnonzero(deg == 1)[0]) if (deg == 1).any() else 0
    pg = dict(pos=pos, edge_index=ei, radius=radius, meta=dict(root_point=root))
    tang = graph_tangents(pg, sigma)
    scl = local_scales(pg)
    f32 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    extra = dict(tangent_pts=f32(tang), scale_pts=f32(scl[:, None] / unit))
    if site is not None:
        extra.update(site=torch.tensor([site]), tangent=f32(tang[site])[None], scale=f32([[scl[site] / unit]]))
    data = OstiumData(**fields, **extra)
    return Batch.from_data_list([data], follow_batch=FOLLOW_BATCH), centre


# ------------------------------------------------------------------------------ site

@torch.no_grad()
def site_scores(M, g, rng, ref, n_crops=24, margin_mm=7.0, candidates=None):
    """Site score per graph point: the site head's probability on a crop, times the crop's number
    of points (1 = uniform), averaged over the crops in which the point lies at least margin_mm
    (along the graph) from a cut end. Crops are drawn like training crops around random candidate
    points. Returns (score, coverage count, list of crop statistics)."""
    from torch_geometric.utils import softmax
    from dataset.ostium_dataset import adjacency, geodesic
    A = adjacency(g["pos"], g["edge_index"])
    cand = np.flatnonzero(g["in_core"]) if candidates is None else candidates
    n = len(g["pos"])
    acc, cnt, cstats = np.zeros(n), np.zeros(n), []
    for _ in range(n_crops):
        c = int(rng.choice(cand))
        sel = training_like_crop(g, A, c, rng, ref)
        gc, _ = subgraph(g, sel)
        b, _ = make_batch(gc, rng, M.args)
        b = b.to(M.dev)
        enc = M.model.enc(b)
        p = softmax(M.model.site_logits(enc, b), b.batch).cpu().numpy() * len(sel)
        ends = cut_ends(g, sel)
        far = np.ones(len(sel), bool)
        if len(ends):
            from scipy.sparse.csgraph import dijkstra
            far = dijkstra(A, directed=False, indices=ends, min_only=True)[sel] >= margin_mm
        acc[sel[far]] += p[far]
        cnt[sel[far]] += 1
        keep = np.zeros(n, bool)
        keep[sel] = True
        cstats.append(graph_stats(g["pos"], g["edge_index"], keep))
    score = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return score, cnt, cstats


def sample_sites(score, allowed, k, rng, temperature=1.0):
    """k site indices drawn from score ** (1/temperature) over the allowed points."""
    w = np.where(allowed, np.maximum(score, 0) ** (1.0 / temperature), 0.0)
    if w.sum() <= 0:
        w = allowed.astype(float)
    return rng.choice(len(w), size=k, replace=True, p=w / w.sum())


# ------------------------------------------------------------------------------ prompts

def end_distance(g, A=None):
    """Distance along the graph (mm) from every point to the nearest end (degree-1 point)."""
    from scipy.sparse.csgraph import dijkstra
    from dataset.ostium_dataset import adjacency
    deg = np.bincount(g["edge_index"][0], minlength=len(g["pos"]))
    ends = np.flatnonzero(deg == 1)
    if len(ends) == 0:
        return np.full(len(g["pos"]), np.inf)
    A = adjacency(g["pos"], g["edge_index"]) if A is None else A
    return dijkstra(A, directed=False, indices=ends, min_only=True)


def prompt_reference(path=PROMPT_REFERENCE):
    """Distance (mm, along the graph) from each training site to the nearest end of its graph,
    quantiles 0..100 (cached as JSON). Random prompts keep at least the 5th percentile."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    from dataset.ostium_dataset import OstiumGraphDataset
    vn = [r["case"] for r in json.load(open(REPO / "dataset" / "ostium_val_cases.json"))["cases"]]
    ds = OstiumGraphDataset("train", augment=False, complex_targets=False, val_cases=vn)
    dist = []
    for i in range(len(ds)):
        d = ds.case_data(i)
        dist.append(float(end_distance(dict(pos=d["pos"], edge_index=d["ei"]))[d["site"]]))
    ref = dict(note="distance from training sites to the nearest graph end (mm), quantiles 0..100",
               site_end_distance_mm=[float(x) for x in np.percentile(dist, np.arange(101))], n=len(dist))
    path.write_text(json.dumps(ref, indent=1))
    return ref


def sample_prompts(g, k, rng, min_end_mm, A=None):
    """k random prompts: graph points in the core at least min_end_mm from any end, drawn uniformly
    along the centerline (each point weighted by half its incident edge length)."""
    ok = g["in_core"] & (end_distance(g, A) >= min_end_mm)
    if not ok.any():
        ok = g["in_core"].copy()
    ei = g["edge_index"]
    w = np.bincount(ei[0], weights=np.linalg.norm(g["pos"][ei[0]] - g["pos"][ei[1]], axis=1) / 2, minlength=len(ok))
    w = np.where(ok, w, 0.0)
    return rng.choice(len(w), size=k, replace=True, p=w / w.sum())


def prompt_site_probs(score, allowed, g, prompt, sigma_mm, temperature=1.0, A=None):
    """p(site | prompt) over the graph points: score ** (1/temperature) times a Gaussian of the
    distance along the graph to the prompt, on the allowed points. Returns (probabilities,
    distances to the prompt in mm). Falls back to the kernel alone where the score is zero."""
    from scipy.sparse.csgraph import dijkstra
    from dataset.ostium_dataset import adjacency
    A = adjacency(g["pos"], g["edge_index"]) if A is None else A
    d = dijkstra(A, directed=False, indices=int(prompt))
    k = np.where(np.isfinite(d), np.exp(-0.5 * (d / sigma_mm) ** 2), 0.0)
    w = np.where(allowed, np.maximum(score, 0) ** (1.0 / temperature) * k, 0.0)
    if w.sum() <= 0:
        w = np.where(allowed, k, 0.0)
    if w.sum() <= 0:
        w = (d == 0).astype(float)
    return w / w.sum(), d


def nearest_point(g, xyz):
    """Index of the graph point nearest to a mm position (for a prompt given as coordinates)."""
    return int(np.linalg.norm(g["pos"] - np.asarray(xyz, float), axis=1).argmin())


# ------------------------------------------------------------------------------ generation

@torch.no_grad()
def decode(M, b, centre, y, x, t):
    """Ostium code y, complex x and type t (batch tensors, destandardised) at the batch's site ->
    ring, placed complex mesh and the sensor's dome probability, in the graph's mm frame."""
    T = M.T
    from models.ostium_model import frame_from_code
    dec = M.dec
    idx = b.site
    ring = T.decode_ring(y, b.pos[idx], b.tangent, b.scale)                  # batch units
    Rf = frame_from_code(y[:, :3], b.tangent)
    C = b.pos[idx] + y[:, :3] * b.scale
    phi, logs = dec.phi_from_z(x[:, :16], t)
    V_loc = T.local_vertices(dec.vcan(phi, t), logs, x[:, 16:])
    p_dome, _, _, _ = T.sensor_rim(dec, phi, V_loc, t)
    V = C[:, None] + b.scale[:, None] * torch.einsum("bnc,bcd->bnd", V_loc, Rf)
    to_mm = lambda u: u[0].cpu().numpy().astype(float) * 10.0 + centre
    rec = dec.mc.get(int(t[0]))
    faces = rec.ghd_forward_as_Meshes(torch.zeros(1, 432, device=rec.GHD_eigvec.device),
                                      denormalize_shape=False).faces_packed().cpu().numpy()
    return dict(ring_mm=to_mm(ring), centre_mm=to_mm(C[:, None])[0], frame=Rf[0].cpu().numpy(),
                scale_mm=float(b.scale[0, 0]) * 10.0, atype=int(t[0]), y=y[0].cpu().numpy(), x=x[0].cpu().numpy(),
                verts_mm=to_mm(V), faces=faces, p_dome=p_dome[0].cpu().numpy())


def is_morph(M):
    return hasattr(M.model, "morph_ost")


def morph_reference(path=MORPH_REFERENCE):
    return json.loads(Path(path).read_text())


def sample_morph(rng, ref=None):
    """A requested (volume mm3, aspect ratio) drawn from the real aneurysms (one case's markers)."""
    ref = morph_reference() if ref is None else ref
    r = ref["cases"][int(rng.integers(len(ref["cases"])))]
    return float(r["volume_mm3"]), float(r["aspect_ratio"])


def dome_morph(out):
    """Markers of a generated dome (models/dome_markers.py, sensor p = 0.5 cut), in mm:
    volume_mm3, aspect_ratio (height along the neck normal / mean neck radius), height_mm, neck_radius_mm."""
    from models.dome_markers import DomeMarkers
    V = torch.tensor(out["verts_mm"], dtype=torch.float64)[None]
    p = torch.tensor(out["p_dome"], dtype=torch.float64)[None]
    o = DomeMarkers()(V, p, torch.as_tensor(out["faces"], dtype=torch.long))
    return dict(volume_mm3=float(o["volume"]), aspect_ratio=float(o["aspect_ratio"]), height_mm=float(o["height"]),
                neck_radius_mm=float(o["neck_radius"]))


@torch.no_grad()
def generate_at(M, g, site, rng, ref, sample_type=True, morph=None):
    """One aneurysm at graph point `site`: a training-like crop around it, then ostium -> type ->
    complex -> frozen decoder -> placed mesh -> sensor dome. Everything returned in the graph's
    mm frame. morph = (volume mm3, aspect ratio): the requested dome, required by a
    morphology-conditioned model (ignored otherwise); the output then carries morph_target."""
    from dataset.ostium_dataset import adjacency
    A = adjacency(g["pos"], g["edge_index"])
    sel = training_like_crop(g, A, site, rng, ref)
    gc, remap = subgraph(g, sel)
    sigma = int(rng.choice([-1, 1]))                    # the tangent sign is random in training too
    b, centre = make_batch(gc, rng, M.args, site=int(remap[site]), sigma=sigma)
    b = b.to(M.dev)
    m, ys, xs = M.model, M.ys, M.xs
    enc = m.enc(b)
    idx = b.site
    mk = {}
    if is_morph(M):
        assert morph is not None, "this model is conditioned on the dome morphology: pass morph=(volume_mm3, aspect_ratio)"
        mk = dict(morph=torch.tensor([[np.log(morph[0]), morph[1]]], dtype=torch.float32, device=M.dev))
    y = ys.inv(m.ost.sample(m.ostium_condition(enc, b, idx, **mk)))
    c2 = m.complex_condition(enc, b, ys.fwd(y), y, idx, **mk)
    p_type = torch.softmax(m.type_head(c2), 1)
    t = torch.multinomial(p_type, 1)[:, 0] if sample_type else p_type.argmax(1)
    x = M.xs.inv(m.cx.sample(m.with_type(c2, t)))
    out = decode(M, b, centre, y, x, t)
    out.update(site=int(site), sigma=sigma, crop_points=sel, p_type=p_type[0].cpu().numpy())
    if mk:
        out["morph_target"] = np.array(morph, float)
    return out


@torch.no_grad()
def reference_sample(M, ds, i):
    """The real aneurysm of dataset case i (whole graph, no augmentation): its true ring and its
    true complex decoded through the same frozen VAE decoder, GHD and sensor as a generated one.
    Returns (graph, out) for plausibility()."""
    from torch_geometric.data import Batch
    from dataset.ostium_dataset import FOLLOW_BATCH
    assert not ds.augment
    d = ds.case_data(i)
    b = Batch.from_data_list([ds[i]], follow_batch=FOLLOW_BATCH).to(M.dev)
    out = decode(M, b, d["pos"].mean(0), b.y_ost, b.x_cx, b.atype)
    out.update(site=int(d["site"]), case=d["case"], ds=d["ds"])
    g = dict(pos=d["pos"], radius=d["radius"], edge_index=d["ei"], meta={})
    return g, out


# ------------------------------------------------------------------------------ plausibility

def _segment_hits_triangles(p0, p1, tri):
    """For segments p0->p1 (S, 3) and triangles tri (F, 3, 3): bool (S,) whether the segment
    crosses any triangle (Moller-Trumbore, vectorised)."""
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    d = p1 - p0
    h = np.cross(d[:, None], e2[None])                                   # (S, F, 3)
    a = (e1[None] * h).sum(-1)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p0[:, None] - tri[None, :, 0]
    u = f * (s * h).sum(-1)
    q = np.cross(s, e1[None])
    v = f * (d[:, None] * q).sum(-1)
    t = f * (e2[None] * q).sum(-1)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    return hit.any(1)


def plausibility(g, out, dome_thr=0.5):
    """Checks on one generated sample, against the full graph (mm):
      ring_wall_ratio      ring points: distance to the centerline / local radius
                           (training rings: median 1.67, 10th pct 1.30; < 1 means inside the vessel)
      skeleton_through_ring  centerline edges that cross the ostium disk (a fan on the ring); a
                           neck on the vessel wall has none
      dome_inside_tube_frac  dome vertices closer to the centerline than 0.8 x the local radius
                           (the dome reaching into a vessel)
      dome_height_mm, neck_diameter_mm, aspect_ratio, size_ratio (dome height / parent diameter)
      dome_in_core_frac    dome vertices inside the patch core (if the graph has a core box)"""
    from scipy.spatial import cKDTree
    pos, rad, ei = g["pos"], g["radius"], g["edge_index"]
    tree = cKDTree(pos)
    ring = out["ring_mm"]
    dr, ir = tree.query(ring)
    ratio = dr / np.maximum(rad[ir], 1e-6)
    c = ring.mean(0)
    tri = np.stack([np.repeat(c[None], len(ring), 0), ring, np.roll(ring, -1, 0)], 1)
    und = ei[:, ei[0] < ei[1]]
    hits = _segment_hits_triangles(pos[und[0]], pos[und[1]], tri)
    dome_v = out["p_dome"] > dome_thr
    V = out["verts_mm"]
    res = dict(ring_wall_ratio_min=round(float(ratio.min()), 2), ring_wall_ratio_median=round(float(np.median(ratio)), 2),
               skeleton_through_ring=int(hits.sum()), dome_vertices=int(dome_v.sum()))
    if dome_v.any():
        Vd = V[dome_v]
        dd, idd = tree.query(Vd)
        res["dome_inside_tube_frac"] = round(float((dd < 0.8 * rad[idd]).mean()), 3)
        nrm = out["frame"][2]                                             # ostium normal (outwards)
        h = (Vd - c) @ nrm
        res["dome_height_mm"] = round(float(h.max()), 2)
        neck = float(np.linalg.norm(ring[:, None] - ring[None], axis=2).max())
        res["neck_diameter_mm"] = round(neck, 2)
        res["aspect_ratio"] = round(float(h.max()) / max(neck, 1e-6), 2)
        res["size_ratio"] = round(float(h.max()) / max(2 * rad[out["site"]], 1e-6), 2)
        box = g["meta"].get("core_box_mm") if isinstance(g.get("meta"), dict) else None
        if box is None and "core_box_mm" in g:
            box = g["core_box_mm"]
        if box is not None:
            lo, hi = np.array(box[0]), np.array(box[1])
            res["dome_in_core_frac"] = round(float(((Vd >= lo) & (Vd <= hi)).all(1).mean()), 3)
    res["flags"] = [k for k, bad in (
        ("ring_inside_vessel", res["ring_wall_ratio_min"] < 0.5),
        ("skeleton_through_ostium", res["skeleton_through_ring"] > 0),
        ("dome_into_vessel", res.get("dome_inside_tube_frac", 0) > 0.05),
        ("no_dome", res["dome_vertices"] == 0),
        ("dome_outside_core", res.get("dome_in_core_frac", 1.0) < 0.95)) if bad]
    return res
