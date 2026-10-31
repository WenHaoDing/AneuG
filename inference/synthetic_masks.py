"""Synthetic aneurysm masks: a healthy vessel patch in, the patch's label map with a generated
aneurysm out. The ostium model generates an aneurysm on the patch's 1D vessel graph
(inference/ostium_inference.py), and inference/merge.py attaches its dome to the artery and
voxelises it on the patch grid. Only samples that pass the merge checks are returned.

  from inference.synthetic_masks import AneurysmMaskGenerator, load_patch
  gen = AneurysmMaskGenerator("runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth")
  patch = load_patch(".../graph1d_aneug_test/1KPkS2Uocy_healthy")    # or build the dict yourself
  masks = gen.generate(**patch, n=4)                                   # sites chosen by the model
  masks = gen.generate(**patch, n=1, site="prompt", prompt_mm=(x, y, z))
  masks[0]["seg"]       # (z, y, x) uint8: 0 background, 1 artery, 2 generated aneurysm

A patch is
  graph     path to a point-graph .npz, or a dict with pos (N, 3) mm, radius (N,) mm, edge_index (2, E)
  seg       (z, y, x) labels of the patch, 1 = artery; any label 2 (a real aneurysm) is removed first
  origin, spacing   (x, y, z) mm: the frame mm = origin + (x, y, z) * spacing, no direction matrix
  core      (start_zyx, size) or None: the region the aneurysm must lie in (the image model's core;
            size an int or per axis). None = the whole patch. Sites are only placed on graph
            points inside it, and label 2 is only written inside it.

Sites (site=):
  "model"   drawn from the model's site score over the graph points in the core (temperature)
  "prompt"  at the graph point nearest prompt_mm (x, y, z mm) or prompt_vox (z, y, x index); with
            prompt_sigma > 0 mm, drawn from the model's score near it (sigma along the graph)
  "random"  a uniformly random graph point in the core, away from the vessel ends

Morphology: a checkpoint trained with --morph_cond 1 is conditioned on the requested dome volume
(mm3) and aspect ratio (dome height / mean neck radius); whichever is not given is drawn from the
real aneurysms. Other checkpoints refuse them.
"""
import json
from pathlib import Path

import numpy as np

from inference import merge as Mg
from inference import ostium_inference as I


def load_patch(folder):
    """A patch folder as the image-generation repo exports it (graph.npz, patch_seg.npy, meta.json)
    -> the keyword arguments of AneurysmMaskGenerator.generate."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    core = (meta["core_start_zyx"], meta["core_size"]) if "core_start_zyx" in meta else None
    return dict(graph=folder / "graph.npz", seg=np.load(folder / "patch_seg.npy"), origin=meta["origin_xyz"],
                spacing=meta["spacing_xyz"], core=core)


class AneurysmMaskGenerator:
    def __init__(self, ckpt, device="cpu", n_crops=32, temperature=1.0):
        """ckpt: an ostium-model checkpoint (runtime_train/ostium_model/<run>/epoch_XXXX.pth). device
        "cpu" or "cuda" (set CUDA_VISIBLE_DEVICES first; never GPU 2). n_crops: crops averaged for
        the site score; temperature: site sampling temperature."""
        self.M = I.load_model(str(ckpt), device)
        self.ref = I.training_reference()
        self.morph_ref = I.morph_reference() if I.is_morph(self.M) else None
        self.n_crops, self.temperature = n_crops, temperature
        self.ckpt, self.epoch = str(ckpt), self.M.epoch
        self.attempts = []                      # merge report of every attempt of the last generate()

    @property
    def morph_conditioned(self):
        return self.morph_ref is not None

    def generate(self, graph, seg, origin, spacing, core=None, n=1, site="model", prompt_mm=None, prompt_vox=None,
                 prompt_sigma=0.0, volume_mm3=None, aspect_ratio=None, seed=0, max_attempts=None):
        """Up to n masks (dicts, see below), from at most max_attempts samples (default 4 n): a
        sample is dropped when the merge cannot attach its dome or rejects it. Each mask:
          seg            (z, y, x) uint8, 0 / 1 / 2 (2 = generated aneurysm, inside the core)
          site_mm        the ostium centre (x, y, z mm); site the graph point
          atype          aneurysm type (0 / 1 / 2)
          sac_voxels, sac_volume_mm3
          morph_target, dome_morph   requested and generated (volume_mm3, aspect_ratio), conditioned models
          report         the merge report (flags, checks)"""
        if not self.morph_conditioned and (volume_mm3 is not None or aspect_ratio is not None):
            raise ValueError(f"{self.ckpt} is not conditioned on the dome morphology: volume_mm3 / aspect_ratio "
                             "need a checkpoint trained with --morph_cond 1")
        if site not in ("model", "prompt", "random"):
            raise ValueError(f"site must be 'model', 'prompt' or 'random', not {site!r}")
        rng = np.random.default_rng(seed)
        origin, spacing = np.asarray(origin, float), np.asarray(spacing, float)
        g = I.load_graph(graph) if isinstance(graph, (str, Path)) else dict(
            pos=np.asarray(graph["pos"], float), radius=np.asarray(graph["radius"], float),
            edge_index=np.asarray(graph["edge_index"], int), meta={})
        seg = np.where(np.asarray(seg) == 2, 0, np.asarray(seg)).astype(np.uint8)
        box = None
        if core is not None:
            start, size = np.asarray(core[0], int), np.broadcast_to(np.asarray(core[1], int), (3,))
            box = [(int(start[k]), int(start[k] + size[k])) for k in range(3)]
            zyx = Mg.mm_to_index(g["pos"], origin, spacing)
            g["in_core"] = np.all([(zyx[:, k] >= box[k][0]) & (zyx[:, k] <= box[k][1] - 1) for k in range(3)], 0)
            g["meta"]["core_box_mm"] = [Mg.index_to_mm(np.array([b[0] for b in box]), origin, spacing).tolist(),
                                        Mg.index_to_mm(np.array([b[1] - 1 for b in box]), origin, spacing).tolist()]
        else:
            g["in_core"] = np.ones(len(g["pos"]), bool)
        if not g["in_core"].any():
            raise ValueError("no graph point inside the core")
        score = allowed = None
        if site == "model" or (site == "prompt" and prompt_sigma > 0):
            score, cover, _ = I.site_scores(self.M, g, rng, self.ref, n_crops=self.n_crops)
            allowed = g["in_core"] & (cover > 0)
        if site == "prompt":
            if (prompt_mm is None) == (prompt_vox is None):
                raise ValueError("site='prompt' needs exactly one of prompt_mm (x, y, z) or prompt_vox (z, y, x)")
            p = np.asarray(prompt_mm, float) if prompt_mm is not None else Mg.index_to_mm(np.asarray(prompt_vox, float), origin, spacing)
            q = I.nearest_point(g, p)
            if not g["in_core"][q]:
                raise ValueError(f"the graph point nearest the prompt ({np.round(g['pos'][q], 2).tolist()} mm) is outside the core")
        self.attempts, out = [], []
        for k in range(max_attempts or 4 * n):
            if len(out) >= n:
                break
            if site == "model":
                s = int(I.sample_sites(score, allowed, 1, rng, self.temperature)[0])
            elif site == "random":
                s = int(I.sample_prompts(g, 1, rng, I.prompt_reference()["site_end_distance_mm"][5])[0])
            elif prompt_sigma > 0:
                pr, _ = I.prompt_site_probs(score, allowed, g, q, prompt_sigma, self.temperature)
                s = int(rng.choice(len(pr), p=pr))
            else:
                s = int(q)
            morph = None
            if self.morph_conditioned:
                rv, ra = I.sample_morph(rng, self.morph_ref)
                morph = (volume_mm3 if volume_mm3 is not None else rv, aspect_ratio if aspect_ratio is not None else ra)
            o = I.generate_at(self.M, g, s, rng, self.ref, morph=morph)
            new, rep = Mg.merge_sample(seg, origin, spacing, o, g, core=box)
            ok = new is not None and not rep.get("rejected")
            self.attempts.append(dict(attempt=k, site=s, accepted=ok, flags=rep["flags"], rejected=rep.get("rejected", [])))
            if not ok:
                continue
            sac = new == 2
            m = dict(seg=new.astype(np.uint8), site=s, site_mm=o["ring_mm"].mean(0).tolist(), atype=int(o["atype"]),
                     sac_voxels=int(sac.sum()), sac_volume_mm3=float(sac.sum() * np.prod(spacing)), report=rep)
            if morph is not None:
                dm = I.dome_morph(o)
                m.update(morph_target=dict(volume_mm3=float(morph[0]), aspect_ratio=float(morph[1])),
                         dome_morph=dict(volume_mm3=float(dm["volume_mm3"]), aspect_ratio=float(dm["aspect_ratio"])))
            out.append(m)
        return out
