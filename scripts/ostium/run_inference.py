"""Run the ostium model on a folder of vessel graphs (image-model patches) and save what it
generates, with plausibility checks and renders for checking by eye.

Each case folder holds graph.npz (save_point_graph format, optionally with in_core), and
optionally meta.json (core_box_mm, origin_xyz, spacing_xyz) and patch_seg.npy (z, y, x labels:
1 artery, 2 real sac) for the renders. Runs on the CPU by default (never GPU 2).

  python scripts/ostium/run_inference.py --ckpt runtime_train/ostium_model/abl_mlp_crop_norot/epoch_1000.pth \
      --graphs /media/yaplab2/HDD\\ Storage/wenhao/image-generation/debugging/graph1d_aneug_test

Prompted generation (the site is drawn near a prompt, see inference/ostium_inference.py):
  --prompts K          K random prompts per case (core, away from the vessel ends), one sample each
  --prompt_mm X,Y,Z    a given prompt position (nearest graph point; repeat for several)
  --prompt_sigma MM    0 (default): the site is the prompt point itself; > 0: the site is drawn
                       near the prompt (model score x Gaussian of this width along the graph)

Morphology-conditioned models (trained with --morph_cond 1) need the requested dome:
  --volume_mm3 V --aspect_ratio A   for every sample (AR = dome height / mean neck radius)
  otherwise each sample's target is drawn from the real aneurysms (dataset/ostium_morph_reference.json)
The target and the generated dome's markers are stored per sample (morph_target, dome_morph).

Writes <out>/<case>/: site_scores.npy, sample_XX.npz (site, ring, complex mesh, dome probability,
codes), case.json; with --render also mesh views sample_XX.png and overview.png; and
<out>/summary.json, summary.md. Merge into the patch masks with run_merge.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--graphs", required=True, help="folder of case folders, each with graph.npz")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--n_sites", type=int, default=6, help="aneurysms generated per case")
    p.add_argument("--n_crops", type=int, default=32, help="crops averaged for the site score")
    p.add_argument("--temperature", type=float, default=1.0, help="site sampling temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", action="store_true", help="also render the decoded meshes (overview + per sample)")
    p.add_argument("--prompts", type=int, default=0, help="K random prompts per case (replaces --n_sites)")
    p.add_argument("--prompt_mm", nargs="*", default=None, help="prompt positions 'x,y,z' (mm, the graph frame)")
    p.add_argument("--prompt_sigma", type=float, default=0.0, help="0: site = prompt; > 0: mm along the graph")
    p.add_argument("--prompt_min_end_mm", type=float, default=None,
                   help="random prompts keep this distance from the vessel ends (default: training sites' 5th pct)")
    p.add_argument("--volume_mm3", type=float, default=None, help="morphology-conditioned models: requested dome volume")
    p.add_argument("--aspect_ratio", type=float, default=None,
                   help="morphology-conditioned models: requested dome height / mean neck radius")
    a = p.parse_args()
    import os
    if a.device.startswith("cuda"):
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "2"), "set CUDA_VISIBLE_DEVICES (never 2)"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the VAE decoder would otherwise pick up cuda:0 itself
    import torch
    torch.set_num_threads(8)
    from inference import ostium_inference as I
    ref = I.training_reference()
    real = json.loads((REPO / "dataset" / "ostium_plausibility_reference.json").read_text())
    M = I.load_model(a.ckpt, a.device)
    mref = I.morph_reference() if I.is_morph(M) else None
    run = Path(a.ckpt).parent.name
    out = Path(a.out) if a.out else REPO / "runtime_infer" / "ostium" / f"{run}_ep{M.epoch:04d}"
    out.mkdir(parents=True, exist_ok=True)
    folders = sorted(f.parent for f in Path(a.graphs).glob("*/graph.npz"))
    if a.cases:
        folders = [f for f in folders if f.name in a.cases]
    rng = np.random.default_rng(a.seed)
    summary = []
    for f in folders:
        t0 = time.time()
        g = I.load_graph(f / "graph.npz")
        meta = json.loads((f / "meta.json").read_text()) if (f / "meta.json").exists() else {}
        if "core_box_mm" in meta:
            g["meta"]["core_box_mm"] = meta["core_box_mm"]
        co = out / f.name
        co.mkdir(exist_ok=True)
        score, cover, site_crops = I.site_scores(M, g, rng, ref, n_crops=a.n_crops)
        np.save(co / "site_scores.npy", np.stack([score, cover]))
        allowed = g["in_core"] & (cover > 0)
        prompts = [None] * a.n_sites
        if a.prompt_mm:
            prompts = [I.nearest_point(g, [float(v) for v in t.split(",")]) for t in a.prompt_mm]
        elif a.prompts:
            min_end = a.prompt_min_end_mm if a.prompt_min_end_mm is not None else I.prompt_reference()["site_end_distance_mm"][5]
            prompts = list(I.sample_prompts(g, a.prompts, rng, min_end))
        if prompts[0] is None:
            sites = I.sample_sites(score, allowed, a.n_sites, rng, a.temperature)
        else:
            sites = []
            for q in prompts:
                if a.prompt_sigma <= 0:
                    sites.append(int(q))
                    continue
                pr, dq = I.prompt_site_probs(score, allowed, g, q, a.prompt_sigma, a.temperature)
                sites.append(int(rng.choice(len(pr), p=pr)))
        outs, rows = [], []
        for k, s in enumerate(sites):
            morph = None
            if I.is_morph(M):
                rv, ra = I.sample_morph(rng, mref)
                morph = (a.volume_mm3 if a.volume_mm3 is not None else rv, a.aspect_ratio if a.aspect_ratio is not None else ra)
            o = I.generate_at(M, g, int(s), rng, ref, morph=morph)
            if prompts[k] is not None:
                from scipy.sparse.csgraph import dijkstra
                from dataset.ostium_dataset import adjacency
                d = dijkstra(adjacency(g["pos"], g["edge_index"]), directed=False, indices=int(prompts[k]))[int(s)]
                o.update(prompt=int(prompts[k]), prompt_mm=g["pos"][int(prompts[k])], prompt_site_mm=float(d))
            m = I.plausibility(g, o)
            if morph is not None:
                m.update(morph_target=dict(volume_mm3=round(morph[0], 2), aspect_ratio=round(morph[1], 3)),
                         dome_morph={k_: round(v_, 3) for k_, v_ in I.dome_morph(o).items()})
            keep = np.zeros(len(g["pos"]), bool)
            keep[o["crop_points"]] = True
            m["crop"] = I.graph_stats(g["pos"], g["edge_index"], keep)
            np.savez_compressed(co / f"sample_{k:02d}.npz", **{x: v for x, v in o.items() if not isinstance(v, (dict, list))},
                                metrics=json.dumps(m, default=float))
            outs.append(o)
            if "prompt" in o:
                m.update(prompt=o["prompt"], prompt_site_mm=round(o["prompt_site_mm"], 2))
            rows.append(dict(sample=k, site=int(s), site_score=round(float(score[s]), 2), atype=o["atype"],
                             p_type=[round(float(v), 2) for v in o["p_type"]], **m))
        case = dict(case=f.name, graph=I.graph_stats(g["pos"], g["edge_index"]), core_points=int(g["in_core"].sum()),
                    site_crops=dict(median_length_mm=float(np.median([c["length_mm"] for c in site_crops])),
                                    median_ends=float(np.median([c["ends"] for c in site_crops])),
                                    median_junctions=float(np.median([c["junctions"] for c in site_crops]))),
                    samples=rows, seconds=round(time.time() - t0, 1))
        (co / "case.json").write_text(json.dumps(case, indent=1, default=float))
        summary.append(case)
        if a.render:
            from inference import render as Rd
            seg = np.load(f / "patch_seg.npy") if (f / "patch_seg.npy").exists() else None
            org, sp = meta.get("origin_xyz"), meta.get("spacing_xyz")
            artery = Rd.label_surface(seg, 1, org, sp) if seg is not None and org else None
            sac = Rd.label_surface(seg, 2, org, sp) if seg is not None and org else None
            box = meta.get("core_box_mm")
            Rd.overview(co / "overview.png", g, score, outs, artery, sac, box, title=f"{f.name}  ({run}, epoch {M.epoch})")
            for k, (o, r) in enumerate(zip(outs, rows)):
                show = {x: r[x] for x in ("atype", "neck_diameter_mm", "dome_height_mm", "aspect_ratio", "size_ratio",
                                          "ring_wall_ratio_min", "skeleton_through_ring", "dome_inside_tube_frac", "flags") if x in r}
                Rd.sample(co / f"sample_{k:02d}.png", g, o, show, artery, sac, title=f"{f.name}  sample {k}  site {o['site']}")
        print(f"{f.name}: {len(rows)} samples, flags {[r['flags'] for r in rows]}  ({time.time() - t0:.0f}s)", flush=True)
    write_summary(out, summary, real, ref, a, M)


def write_summary(out, summary, real, ref, a, M):
    rows = [r for c in summary for r in c["samples"]]
    pd = [r["prompt_site_mm"] for r in rows if "prompt_site_mm" in r]
    keys = ["neck_diameter_mm", "dome_height_mm", "aspect_ratio", "size_ratio", "ring_wall_ratio_min"]
    gen = dict(n=len(rows), flag_rate={f: float(np.mean([f in r["flags"] for r in rows])) for f in real["flag_rate"]},
               medians={k: float(np.median([r[k] for r in rows if k in r])) for k in keys},
               types={t: int(sum(r["atype"] == t for r in rows)) for t in (0, 1, 2)})
    (out / "summary.json").write_text(json.dumps(dict(args=vars(a), epoch=M.epoch, generated=gen, cases=summary), indent=1, default=float))
    L = [f"# Ostium inference: {Path(a.ckpt).parent.name}, epoch {M.epoch}", "",
         f"{len(summary)} graphs, {len(rows)} generated aneurysms. Real = the same checks on {real['n']} real training aneurysms.", "",
         "| check | generated | real |", "|---|---|---|"]
    for f_, v in gen["flag_rate"].items():
        L.append(f"| {f_} | {v:.0%} | {real['flag_rate'][f_]:.0%} |")
    L += ["", "| median | generated | real |", "|---|---|---|"]
    for k in keys:
        L.append(f"| {k} | {gen['medians'][k]:.2f} | {real['quantiles'][k][50]:.2f} |")
    L += ["", f"types generated (0/1/2): {gen['types'][0]} / {gen['types'][1]} / {gen['types'][2]}", "",
          "| case | length mm | ends | junctions | site crops: median length, ends, junctions | flags per sample |", "|---|---|---|---|---|---|"]
    for c in summary:
        sc = c["site_crops"]
        L.append(f"| {c['case']} | {c['graph']['length_mm']:.0f} | {c['graph']['ends']} | {c['graph']['junctions']} | "
                 f"{sc['median_length_mm']:.0f}, {sc['median_ends']:.0f}, {sc['median_junctions']:.0f} | "
                 + "; ".join(",".join(r["flags"]) or "ok" for r in c["samples"]) + " |")
    mt = [(r["morph_target"], r["dome_morph"]) for r in rows if "morph_target" in r]
    if mt:
        ev = [100 * abs(d["volume_mm3"] / t["volume_mm3"] - 1) for t, d in mt]
        ea = [abs(d["aspect_ratio"] - t["aspect_ratio"]) for t, d in mt]
        L += ["", f"Morphology condition ({len(mt)} samples): volume error median {np.median(ev):.0f}% "
                  f"(90th pct {np.percentile(ev, 90):.0f}%), aspect-ratio error median {np.median(ea):.2f} "
                  f"(90th pct {np.percentile(ea, 90):.2f})"]
    if pd:
        L += ["", f"Prompts: sigma {a.prompt_sigma} mm; site to prompt along the graph (mm): median {np.median(pd):.2f}, "
                  f"90th pct {np.percentile(pd, 90):.2f}, max {np.max(pd):.2f}"]
    L += ["", f"Training crops (median): length {ref['crops']['length_mm'][50]:.0f} mm, ends {ref['crops']['ends'][50]:.0f}, "
              f"junctions {ref['crops']['junctions'][50]:.0f}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
