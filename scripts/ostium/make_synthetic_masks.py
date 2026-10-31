"""Synthetic aneurysm masks for vessel patches (command line for inference/synthetic_masks.py).

For each patch folder (graph.npz, patch_seg.npy, meta.json, as the image-generation repo exports
them), generate n aneurysms with the given ostium-model checkpoint and write the accepted ones to
--out:
  <out>/<patch>_<seed>_<k>.npz   seg: (z, y, x) uint8, 0 background, 1 artery, 2 generated aneurysm
  <out>/<patch>_<seed>_<k>.png   with --preview: voxel and surface render around the aneurysm
  <out>/masks.json               per mask: patch, frame, core, site, type, sac size, morphology,
                                 checkpoint, seed; updated (not overwritten) by later runs

  python scripts/ostium/make_synthetic_masks.py --ckpt <checkpoint.pth> --patches <patch folder or folder of them> \
      --out <folder> [--n 4] [--site model|prompt|random] [--prompt_mm X,Y,Z | --prompt_vox Z,Y,X] \
      [--prompt_sigma MM] [--volume_mm3 V] [--aspect_ratio A] [--seed S] [--preview] [--device cpu]

--site model: sites drawn from the model's site score (default); prompt: at the given point (one
patch at a time; coordinates are the patch's mm frame or voxel index); random: uniform along the
vessel in the core. --volume_mm3 / --aspect_ratio (dome height / mean neck radius) need a
morphology-conditioned checkpoint (trained with --morph_cond 1); with such a checkpoint, the one not
given is drawn from the real aneurysms. CPU by default; for a GPU set CUDA_VISIBLE_DEVICES (never 2).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def triple(s):
    return [float(v) for v in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="ostium-model checkpoint (.pth)")
    p.add_argument("--patches", required=True, nargs="+", help="patch folders, or folders of patch folders")
    p.add_argument("--out", required=True, help="where the masks are written")
    p.add_argument("--n", type=int, default=4, help="accepted masks wanted per patch")
    p.add_argument("--max_attempts", type=int, default=None, help="samples tried per patch (default 4 x n)")
    p.add_argument("--site", choices=["model", "prompt", "random"], default="model")
    p.add_argument("--prompt_mm", type=triple, default=None, help="x,y,z in the patch's mm frame")
    p.add_argument("--prompt_vox", type=triple, default=None, help="z,y,x voxel index in patch_seg")
    p.add_argument("--prompt_sigma", type=float, default=0.0, help="0: site at the prompt; > 0: drawn near it (mm)")
    p.add_argument("--volume_mm3", type=float, default=None)
    p.add_argument("--aspect_ratio", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0, help="site sampling temperature (--site model)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.device.startswith("cuda"):
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "2"), "set CUDA_VISIBLE_DEVICES (never 2)"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    folders = []
    for f in map(Path, a.patches):
        folders += [f] if (f / "graph.npz").exists() else sorted(d.parent for d in f.glob("*/graph.npz"))
    if not folders:
        sys.exit("no patch folders (with graph.npz) found")
    if a.site == "prompt" and len(folders) > 1:
        sys.exit("--site prompt takes one patch: the prompt is in that patch's frame")
    from inference.synthetic_masks import AneurysmMaskGenerator, load_patch
    gen = AneurysmMaskGenerator(a.ckpt, a.device, temperature=a.temperature)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "masks.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for f in folders:
        patch = load_patch(f)
        meta = json.loads((f / "meta.json").read_text())
        try:
            masks = gen.generate(**patch, n=a.n, site=a.site, prompt_mm=a.prompt_mm, prompt_vox=a.prompt_vox,
                                 prompt_sigma=a.prompt_sigma, volume_mm3=a.volume_mm3, aspect_ratio=a.aspect_ratio,
                                 seed=a.seed, max_attempts=a.max_attempts)
        except ValueError as e:
            sys.exit(f"{f.name}: {e}")
        for k, m in enumerate(masks):
            mid = f"{f.name}_{a.seed}_{k:02d}"
            np.savez_compressed(out / f"{mid}.npz", seg=m["seg"])
            index[mid] = dict(file=f"{mid}.npz", patch=f.name, patch_dir=str(f.resolve()), case=meta.get("case"),
                              origin_xyz=patch["origin"], spacing_xyz=patch["spacing"],
                              patch_start_zyx_in_scan=meta.get("patch_start_zyx_in_scan"),
                              core_start_zyx=meta.get("core_start_zyx"), core_size=meta.get("core_size"),
                              site=a.site, site_mm=[round(v, 3) for v in m["site_mm"]], atype=m["atype"],
                              sac_voxels=m["sac_voxels"], sac_volume_mm3=round(m["sac_volume_mm3"], 2),
                              morph_target=m.get("morph_target"), dome_morph=m.get("dome_morph"),
                              merge_flags=m["report"]["flags"], ckpt=str(a.ckpt), epoch=gen.epoch, seed=a.seed)
            if a.preview:
                from inference import render as Rd
                seg0 = np.load(f / "patch_seg.npy")
                Rd.voxels(out / f"{mid}.png", m["seg"], patch["origin"], patch["spacing"], np.array(m["site_mm"]),
                          title=f"{mid}  type {m['atype']}  sac {m['sac_volume_mm3']:.1f} mm3",
                          original=seg0 if (seg0 == 2).any() else None)
        n_try = len(gen.attempts)
        print(f"{f.name}: {len(masks)} / {a.n} masks from {n_try} samples"
              + (f"; dropped: {[x['flags'] for x in gen.attempts if not x['accepted']]}" if n_try > len(masks) else ""),
              flush=True)
        index_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
