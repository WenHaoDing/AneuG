"""Generate temporary processed v2-format checkpoints with synthetic centerlines."""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "smoke_centerlines"


def catmull_rom(control, n_points):
    control = np.asarray(control, dtype=np.float32)
    padded = np.vstack([control[:1], control, control[-1:]])
    per_seg = max(4, int(np.ceil(n_points / (len(control) - 1))))
    pieces = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t = np.linspace(0.0, 1.0, per_seg, endpoint=False, dtype=np.float32)[:, None]
        pieces.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    pts = np.vstack(pieces + [control[-1:]])
    x = np.linspace(0, len(pts) - 1, n_points)
    return np.stack([np.interp(x, np.arange(len(pts)), pts[:, d]) for d in range(3)], axis=1).astype(np.float32)


def branch(start, end, rng, n_points):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    mid = 0.5 * (start + end)
    control = np.vstack([start, 0.5 * (start + mid), mid, 0.5 * (mid + end), end])
    control[1:-1] += rng.normal(0.0, 0.18, control[1:-1].shape).astype(np.float32)
    return catmull_rom(control, n_points)


def centerline(case_idx, rng, n_points):
    root = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    inlet = np.array([0.08 * case_idx, -0.2, -2.5], dtype=np.float32)
    outlets = [
        np.array([-1.5, 0.15, 1.9], dtype=np.float32),
        np.array([1.45, -0.1, 1.85], dtype=np.float32),
    ]
    points = [branch(inlet, root, rng, n_points)]
    points.extend(branch(root, end, rng, n_points) for end in outlets)
    branches = [{"branch_id": i, "group": i, "pts": pts} for i, pts in enumerate(points)]
    ids = np.arange(len(branches), dtype=np.int64)
    return {
        "coordinate_system": "ghd",
        "source_coordinate_system": "world",
        "upstream_id": 0,
        "connected_branch_ids": ids,
        "base_branch_ids": ids,
        "branch_ranking": ids,
        "branch_ids": ids,
        "branch_points": points,
        "branches": branches,
    }


def placeholder_ghd():
    return {
        "phi": np.zeros((144, 3), dtype=np.float32),
        "w_rot": np.zeros(3, dtype=np.float32),
        "log_scale": np.array(0.0, dtype=np.float64),
        "t_vec": np.zeros(3, dtype=np.float32),
    }


def placeholder_affine():
    eye = np.eye(3, dtype=np.float64)
    return {
        "R": eye.copy(),
        "scale": np.array(1.0, dtype=np.float64),
        "s_can": np.array(1.0, dtype=np.float64),
        "R_frame": eye.copy(),
        "neck_pt_world": np.zeros(3, dtype=np.float64),
        "R_equivalent": eye.copy(),
        "s_equivalent": np.array(1.0, dtype=np.float64),
        "t_equivalent": np.zeros(3, dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-cases", type=int, default=16)
    parser.add_argument("--points-per-branch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.num_cases):
        aneurysm_type = i % 3
        path = out_dir / f"smoke_centerline_{i:03d}.npy"
        if path.exists() and not args.overwrite:
            print(f"[skip] {path}")
            continue
        np.save(path, {
            "case": path.stem,
            "version": "v2",
            "aneurysm_type": aneurysm_type,
            "canonical_type": "bifurcated" if aneurysm_type == 0 else "sidewall",
            "ghd": placeholder_ghd(),
            "affine": placeholder_affine(),
            "clipped_centerline": centerline(i, rng, args.points_per_branch),
            "sources": {"synthetic": "dataset/generate_smoke_centerlines.py"},
        }, allow_pickle=True)
        print(f"[ok] {path}")


if __name__ == "__main__":
    main()
