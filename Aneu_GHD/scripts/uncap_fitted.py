#!/usr/bin/env python3
"""
uncap_fitted.py — Remove vmtk-added caps from a GHD-fitted mesh.

The fitted mesh shares the canonical topology (same vertices/faces order).
Cap boundaries are identified by ring vertex indices stored in the canonical
landmarks.npz.  Faces are separated into body vs cap by blocking face-adjacency
across ring edges and keeping the largest connected component.

Usage:
    python scripts/uncap_fitted.py \\
        --fitted    results/C0002/fitted_mesh.obj \\
        --landmarks /path/to/canonical/landmarks.npz \\
        --out       results/C0002/fitted_mesh_uncapped.obj

    # or let it auto-name next to the input:
    python scripts/uncap_fitted.py \\
        --fitted    results/C0002/fitted_mesh.obj \\
        --landmarks /path/to/canonical/landmarks.npz
        
    cd "/media/yaplab2/HDD Storage/almaha" && /home/yaplab2/miniconda3/envs/ghd/bin/python3 Aneu_GHD/scripts/uncap_fitted.py \
        --fitted    "Aneu_GHD/fitting_results/AnueX/_testE/C0005_cut2/ghd_fitted.obj" \
        --landmarks "Aneu_GHD/canonical/Sidewall/landmarks.npz"

    cd "/media/yaplab2/HDD Storage/almaha" && \
        /home/yaplab2/miniconda3/envs/ghd/bin/python3 Aneu_GHD/scripts/uncap_fitted.py \
            --type bifurcation \
            --fitted    "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/fitting_results/AnueX/C0011_cut2/ghd_fitted.obj" \
            --canonical-rings "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/canonical/Bifricated/canonical_typeB/opa_checkpoint.pkl" \
            
    

"""

import argparse
import os

import numpy as np
import trimesh
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components


def uncap_mesh_bif(mesh: trimesh.Trimesh,
                   cap_interior_vertex_sets: list) -> trimesh.Trimesh:
    """
    Remove cap faces from a bifurcation fitted mesh using op_rec_v_indices_map.

    Step 1: remove any face containing a cap-interior vertex (catches most cap faces).
    Step 2: drop any remaining small disconnected components — these are leftover
            cap boundary faces that contain only ring-edge vertices and slipped
            through step 1.

    Parameters
    ----------
    cap_interior_vertex_sets : list of array-like
        One array per opening — canonical vertex indices of cap-interior verts.
        Pass opa['op_rec_v_indices_map'] directly.
    """
    faces = mesh.faces  # (F, 3)

    # Step 1: remove faces containing cap-interior vertices
    cap_verts = set()
    for verts in cap_interior_vertex_sets:
        cap_verts.update(np.array(verts).tolist())

    keep_mask = ~np.array([any(v in cap_verts for v in f) for f in faces])
    n_removed_step1 = (~keep_mask).sum()
    print(f"  Step 1 — cap-interior faces removed : {n_removed_step1}")

    partial = trimesh.Trimesh(vertices=mesh.vertices, faces=faces[keep_mask], process=False)

    # Step 2: drop small disconnected components (leftover cap boundary faces)
    adj = lil_matrix((len(partial.faces), len(partial.faces)), dtype=np.int8)
    for f1, f2 in partial.face_adjacency:
        adj[f1, f2] = 1
        adj[f2, f1] = 1
    n_comp, labels = connected_components(adj.tocsr(), directed=False)
    counts     = np.bincount(labels)
    main_label = int(np.argmax(counts))

    print(f"  Step 2 — components after step 1: {n_comp}")
    for c in range(n_comp):
        marker = " ← kept (body)" if c == main_label else " ← removed (fragment)"
        print(f"    component {c}: {counts[c]} faces{marker}")

    keep_faces = partial.faces[labels == main_label]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=keep_faces, process=False)


def uncap_mesh(mesh: trimesh.Trimesh,
               ring_index_sets: list) -> trimesh.Trimesh:
    """
    Remove cap faces from a fitted mesh using ring boundary vertex indices.

    Blocks face-adjacency across ring edges (edges whose both endpoints are
    ring vertices), then takes the largest connected component (vessel body).
    Works on the fitted mesh directly — no cutting planes needed since vertex
    positions have moved out of canonical space after GHD deformation.

    Parameters
    ----------
    ring_index_sets : list of np.ndarray
        Ring vertex indices for each opening (2 for sidewall, 3 for bifurcation).
    """
    faces     = mesh.faces
    adj_pairs = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    ring_verts = set().union(*(set(r.tolist()) for r in ring_index_sets))

    n_faces = len(faces)
    adj = lil_matrix((n_faces, n_faces), dtype=np.int8)
    for (f1, f2), (v1, v2) in zip(adj_pairs, adj_edges):
        if v1 in ring_verts and v2 in ring_verts:
            continue          # ring edge — acts as a cut boundary
        adj[f1, f2] = 1
        adj[f2, f1] = 1

    n_comp, labels = connected_components(adj.tocsr(), directed=False)
    counts     = np.bincount(labels)
    main_label = int(np.argmax(counts))

    print(f"  Connected components: {n_comp}")
    for c in range(n_comp):
        marker = " ← kept (body)" if c == main_label else " ← removed (cap)"
        print(f"    component {c}: {counts[c]} faces{marker}")

    keep_faces = faces[labels == main_label]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=keep_faces, process=False)


def main():
    parser = argparse.ArgumentParser(
        description="Remove vmtk caps from a GHD-fitted mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fitted",          required=True, help="Path to fitted_mesh.obj")
    parser.add_argument("--type",            default="sidewall", choices=["sidewall", "bifurcation"],
                                                                help="Aneurysm type")
    parser.add_argument("--landmarks",       default=None,  help="[sidewall] Path to canonical landmarks.npz")
    parser.add_argument("--canonical-rings", default=None,  help="[bifurcation] Path to canonical opa_checkpoint.pkl")
    parser.add_argument("--out",             default=None,  help="Output path (default: <fitted>_uncapped.obj)")
    args = parser.parse_args()

    if args.out is None:
        base, ext = os.path.splitext(args.fitted)
        args.out = base + "_uncapped" + ext

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading fitted mesh : {args.fitted}")
    mesh = trimesh.load(args.fitted, process=False)
    print(f"  {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

    if args.type == "sidewall":
        print(f"Loading landmarks   : {args.landmarks}")
        lm = np.load(args.landmarks, allow_pickle=True)
        ring_index_sets = [lm["ring_up_idxs"], lm["ring_dn_idxs"]]
        print(f"  ring_up_idxs: {len(ring_index_sets[0])}  ring_dn_idxs: {len(ring_index_sets[1])}")
    else:
        import pickle
        print(f"Loading canonical rings : {args.canonical_rings}")
        with open(args.canonical_rings, "rb") as f:
            opa = pickle.load(f)
        cap_interior_vertex_sets = [np.array(opa["op_rec_v_indices_map"][i]) for i in range(3)]
        for i, c in enumerate(cap_interior_vertex_sets):
            print(f"  ring {i}: {len(c)} cap-interior vertices")

    # ── Uncap ────────────────────────────────────────────────────────────────
    print("Removing caps...")
    if args.type == "bifurcation":
        uncapped = uncap_mesh_bif(mesh, cap_interior_vertex_sets)
    else:
        uncapped = uncap_mesh(mesh, ring_index_sets)
    print(f"  Result: {uncapped.vertices.shape[0]} verts, {uncapped.faces.shape[0]} faces")

    # ── Save ─────────────────────────────────────────────────────────────────
    uncapped.export(args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()