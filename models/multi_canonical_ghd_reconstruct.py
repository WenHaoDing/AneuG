from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.ghd_reconstruct import GHD_Reconstruct
from utils.utils import safe_load_mesh


@dataclass(frozen=True)
class CanonicalSpec:
    root: str
    mesh_name: str = "mesh.obj"
    eigen_name: str = "eigenvectors.npy"
    num_basis: int = 12 ** 2

    @property
    def mesh_path(self):
        root = Path(self.root)
        mesh_path = root / self.mesh_name
        if mesh_path.exists():
            return mesh_path
        return root / "part_aligned.obj"

    @property
    def eigen_path(self):
        return Path(self.root) / self.eigen_name


class MultiCanonicalGHDReconstruct:
    """
    Lazily manage one GHD_Reconstruct per aneurysm type.

    Type 0 uses the bifurcated canonical.
    Types 1 and 2 use the sidewall canonical.
    """

    def __init__(self, canonical_root, specs=None, device=None):
        canonical_root = Path(canonical_root)
        self.specs = {
            0: CanonicalSpec(str(canonical_root / "Bifurcated")),
            1: CanonicalSpec(str(canonical_root / "Sidewall")),
            2: CanonicalSpec(str(canonical_root / "Sidewall")),
        }
        if specs is not None:
            self.specs.update(specs)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.reconstructors = {}
        self._edge_cache    = {}   # type_id → edge_index [2, E]
        self._opening_cache = {}   # type_id → list of long tensors [K_i]
        self._trimmed_faces_cache = {}   # type_id → trimmed faces [F', 3] in canonical vertex indexing

    def get(self, aneurysm_type) -> GHD_Reconstruct:
        aneurysm_type = int(aneurysm_type)
        if aneurysm_type not in self.reconstructors:
            spec = self.specs.get(aneurysm_type)
            if spec is None:
                raise ValueError(f"No canonical spec configured for aneurysm_type={aneurysm_type}")
            self.reconstructors[aneurysm_type] = self._build(spec)
        return self.reconstructors[aneurysm_type]

    def _build(self, spec):
        mesh = safe_load_mesh(str(spec.mesh_path))
        recon = GHD_Reconstruct(
            mesh,
            str(spec.eigen_path),
            num_Basis=spec.num_basis,
            device=self.device,
        )
        recon.canonical_Meshes = recon.canonical_Meshes.to(self.device)
        recon.GHD_eigvec = recon.GHD_eigvec.to(self.device)
        return recon

    def _get_edge_index(self, atype):
        """Build and cache undirected edge_index [2, E] for the given type's canonical mesh."""
        if atype not in self._edge_cache:
            faces = self.get(atype).canonical_Meshes.faces_packed()  # [F, 3]
            e01, e12, e02 = faces[:, :2], faces[:, 1:], faces[:, [0, 2]]
            edges = torch.cat([e01, e12, e02, e01.flip(1), e12.flip(1), e02.flip(1)], dim=0)
            self._edge_cache[atype] = torch.unique(edges.t().contiguous(), dim=1)
        return self._edge_cache[atype]

    def to_pyg_batch(self, phi, aneurysm_types, point_std=None):
        """Reconstruct meshes from GHD coefficients and return a PyG Batch.

        Args:
            phi:            float32 [B, num_coeffs, 3]
            aneurysm_types: int64   [B]
            point_std:      float32 [1, 3] or None  — dataset branch point_std.
                            Vertices are divided by point_std after denormalization
                            so that the mesh coordinate scale matches the normalized
                            local_points the model is trained to predict.

        Returns:
            torch_geometric.data.Batch  — B graphs, each with:
                x:          [N_t, 3]   reconstructed vertex positions
                edge_index: [2, E_t]   mesh edges (undirected, cached per type)
        """
        from torch_geometric.data import Data, Batch

        B = phi.size(0)
        types = aneurysm_types.tolist()
        data_list = [None] * B

        for atype in set(int(t) for t in types):
            idx = [b for b in range(B) if int(types[b]) == atype]
            recon     = self.get(atype)
            phi_sub   = phi[idx]                                                  # [K, num_coeffs, 3]
            # eigvec [N, num_coeffs]; einsum → offset [K, N, 3]
            # * norm_canonical: bring vertices into physical GHD space (same as start_points)
            # / point_std: match the scale of normalized local_points the model predicts
            offset    = torch.einsum('nm,bmc->bnc', recon.GHD_eigvec, phi_sub)
            verts_all = (recon.canonical_Meshes.verts_packed().unsqueeze(0) + offset) * recon.norm_canonical  # [K, N, 3]
            if point_std is not None:
                verts_all = verts_all / point_std.to(verts_all.device)           # [K, N, 3]
            edge_index = self._get_edge_index(atype)
            for k, b in enumerate(idx):
                data_list[b] = Data(x=verts_all[k], edge_index=edge_index)

        return Batch.from_data_list(data_list)

    def _load_openings(self, atype):
        """Load and cache opening vertex indices for the given type from openings.npz."""
        if atype not in self._opening_cache:
            path = Path(self.specs[atype].root) / "openings.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"openings.npz not found at {path}. "
                    "Run dataset/record_openings.ipynb first."
                )
            data = np.load(path)
            n    = int(data["num_openings"])
            self._opening_cache[atype] = [
                torch.from_numpy(data[f"indices_{i}"]).long().to(self.device)
                for i in range(n)
            ]
        return self._opening_cache[atype]

    def compute_branch_conditions(self, phi, atype, max_branches=None):
        """Compute start_points and directions for a same-type batch of phi.

        All samples in phi must belong to the same aneurysm type.

        Args:
            phi:          float32 [B, num_coeffs, 3]
            atype:        int — single aneurysm type shared by all samples
            max_branches: int or None — output width; auto = num openings for this type

        Returns:
            start_points:      float32 [B, max_branches, 3]
            branch_directions: float32 [B, max_branches, 3]  unit vectors
            branch_mask:       bool    [B, max_branches]
        """
        B               = phi.size(0)
        atype           = int(atype)
        recon           = self.get(atype)
        opening_indices = self._load_openings(atype)
        n_open          = len(opening_indices)
        if max_branches is None:
            max_branches = n_open

        starts    = phi.new_zeros(B, max_branches, 3)
        dirs      = phi.new_zeros(B, max_branches, 3)
        mask      = torch.zeros(B, max_branches, dtype=torch.bool, device=phi.device)

        offset    = torch.einsum('nm,bmc->bnc', recon.GHD_eigvec, phi)
        verts_all = (recon.canonical_Meshes.verts_packed().unsqueeze(0)
                     + offset) * recon.norm_canonical                             # [B, N, 3]
        mesh_ctr  = verts_all.mean(1)                                            # [B, 3]

        for o in range(min(n_open, max_branches)):
            bv        = verts_all[:, opening_indices[o], :]                      # [B, K_o, 3]
            centroid  = bv.mean(1)                                               # [B, 3]
            _, _, Vt  = torch.linalg.svd(bv - centroid.unsqueeze(1), full_matrices=False)
            normal    = Vt[:, -1, :]                                             # [B, 3]
            flip      = (normal * F.normalize(centroid - mesh_ctr, dim=-1)).sum(-1) < 0
            normal    = torch.where(flip.unsqueeze(-1), -normal, normal)
            starts[:, o] = centroid
            dirs[:, o]   = F.normalize(normal, dim=-1)
            mask[:, o]   = True

        return starts, dirs, mask

    def compute_branch_directions(self, phi, aneurysm_types):
        """SVD outward normals for a mixed-type batch. Loops per type externally."""
        B        = phi.size(0)
        types    = aneurysm_types.tolist()
        unique   = sorted(set(int(t) for t in types))
        max_open = max(len(self._load_openings(t)) for t in unique)
        dirs     = phi.new_zeros(B, max_open, 3)
        for atype in unique:
            idx      = [b for b in range(B) if int(types[b]) == atype]
            _, d, _  = self.compute_branch_conditions(phi[idx], atype, max_branches=max_open)
            for k, b in enumerate(idx):
                dirs[b] = d[k]
        return dirs

    def _reconstruct_verts_np(self, phi, atype):
        """Deformed canonical vertices in physical GHD space (× norm_canonical).

        Matches the space of compute_branch_conditions / the generated branch
        points: (canonical + eigvec @ phi) * norm_canonical. phi: [num_coeffs, 3].
        """
        recon = self.get(atype)
        if torch.is_tensor(phi):
            phi_t = phi.detach().to(self.device, dtype=torch.float32)
        else:
            phi_t = torch.as_tensor(np.asarray(phi), dtype=torch.float32, device=self.device)
        offset = torch.einsum('nm,mc->nc', recon.GHD_eigvec, phi_t)
        verts = (recon.canonical_Meshes.verts_packed() + offset) * recon.norm_canonical
        return verts.detach().cpu().numpy()

    def _trimmed_faces(self, atype):
        """Faces of mesh_trimmed.obj remapped to canonical vertex indices (cached).

        mesh_trimmed.obj shares exact vertex positions with mesh.obj (== the
        canonical mesh), so we match trimmed verts to canonical verts by position.
        Applying these faces to the deformed verts gives the uncapped shape.
        """
        if atype not in self._trimmed_faces_cache:
            import trimesh
            from scipy.spatial import cKDTree
            recon = self.get(atype)
            full_phys = (recon.canonical_Meshes.verts_packed() * recon.norm_canonical).detach().cpu().numpy()
            trim = trimesh.load(Path(self.specs[atype].root) / "mesh_trimmed.obj", process=False)
            tverts = np.asarray(trim.vertices)
            dist, idx = cKDTree(full_phys).query(tverts)
            scale = float(np.linalg.norm(full_phys.max(0) - full_phys.min(0)))
            if dist.max() > 1e-4 * scale:
                raise ValueError(
                    f"mesh_trimmed.obj verts do not match canonical verts for type {atype} "
                    f"(max NN dist {dist.max():.3e}); cannot map trimmed faces."
                )
            self._trimmed_faces_cache[atype] = idx[np.asarray(trim.faces)].astype(np.int64)
        return self._trimmed_faces_cache[atype]

    @staticmethod
    def _arc_length(pts):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def reconstruct_fused_mesh(self, phi, aneurysm_type, branch_points, branch_mask,
                               extrude_length=3.0, min_branch_arc=3.0,
                               init_step=1, max_cl_length=None, ds_r=3,
                               min_torsion=True, flaw_opening_min_size=10,
                               smooth=True, smooth_n_rings=3, smooth_n_iter=10, smooth_lam=0.5,
                               planarize_n_iter=10, planarize_lam=0.5,
                               save_path=None):
        """Fuse a GHD aneurysm dome with tubular branch extensions.

        Reconstructs the mesh from `phi`, uncaps it using the canonical trimmed
        topology, and at each opening either sweeps a tube along the generated
        centerline (`branch_points[o]`) or — if the branch is invalid / too short
        — extrudes the cross-section straight out by `extrude_length` along the
        opening's outward normal. Opening o corresponds to branch o.

        Args:
            phi:           [num_coeffs, 3] GHD coefficients (single sample).
            aneurysm_type: int.
            branch_points: list (len ≤ n_openings) of [N, 3] absolute generated
                           centerlines in physical GHD space (the eval's
                           `abs_branches` output). Missing entries → extrusion.
            branch_mask:   bool [n_branches] — generated branch validity.
            extrude_length: stub length for short/invalid branches (recon units, ≈mm).
            min_branch_arc: arc-length below which a branch is treated as short.
            init_step:     centerline points skipped at the opening end (valid branches only).
            max_cl_length: None, scalar, or per-branch list — truncate centerline arc-length.
            ds_r:          ring downsample rate along the tube (passed to get_tubular_mesh_faces).
            smooth:        if True, smooth the seam and planarize tube-end openings.

        Returns:
            pyvista.PolyData — the merged (and optionally smoothed) mesh.
        """
        import pyvista as pv
        from utils.mesh_fusion import (
            get_cpcd_tangent, get_tubular_l2w_trans, get_tubular_mesh_verts,
            get_tubular_mesh_faces, merge_meshes, resample_branch, avg_edge_length,
            planarize_openings, smooth_near_openings, remove_orphan_vertices,
        )

        atype = int(aneurysm_type)
        verts = self._reconstruct_verts_np(phi, atype)                       # [N, 3]
        mesh_ctr = verts.mean(0)
        opening_indices = [idx.detach().cpu().numpy() for idx in self._load_openings(atype)]
        n_open = len(opening_indices)

        # Uncapped dome: trimmed faces applied to the deformed verts.
        tri = self._trimmed_faces(atype)
        uncapped = pv.PolyData(verts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
        step = avg_edge_length(uncapped)

        if branch_mask is not None:
            branch_mask = np.asarray(branch_mask).astype(bool).ravel()
        if max_cl_length is not None and not isinstance(max_cl_length, (list, tuple)):
            max_cl_length = [max_cl_length] * n_open
        vid_list = list(opening_indices)

        def _outward_normal(ring, centroid):
            _, _, Vt = np.linalg.svd(ring - centroid, full_matrices=False)
            normal = Vt[-1]
            if np.dot(normal, centroid - mesh_ctr) < 0:
                normal = -normal
            return normal / np.linalg.norm(normal)

        # Pass 1: per-opening centerline source, outward normal and validity.
        # A branch absent from the presence head is simply not supplied in
        # branch_points (entry None) -> it falls through to a short straight
        # extrusion stub along the opening's outward normal.
        specs = []
        for o in range(n_open):
            ring0 = verts[opening_indices[o]]
            centroid0 = ring0.mean(0)

            bpts = None
            if branch_points is not None and o < len(branch_points) and branch_points[o] is not None:
                bp = branch_points[o]
                bpts = bp.detach().cpu().numpy() if torch.is_tensor(bp) else np.asarray(bp, dtype=float)
            valid = (
                bpts is not None and len(bpts) >= 2
                and (branch_mask is None or (o < len(branch_mask) and branch_mask[o]))
                and self._arc_length(bpts) >= min_branch_arc
            )

            if valid:
                cl  = resample_branch(bpts, step)
                tan = get_cpcd_tangent(cl)
                cl, tan = cl[init_step:], tan[init_step:]
                if max_cl_length is not None:
                    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))])
                    cut = int(np.searchsorted(arc, max_cl_length[o], side='right'))
                    cl, tan = cl[:max(cut, 2)], tan[:max(cut, 2)]
                normal0 = tan[0]                                  # outward dir the vessel leaves by
            else:
                cl = tan = None
                normal0 = _outward_normal(ring0, centroid0)       # extrusion direction
            specs.append({"valid": valid, "cl": cl, "tan": tan,
                          "normal0": normal0, "centroid0": centroid0})

        # Planarize the GHD-reconstructed mesh's openings BEFORE tubing (only when
        # smoothing), using each opening's outward direction as the plane normal.
        if smooth:
            import os, tempfile
            with tempfile.TemporaryDirectory() as td:
                pre_info = os.path.join(td, "pre_planarize.npz")
                np.savez(pre_info,
                         cpcd_glo_tangent=np.array([s["normal0"][None] for s in specs], dtype=object),
                         opening_centroids=np.array([s["centroid0"] for s in specs]),
                         allow_pickle=True)
                uncapped = planarize_openings(
                    uncapped, r_forward_fusion_info_filename=pre_info,
                    flaw_opening_min_size=flaw_opening_min_size,
                    n_iter=planarize_n_iter, lam=planarize_lam, smooth=True,
                )
            verts = np.asarray(uncapped.points)

        # Pass 2: assemble tube inputs from the (possibly planarized) rings.
        opening_pcd, cpcd_glo, cpcd_tan, centroids = [], [], [], []
        for o, s in enumerate(specs):
            ring = verts[opening_indices[o]]
            centroid = ring.mean(0)
            if s["valid"]:
                cl, tan = s["cl"], s["tan"]
            else:
                normal0 = s["normal0"]
                k = max(5, int(np.ceil(extrude_length / step)) + 1)
                cl = centroid + normal0 * np.linspace(0.0, extrude_length, k)[:, None]
                tan = np.tile(normal0, (k, 1))
            opening_pcd.append(ring)
            cpcd_glo.append(cl)
            cpcd_tan.append(tan)
            centroids.append(centroid)

        # Build and glue tubes.
        l2w = [get_tubular_l2w_trans(t, min_torsion=min_torsion) for t in cpcd_tan]
        tube_verts, sort_idx = get_tubular_mesh_verts(opening_pcd, cpcd_glo, cpcd_tan, l2w)
        opening_idx_sorted = [vid[si] for vid, si in zip(vid_list, sort_idx)]
        tube_faces, tube_verts = get_tubular_mesh_faces(tube_verts, ds_r=ds_r)
        merged = merge_meshes(uncapped, tube_faces, tube_verts, opening_idx_sorted)

        if smooth:
            # smooth_near_openings reads opening_vertex_ids from a file; the
            # original opening indices are still valid in the merged mesh.
            import os, tempfile
            with tempfile.TemporaryDirectory() as td:
                info_path = os.path.join(td, "fusion_info.npz")
                np.savez(info_path,
                         cpcd_glo_tangent=np.array(cpcd_tan, dtype=object),
                         opening_centroids=np.array(centroids),
                         opening_vertex_ids=np.array(vid_list, dtype=object),
                         allow_pickle=True)
                merged = smooth_near_openings(merged, r_forward_fusion_info_filename=info_path,
                                              n_rings=smooth_n_rings, n_iter=smooth_n_iter, lam=smooth_lam)
            merged = remove_orphan_vertices(merged)
            merged = planarize_openings(merged, r_forward_fusion_info_filename=None,
                                        flaw_opening_min_size=flaw_opening_min_size, smooth=False)
        else:
            merged = remove_orphan_vertices(merged)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix.lower() == ".obj":
                pv.save_meshio(str(save_path), merged)
            else:
                merged.save(str(save_path))
        return merged

    def forward_as_meshes(self, ghd, aneurysm_type, **kwargs):
        return self.get(aneurysm_type).ghd_forward_as_Meshes(ghd, **kwargs)

    def forward_many(self, ghd_list, aneurysm_types, **kwargs):
        return [
            self.forward_as_meshes(ghd, aneurysm_type, **kwargs)
            for ghd, aneurysm_type in zip(ghd_list, aneurysm_types)
        ]
