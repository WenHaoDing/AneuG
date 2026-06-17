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

    def forward_as_meshes(self, ghd, aneurysm_type, **kwargs):
        return self.get(aneurysm_type).ghd_forward_as_Meshes(ghd, **kwargs)

    def forward_many(self, ghd_list, aneurysm_types, **kwargs):
        return [
            self.forward_as_meshes(ghd, aneurysm_type, **kwargs)
            for ghd, aneurysm_type in zip(ghd_list, aneurysm_types)
        ]
