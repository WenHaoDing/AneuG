from dataclasses import dataclass
from pathlib import Path

import torch

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
        self._edge_cache = {}   # type_id → edge_index [2, E]

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

    def forward_as_meshes(self, ghd, aneurysm_type, **kwargs):
        return self.get(aneurysm_type).ghd_forward_as_Meshes(ghd, **kwargs)

    def forward_many(self, ghd_list, aneurysm_types, **kwargs):
        return [
            self.forward_as_meshes(ghd, aneurysm_type, **kwargs)
            for ghd, aneurysm_type in zip(ghd_list, aneurysm_types)
        ]
