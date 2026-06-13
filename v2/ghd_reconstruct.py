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

    def forward_as_meshes(self, ghd, aneurysm_type, **kwargs):
        return self.get(aneurysm_type).ghd_forward_as_Meshes(ghd, **kwargs)

    def forward_many(self, ghd_list, aneurysm_types, **kwargs):
        return [
            self.forward_as_meshes(ghd, aneurysm_type, **kwargs)
            for ghd, aneurysm_type in zip(ghd_list, aneurysm_types)
        ]
