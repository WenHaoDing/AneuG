"""VMTK centerlines between the manual endpoints, saved per path (run with the vmtk310 env).

Same settings as AneuSeg's vmtk_extract_centerline (source = endpoint 0, resample 0.1 mm,
100 smoothing iterations at 0.1), but every source->target path is kept whole, with its
MaximumInscribedSphereRadius, instead of being split into branch groups.

  python geometry/vmtk_centerlines.py --mesh m.ply --endpoints endpoints_manual.npy --out cl.npz
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--endpoints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source_id", type=int, default=0)
    a = p.parse_args()

    from vmtk import vmtkscripts
    from vtk.util.numpy_support import vtk_to_numpy

    ep = np.asarray(np.load(a.endpoints, allow_pickle=True).item()["endpoints"], float)
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = a.mesh
    reader.Execute()

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = reader.Surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = ep[a.source_id].tolist()
    cl.TargetPoints = [x for j, pt in enumerate(ep) if j != a.source_id for x in pt.tolist()]
    cl.AppendEndPoints = 1
    cl.Execute()

    rs = vmtkscripts.vmtkCenterlineResampling()
    rs.Centerlines = cl.Centerlines
    rs.Length = 0.1
    rs.Execute()
    sm = vmtkscripts.vmtkCenterlineSmoothing()
    sm.Centerlines = rs.Centerlines
    sm.NumberOfSmoothingIterations = 100
    sm.SmoothFactor = 0.1
    sm.Execute()

    poly = sm.Centerlines
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    rad = vtk_to_numpy(poly.GetPointData().GetArray("MaximumInscribedSphereRadius")).astype(float)
    paths = {}
    for i in range(poly.GetNumberOfCells()):
        ids = poly.GetCell(i).GetPointIds()
        idx = np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
        paths[f"path{i}_pts"] = pts[idx]
        paths[f"path{i}_r"] = rad[idx]
    np.savez(a.out, n_paths=poly.GetNumberOfCells(), endpoints=ep, source_id=a.source_id, **paths)
    print("paths:", poly.GetNumberOfCells(), [len(paths[f"path{i}_pts"]) for i in range(poly.GetNumberOfCells())])


if __name__ == "__main__":
    main()
