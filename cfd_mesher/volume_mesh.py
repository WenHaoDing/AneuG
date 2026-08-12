"""Tetrahedralize a CFD-ready surface into a boundary-tagged volume mesh.

Ported from AneuSeg/cfd_meshing/get_mesh_aneux_tune_elements_v3.py, the volume stage
this package's surface stage (generate_cfd_surface_meshes.py / vessel_reconstruct.py)
was written to feed. That script's per-case fusion bookkeeping already writes exactly
the npz schema this module reads (opening_centroids, cpcd_glo, cpcd_glo_tangent), since
both were developed against the same PostTr/GHD pipeline.

Per case, four things happen to a closed-except-openings surface (.vtp):
  1. optional inlet/outlet flow extension (vtkvmtkPolyDataFlowExtensionsFilter),
     giving the solver a developed-flow length before the boundary condition is imposed
  2. tetrahedralization at an edge length adaptive to the shape's volume and area
     (cfd_mesher.vmtk_backend.cfdmesher_custom)
  3. boundary relabeling: wall/inlet/outlet cell-entity ids are reassigned to a fixed
     policy (WALL_ID/INLET_ID/OUTLET1_ID/OUTLET2_ID) by matching each cap's centroid to
     the branch endpoints recorded in the fusion npz
  4. bookkeeping export: inlet node coordinates (for a Fluent velocity-profile UDF) and,
     for two-outlet cases, a flow-split ratio derived from the two outlets' equivalent
     diameters

Only the functions the original script's __main__ pipeline actually exercised were
ported. Dropped as dead code (present in the original but never called from its
__main__): the centroid-based relabel_inlet_outlets/classify_mesh_surface_entities_by_npz
(superseded by the cpcd-endpoint-based v2 below, which is more robust -- see its
docstring), and _match_centroid (referenced nothing).
"""

import copy
import os

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from vtkmodules.util import numpy_support as ns

import vmtk
from vmtk import vtkvmtk
from vmtk import vmtkscripts

# =========================================================
# Boundary ID policy
# =========================================================
WALL_ID = 3
INLET_ID = 4
OUTLET1_ID = 5
OUTLET2_ID = 6


# =========================================================
# NPZ-based opening identification
# =========================================================
def load_opening_info_from_npz(npz_path):
    """Inlet/outlet centroids from a *_forward_fusion_info.npz.

    Convention inside the npz: opening_centroids[0] = inlet, [1:] = outlet(s).

    Returns (inlet_centroid (3,), outlet_centroids: list of (3,)), length 1 or 2.
    """
    data = np.load(npz_path, allow_pickle=True)
    centroids = data["opening_centroids"]
    n_openings = centroids.shape[0]
    if n_openings not in (2, 3):
        raise RuntimeError("Expected 2 or 3 opening centroids in %s, got %d."
                           % (npz_path, n_openings))
    inlet_centroid = centroids[0].copy()
    outlet_centroids = [centroids[i].copy() for i in range(1, n_openings)]
    return inlet_centroid, outlet_centroids


# =========================================================
# Small geometry helpers
# =========================================================
def _points_from_vtk_cell(cell):
    pts = ns.vtk_to_numpy(cell.GetPoints().GetData())
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return np.asarray(pts, dtype=np.float64)


def _polygon_area_3d(points):
    """Area of a 3D planar polygon from its ordered boundary points (Newell's method)."""
    if points.shape[0] < 3:
        return 0.0
    area_vec = np.zeros(3, dtype=np.float64)
    for i in range(points.shape[0]):
        p0, p1 = points[i], points[(i + 1) % points.shape[0]]
        area_vec += np.cross(p0, p1)
    return 0.5 * np.linalg.norm(area_vec)


def _triangle_area(points):
    if points.shape[0] < 3:
        return 0.0
    return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))


def _polygon_or_triangle_area(points):
    return _triangle_area(points) if points.shape[0] == 3 else _polygon_area_3d(points)


def _area_weighted_centroid(accum_centroid, accum_area, local_centroid, local_area):
    total_area = accum_area + local_area
    if total_area <= 0.0:
        return accum_centroid, accum_area
    new_centroid = (accum_centroid * accum_area + local_centroid * local_area) / total_area
    return new_centroid, total_area


# =========================================================
# Surface/open-boundary analysis on input surface (.vtp)
# =========================================================
def get_open_boundary_info_from_surface(surface):
    """Boundary loops in the order vtkvmtkPolyDataBoundaryExtractor returns them,
    which is the order vtkvmtkPolyDataFlowExtensionsFilter's boundary ids expect."""
    boundary_extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundary_extractor.SetInputData(surface)
    boundary_extractor.Update()
    boundaries = boundary_extractor.GetOutput()

    info = []
    for boundary_id in range(boundaries.GetNumberOfCells()):
        pts = _points_from_vtk_cell(boundaries.GetCell(boundary_id))
        area = _polygon_area_3d(pts)
        centroid = pts.mean(axis=0) if pts.size else np.zeros(3, dtype=np.float64)
        info.append({"boundary_id": boundary_id, "area": float(area), "centroid": centroid,
                     "n_points": int(pts.shape[0])})
    return info


def classify_openings_by_npz(open_boundary_info, inlet_centroid, outlet_centroids):
    """Match detected open boundaries to the npz inlet/outlet centroids by nearest
    distance (the npz centroids are measured pre-smoothing, so they can drift a little
    from the boundary centroids on the final surface).

    Returns (inlet: dict, outlets: list[dict] ordered as outlet_centroids).
    """
    n_open = len(open_boundary_info)
    n_expected = 1 + len(outlet_centroids)
    if n_open != n_expected:
        raise RuntimeError(
            "Expected %d open boundaries (1 inlet + %d outlet(s)), but found %d on the surface."
            % (n_expected, len(outlet_centroids), n_open))

    all_ref_centroids = [inlet_centroid] + outlet_centroids
    detected_centroids = [b["centroid"] for b in open_boundary_info]
    used, matched = set(), {}

    for ref_idx, ref_c in enumerate(all_ref_centroids):
        best_bi, best_dist = -1, np.inf
        for bi, det_c in enumerate(detected_centroids):
            if bi in used:
                continue
            d = np.linalg.norm(ref_c - det_c)
            if d < best_dist:
                best_dist, best_bi = d, bi
        if best_bi < 0:
            raise RuntimeError("Could not match reference centroid index %d to any "
                               "remaining detected boundary." % ref_idx)
        matched[ref_idx] = open_boundary_info[best_bi]
        used.add(best_bi)

    inlet = matched[0]
    outlets = [matched[i] for i in range(1, n_expected)]
    return inlet, outlets


def _run_flow_extensions_on_boundaries(surface, boundary_ids, extension_length,
                                       extension_mode="boundarynormal",
                                       interpolation_mode="thinplatespline",
                                       transition_ratio=0.25, target_boundary_points=50,
                                       sigma=1.0, adaptive_extension_radius=True,
                                       adaptive_extension_length=False):
    """Apply vtkvmtkPolyDataFlowExtensionsFilter to a selected list of open boundaries.

    Returns a fresh vtkPolyData copy so the output stays valid after the filter object
    goes out of scope.
    """
    if extension_length < 0.0:
        raise ValueError("extension_length must be >= 0, got %s" % extension_length)

    boundary_ids = [int(bid) for bid in boundary_ids]
    if len(boundary_ids) == 0 or extension_length == 0.0:
        output_surface = vtk.vtkPolyData()
        output_surface.DeepCopy(surface)
        return output_surface

    vtk_boundary_ids = vtk.vtkIdList()
    for boundary_id in boundary_ids:
        vtk_boundary_ids.InsertNextId(boundary_id)

    flow_ext = vtkvmtk.vtkvmtkPolyDataFlowExtensionsFilter()
    flow_ext.SetInputData(surface)
    flow_ext.SetSigma(float(sigma))
    flow_ext.SetAdaptiveExtensionLength(int(adaptive_extension_length))
    flow_ext.SetAdaptiveExtensionRadius(int(adaptive_extension_radius))
    flow_ext.SetAdaptiveNumberOfBoundaryPoints(0)
    flow_ext.SetExtensionLength(float(extension_length))
    flow_ext.SetExtensionRatio(1.0)
    flow_ext.SetExtensionRadius(1.0)
    flow_ext.SetTransitionRatio(float(transition_ratio))
    flow_ext.SetCenterlineNormalEstimationDistanceRatio(1.0)
    flow_ext.SetNumberOfBoundaryPoints(int(target_boundary_points))
    flow_ext.SetBoundaryIds(vtk_boundary_ids)

    if extension_mode == "boundarynormal":
        flow_ext.SetExtensionModeToUseNormalToBoundary()
    elif extension_mode == "centerlinedirection":
        raise ValueError("centerlinedirection mode requires centerlines; use "
                         "extension_mode='boundarynormal' for this automatic pipeline.")
    else:
        raise ValueError("Unsupported extension_mode: %s" % extension_mode)

    if interpolation_mode == "linear":
        flow_ext.SetInterpolationModeToLinear()
    elif interpolation_mode == "thinplatespline":
        flow_ext.SetInterpolationModeToThinPlateSpline()
    else:
        raise ValueError("Unsupported interpolation_mode: %s" % interpolation_mode)

    flow_ext.Update()
    output_surface = vtk.vtkPolyData()
    output_surface.DeepCopy(flow_ext.GetOutput())
    return output_surface


def extend_inlet_and_outlets(surface_file, output_file, inlet_centroid, outlet_centroids,
                             extend_inlet=True, inlet_extension_length=0.0,
                             extend_outlets=True, outlet_extension_length_one_outlet=0.0,
                             outlet_extension_length_two_outlets=0.0,
                             extension_mode="boundarynormal",
                             interpolation_mode="thinplatespline",
                             transition_ratio=0.25, target_boundary_points=50, sigma=1.0,
                             adaptive_extension_radius=True, adaptive_extension_length=False):
    """Extend inlet and outlet openings with independent lengths, identified by
    matching detected boundary centroids to inlet_centroid/outlet_centroids (from
    load_opening_info_from_npz for real cases; computed however the caller likes
    otherwise -- this function does not care where the reference centroids came from).

    The outlet extension length is picked from outlet_extension_length_one_outlet or
    _two_outlets depending on how many outlets this case has.

    Runs in two passes (optional inlet extension, then re-detect boundaries, then
    optional outlet extension) because vtkvmtkPolyDataFlowExtensionsFilter applies one
    extension length per execution.
    """
    for name, val in [("inlet_extension_length", inlet_extension_length),
                      ("outlet_extension_length_one_outlet", outlet_extension_length_one_outlet),
                      ("outlet_extension_length_two_outlets", outlet_extension_length_two_outlets)]:
        if val < 0.0:
            raise ValueError("%s must be >= 0, got %s" % (name, val))

    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = surface_file
    reader.Execute()
    surface = reader.Surface

    open_boundary_info = get_open_boundary_info_from_surface(surface)
    inlet, outlets = classify_openings_by_npz(open_boundary_info, inlet_centroid, outlet_centroids)
    n_outlets = len(outlets)

    if extend_inlet and inlet_extension_length > 0.0:
        surface = _run_flow_extensions_on_boundaries(
            surface=surface, boundary_ids=[inlet["boundary_id"]],
            extension_length=inlet_extension_length, extension_mode=extension_mode,
            interpolation_mode=interpolation_mode, transition_ratio=transition_ratio,
            target_boundary_points=target_boundary_points, sigma=sigma,
            adaptive_extension_radius=adaptive_extension_radius,
            adaptive_extension_length=adaptive_extension_length)
        # Boundaries may be renumbered after extension; re-detect before the outlet pass.
        open_boundary_info = get_open_boundary_info_from_surface(surface)
        inlet, outlets = classify_openings_by_npz(open_boundary_info, inlet_centroid, outlet_centroids)
        n_outlets = len(outlets)

    if extend_outlets:
        if n_outlets == 1:
            outlet_extension_length = outlet_extension_length_one_outlet
        elif n_outlets == 2:
            outlet_extension_length = outlet_extension_length_two_outlets
        else:
            raise RuntimeError("Unsupported number of outlets after inlet extension: %d" % n_outlets)

        if outlet_extension_length > 0.0:
            surface = _run_flow_extensions_on_boundaries(
                surface=surface, boundary_ids=[o["boundary_id"] for o in outlets],
                extension_length=outlet_extension_length, extension_mode=extension_mode,
                interpolation_mode=interpolation_mode, transition_ratio=transition_ratio,
                target_boundary_points=target_boundary_points, sigma=sigma,
                adaptive_extension_radius=adaptive_extension_radius,
                adaptive_extension_length=adaptive_extension_length)

    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.OutputFileName = output_file
    writer.Execute()


def extend_outlets_only(surface_file, output_file, inlet_centroid, outlet_centroids,
                        extension_length, **kwargs):
    """Apply the same outlet extension length to both 1-outlet and 2-outlet cases."""
    return extend_inlet_and_outlets(
        surface_file=surface_file, output_file=output_file,
        inlet_centroid=inlet_centroid, outlet_centroids=outlet_centroids,
        extend_inlet=False, inlet_extension_length=0.0, extend_outlets=True,
        outlet_extension_length_one_outlet=extension_length,
        outlet_extension_length_two_outlets=extension_length, **kwargs)


# =========================================================
# Boundary/entity analysis on final volume mesh (.vtu)
# =========================================================
def get_surface_entity_info_from_vtu(mesh_file):
    """(mesh_arr, ugrid, info) where info is one dict per 2D CellEntityIds group:
    {entity_id, area, centroid, n_cells}. 3D (volume) entities are skipped."""
    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(mesh_file)
    vtk_reader.Update()
    ugrid = vtk_reader.GetOutput()

    mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
    mesh_reader.InputFileName = mesh_file
    mesh_reader.Execute()
    mesh2np = vmtk.vmtkmeshtonumpy.vmtkMeshToNumpy()
    mesh2np.Mesh = mesh_reader.Mesh
    mesh2np.Execute()
    mesh_arr = mesh2np.ArrayDict

    entity_ids = mesh_arr["CellData"]["CellEntityIds"]
    info = []
    for entity_id in np.unique(entity_ids):
        cell_indices = np.where(entity_ids == entity_id)[0]
        if len(cell_indices) == 0:
            continue
        if ugrid.GetCell(int(cell_indices[0])).GetCellDimension() != 2:
            continue

        total_area, total_centroid, total_cells = 0.0, np.zeros(3, dtype=np.float64), 0
        for cell_idx in cell_indices:
            pts = _points_from_vtk_cell(ugrid.GetCell(int(cell_idx)))
            cell_area = _polygon_or_triangle_area(pts)
            if cell_area <= 0.0:
                continue
            total_centroid, total_area = _area_weighted_centroid(
                total_centroid, total_area, pts.mean(axis=0), cell_area)
            total_cells += 1

        info.append({"entity_id": int(entity_id), "area": float(total_area),
                     "centroid": total_centroid, "n_cells": int(total_cells)})
    return mesh_arr, ugrid, info


def _relabel_surface_entities(mesh_file, wall, inlet, outlets):
    """Shared write-back for both relabeling schemes: remap CellEntityIds on 2D cells
    only (volume-cell ids are left untouched even if they collide with a target id)."""
    mesh_arr, ugrid, _ = get_surface_entity_info_from_vtu(mesh_file)
    old_to_new = {int(wall["entity_id"]): WALL_ID, int(inlet["entity_id"]): INLET_ID}
    if len(outlets) >= 1:
        old_to_new[int(outlets[0]["entity_id"])] = OUTLET1_ID
    if len(outlets) == 2:
        old_to_new[int(outlets[1]["entity_id"])] = OUTLET2_ID

    old_ids = np.asarray(mesh_arr["CellData"]["CellEntityIds"]).copy()
    new_ids = old_ids.copy()
    surface_mask = np.array([ugrid.GetCell(i).GetCellDimension() == 2
                             for i in range(ugrid.GetNumberOfCells())])
    for old_id, new_id in old_to_new.items():
        new_ids[surface_mask & (old_ids == old_id)] = new_id
    mesh_arr["CellData"]["CellEntityIds"] = new_ids

    np2mesh = vmtk.vmtknumpytomesh.vmtkNumpyToMesh()
    np2mesh.ArrayDict = mesh_arr
    np2mesh.Execute()
    writer = vmtk.vmtkmeshwriter.vmtkMeshWriter()
    writer.Mesh = np2mesh.Mesh
    writer.OutputFileName = mesh_file
    writer.Execute()


def relabel_inlet_outlets_by_centroids(mesh_file, inlet_centroid, outlet_centroids):
    """Reassign boundary ids by matching each cap's centroid to inlet_centroid /
    outlet_centroids: wall -> 3, inlet -> 4, outlet(s) -> 5 [, 6] (in outlet_centroids
    order). Wall is identified as the largest-area 2D entity; among the rest, each cap
    is matched to its nearest reference point (each reference used at most once).

    This is the general-purpose matcher: it doesn't care where the reference points
    came from. relabel_inlet_outlets_v2 below supplies cpcd branch endpoints for real
    cases; the synthetic pipeline supplies per-sample opening centroids instead.
    """
    outlet_centroids = list(outlet_centroids)
    _, _, surface_info = get_surface_entity_info_from_vtu(mesh_file)
    ranked = sorted(surface_info, key=lambda d: -d["area"])
    wall, caps = ranked[0], ranked[1:]                    # wall = largest-area 2D entity

    n_expected_caps = 1 + len(outlet_centroids)
    if len(caps) != n_expected_caps:
        raise RuntimeError("Expected %d cap groups (1 inlet + %d outlet(s)), but found %d."
                           % (n_expected_caps, len(outlet_centroids), len(caps)))

    all_refs = [inlet_centroid] + outlet_centroids
    cap_centroids = [c["centroid"] for c in caps]
    used, matched = set(), {}
    for ref_idx, ref_pt in enumerate(all_refs):
        best_ci, best_dist = -1, np.inf
        for ci, cap_c in enumerate(cap_centroids):
            if ci in used:
                continue
            d = np.linalg.norm(ref_pt - cap_c)
            if d < best_dist:
                best_dist, best_ci = d, ci
        if best_ci < 0:
            raise RuntimeError("Could not match reference point %d to any cap entity." % ref_idx)
        matched[ref_idx] = caps[best_ci]
        used.add(best_ci)

    inlet = matched[0]
    outlets = [matched[i] for i in range(1, n_expected_caps)]
    _relabel_surface_entities(mesh_file, wall, inlet, outlets)

    print("Boundary relabeling finished:")
    print("  wall    old id %s -> new id %s" % (wall["entity_id"], WALL_ID))
    print("  inlet   old id %s -> new id %s" % (inlet["entity_id"], INLET_ID))
    if len(outlets) >= 1:
        print("  outlet1 old id %s -> new id %s" % (outlets[0]["entity_id"], OUTLET1_ID))
    if len(outlets) == 2:
        print("  outlet2 old id %s -> new id %s" % (outlets[1]["entity_id"], OUTLET2_ID))


def relabel_inlet_outlets_v2(mesh_file, npz_path):
    """Reassign boundary ids using the last point of each cpcd branch in the npz as the
    reference for relabel_inlet_outlets_by_centroids: inlet <- cpcd_glo[0][-1],
    outlet(s) <- cpcd_glo[1:][-1], in order.

    Uses cpcd_glo endpoints rather than opening_centroids because cpcd_glo stores the
    full extension centerline path -- its last point sits right at the post-extension
    cap face, matching the real cap centroids on the volume mesh. opening_centroids are
    measured on the pre-extension clipped mesh and can be far from the real caps.
    """
    data = np.load(npz_path, allow_pickle=True)
    cpcd_glo = data["cpcd_glo"]
    n_branches = len(cpcd_glo)
    if n_branches < 2:
        raise RuntimeError("Expected at least 2 cpcd branches in %s, got %d." % (npz_path, n_branches))
    if n_branches > 3:
        raise RuntimeError(
            "Expected 2 or 3 cpcd branches in %s, got %d. Only single- and two-outlet "
            "cases are supported." % (npz_path, n_branches))

    inlet_ref = cpcd_glo[0][-1].copy()
    outlet_refs = [cpcd_glo[i][-1].copy() for i in range(1, n_branches)]
    relabel_inlet_outlets_by_centroids(mesh_file, inlet_ref, outlet_refs)


def scan_inlet_nodes(mesh_file, csv_path=None, scale_factor=0.001, inlet_id=INLET_ID):
    """Write every inlet-cell node coordinate to CSV (mm*scale_factor), for a Fluent
    velocity-profile UDF that needs the inlet geometry directly."""
    csv_path = csv_path or os.path.join(os.path.dirname(mesh_file), "inlet_centroids.csv")

    mesh_reader = vmtk.vmtkmeshreader.vmtkMeshReader()
    mesh_reader.InputFileName = mesh_file
    mesh_reader.Execute()
    mesh2np = vmtk.vmtkmeshtonumpy.vmtkMeshToNumpy()
    mesh2np.Mesh = mesh_reader.Mesh
    mesh2np.Execute()
    entity_ids = mesh2np.ArrayDict["CellData"]["CellEntityIds"]

    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(mesh_file)
    vtk_reader.Update()
    ugrid = vtk_reader.GetOutput()

    cell_indices = np.where(entity_ids == inlet_id)[0]
    if len(cell_indices) == 0:
        raise RuntimeError("No cells found for inlet id=%s in %s" % (inlet_id, mesh_file))

    point_set = np.concatenate(
        [copy.deepcopy(ns.vtk_to_numpy(ugrid.GetCell(int(i)).GetPoints().GetData()))
         for i in cell_indices], axis=0) * scale_factor
    pd.DataFrame(point_set, columns=["x", "y", "z"]).to_csv(csv_path, index=False)
    return csv_path


# =========================================================
# Adaptive edge length
# =========================================================
def get_surface_geometry_stats(surface_file):
    """(volume, surface_area) of an open surface mesh: volume via the divergence
    theorem (pyvista), area by summing face areas. Both approximate for an open mesh,
    but accurate enough to drive edge-length scaling."""
    mesh = pv.read(surface_file)
    return float(abs(mesh.volume)), float(mesh.area)


def compute_adaptive_edge(volume, area, base_edge=0.13, ref_volume=350.0, ref_area=350.0,
                          vol_exponent=0.25, area_exponent=0.40):
    """Scale edge length by whichever of volume or area implies the coarser (larger)
    edge, so a small-volume-but-elongated shape is still caught.

      edge_from_vol  = base_edge * (volume / ref_volume)^vol_exponent   (gentler: keeps
                       larger shapes monotonically getting more elements)
      edge_from_area = base_edge * (area   / ref_area  )^area_exponent  (stronger:
                       aggressively catches thin/elongated high-area, low-volume shapes)
      final = max(edge_from_vol, edge_from_area)

    Shapes at or below the reference volume/area keep base_edge unchanged.
    """
    edge_vol = base_edge if volume <= ref_volume else base_edge * (volume / ref_volume) ** vol_exponent
    edge_area = base_edge if area <= ref_area else base_edge * (area / ref_area) ** area_exponent
    return round(max(edge_vol, edge_area), 4)


# =========================================================
# Flow split (two-outlet cases)
# =========================================================
def compute_equivalent_diameter_from_area(area):
    if area <= 0.0:
        raise ValueError("Area must be positive to compute equivalent diameter, got %s" % area)
    return float(np.sqrt(4.0 * area / np.pi))


def write_flowsplit_ratio(mesh_file, exponent=2.45, output_path=None):
    """For two-outlet cases, write the normalized flow-split ratio implied by
    Q5/Q6 = (d5/d6)^exponent, where d5/d6 are equivalent diameters from outlet cap
    areas. Output is one line "ratio_id5 ratio_id6" (they sum to 1).

    One-outlet cases write nothing (and remove a stale file from a previous run).
    """
    output_path = output_path or os.path.join(os.path.dirname(mesh_file), "flowsplit_ratio.txt")
    _, _, surface_info = get_surface_entity_info_from_vtu(mesh_file)
    info_by_id = {int(item["entity_id"]): item for item in surface_info}

    has_outlet5, has_outlet6 = OUTLET1_ID in info_by_id, OUTLET2_ID in info_by_id
    if has_outlet5 and not has_outlet6:
        if os.path.exists(output_path):
            os.remove(output_path)
        print("Single-outlet case detected. flowsplit_ratio.txt not written.")
        return None
    if not (has_outlet5 and has_outlet6):
        raise RuntimeError(
            "Could not determine a valid outlet configuration in %s. Found outlet ids: %s"
            % (mesh_file, [e for e in (OUTLET1_ID, OUTLET2_ID) if e in info_by_id]))

    d5 = compute_equivalent_diameter_from_area(float(info_by_id[OUTLET1_ID]["area"]))
    d6 = compute_equivalent_diameter_from_area(float(info_by_id[OUTLET2_ID]["area"]))
    ratio56 = float((d5 / d6) ** exponent)
    q5, q6 = ratio56 / (1.0 + ratio56), 1.0 / (1.0 + ratio56)
    s = q5 + q6                                # normalize exactly, against round-off
    q5, q6 = q5 / s, q6 / s

    with open(output_path, "w") as f:
        f.write("%.16f %.16f\n" % (q5, q6))
    print("flowsplit_ratio.txt written: id=5 -> %.16f, id=6 -> %.16f (d5=%.6f, d6=%.6f, exponent=%s)"
          % (q5, q6, d5, d6, exponent))
    return output_path


def report_mesh_stats(vtu_path):
    """Print node/volume-cell/surface-face counts; returns the volume-cell count."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    ugrid = reader.GetOutput()

    n_3d = n_2d = 0
    for i in range(ugrid.GetNumberOfCells()):
        dim = ugrid.GetCell(i).GetCellDimension()
        n_3d += dim == 3
        n_2d += dim == 2

    print("  Mesh stats: %s nodes, %s volume cells, %s surface faces"
          % (ugrid.GetNumberOfPoints(), n_3d, n_2d))
    return n_3d
