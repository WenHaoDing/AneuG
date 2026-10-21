# Cap masks for AneuGv2 generated shapes

Each generated shape's `dome.obj` is a **closed** surface: the vessel openings are capped.
`cap_masks.npz` marks which part of `dome.obj` is a cap, **which opening each cap is**, and
which vertices form a clean vessel wall.

## Where the data is

| | |
|---|---|
| Hugging Face | `whding123/AneuGv2` (dataset), `geometry/<case>/cap_masks.npz` (plus `uncap_sanity.png`) |
| Local | `/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2/<case>/` |
| Mesh to use it on | `dome.obj` in the same folder |
| Coverage | 643 of the 650 `cohort_650_*` cases |

Seven cases have no mask because they failed verification:
`cohort_650_0217_t0`, `0340_t1`, `0372_t1`, `0436_t1`, `0448_t1`, `0567_t1` and `0649_t1`.
**Skip them.**

## Load it

Read `dome.obj` with a plain OBJ parser, **in file order**. The masks index its vertices
and faces directly.

- Don't let a library merge, reorder, clean or remesh it. For example, use
  `trimesh.load(path, process=False)`.
- Check `len(V) == z["dome_n_vertices"]` and `len(F) == z["dome_n_faces"]` before you
  index anything.

```python
import numpy as np, trimesh
z = np.load("<case>/cap_masks.npz", allow_pickle=True)
m = trimesh.load("<case>/dome.obj", process=False)
V, F = np.asarray(m.vertices), np.asarray(m.faces)
assert len(V) == z["dome_n_vertices"] and len(F) == z["dome_n_faces"]

wall_v = z["wall_vertex_mask"]          # (V,) bool -- use these for anything "vessel wall"
wall_f = z["wall_face_mask"]            # (F,) bool
for b, role in enumerate(z["roles"]):   # branch order: 0 is always the inlet
    zone_v = z["excluded_vertex_masks"][b]   # this opening's non-wall vertices
    cap_v  = z["vertex_masks"][b]            # the cap itself, exactly as generated
```

## Which cap is which

Every per-opening array is in **branch order**, and **branch 0 is always the inlet**.
The same opening has a different name in each file, so all of them are stored:

| branch `b` | `roles[b]` | `fluent_ids[b]` (volume mesh / Fluent) | `wall_zones[b]` (`wall_data.pt`) |
|---|---|---|---|
| 0 | `inlet` | 4 | `inlet` |
| 1 | `outlet1` | 5 | **`outlet_0`** |
| 2 | `outlet2` (bifurcated only) | 6 | **`outlet_1`** |

Note the **off-by-one in the outlet names**: `outlet1` is `outlet_0` in `wall_data.pt`.
Use `wall_zones[b]` to look up a CFD zone. Don't build the name from the index.

Sidewall shapes (`aneurysm_type` 1) have two openings, bifurcated shapes (type 0) three.

## Two kinds of mask

| field | shape | meaning |
|---|---|---|
| `vertex_masks`, `face_masks` | (n, V), (n, F) | the **exact cap** the generator cut out. A face counts as cap when all three of its vertices are cap. |
| `excluded_vertex_masks`, `excluded_face_masks` | (n, V), (n, F) | **everything that isn't wall**, assigned to the nearest opening. |
| `wall_vertex_mask`, `wall_face_mask` | (V,), (F,) | the **clean wall** |

**Use the wall and excluded masks for downstream work.** During generation, seam smoothing
moved dome vertices near each rim off the final surface, by up to about 0.6 mm. The wall is
defined as all vertices outside every cap and within `off_surface_tol_mm` (**0.2 mm**) of
the merged surface (`mesh.obj`), keeping only the **single largest connected piece**. So
there are no stray fragments and nothing far from the real surface. Every vertex is either
wall or in exactly one excluded zone.

To use a different threshold, every vertex's distance to the surface is stored in
`surface_distance_mm` (V,). Recompute the wall from that, and re-take the largest connected
piece.

## Other fields

- `rim_loops[b]`: ordered `dome.obj` vertex indices around cap `b`.
- `rim_centroids`, `neck_centroids`: the cap rim centre, regenerated and as the generator
  recorded it.
- `verification`: per-case check results (JSON).
- `uncap_sanity.png`: the caps drawn in strong colours, the off-surface band in pale ones,
  and `mesh.obj` overlaid translucent.

## How it was made

`scripts/generate/regen_cap_masks.py` reruns the generator's own uncapping on `ghd.npy`,
with the same sensor and the same call. It then verifies each case against what the
generator saved:
- every regenerated rim sits on its own branch's stored neck;
- the cap vertices are exactly the ones deleted from `mesh.obj`;
- the far-from-rim wall vertices are exact copies in `mesh.obj`.

The labels therefore follow the generator's branch order. Branch 0 being the inlet is the
generator's convention, and the CFD boundaries follow the same one.
