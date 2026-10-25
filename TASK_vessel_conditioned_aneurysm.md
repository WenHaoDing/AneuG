# New task: vessel-conditioned aneurysm generation for mask-conditioned image synthesis

Status: **design agreed for the condition; data pipeline in progress.** The condition is the
**1D graph of the vessel** (centerline + radius), and a model generates the **ostium** from
it (§5). The graph extraction runs on 9 test cases (`vessel_edit/`, outputs in
`runtime_skeleton/`). The new branch has not been created yet.

---

## 1. What we want

Given a **healthy vessel** from a real TOF-MRA scan, generate an **aneurysm sac**
conditioned on that vessel, attach it at a generated site, and output a **voxel mask in the
scan's own coordinates**. The **parent vessel must stay exactly as it was**: it is taken
from the scan, never generated.

The consumer is the image-generation project (`../image-generation/PROJECT.md`). That
project is a 3D diffusion model that inpaints a TOF-MRA core patch from (a) the
surrounding real image and (b) a binary vessel mask, optionally with an aneurysm channel.
Its intended use is:

```
healthy scan -> vessel segmentation -> [THIS PROJECT: add a sac] -> voxelised mask with aneurysm
             -> crop context, blank the core -> diffusion inpainting -> synthetic scan + matching label
```

So this is **mask-level data augmentation**: synthetic aneurysm masks on real anatomy,
which the image model turns into images. The publication angle is controllable, anatomy-
preserving aneurysm synthesis in image space.

## 2. How this differs from what we have and from SynVA

| | AneuG (current repo) | SynVA-A1 / A2 | This task |
|---|---|---|---|
| What is generated | sac + parent-vessel stubs (complex), plus centerlines and tubes | sac + single ostium only | the **ostium** from the vessel's 1D graph (§5.2), then the sac (§5.4) |
| Parent vessel | **generated** | given, as a mesh | given, **from the scan**, unchanged to the voxel |
| Condition | morphology markers | local vessel point patch + a chosen ostium | the **1D vessel graph** (centerline + radius) |
| Ostium / site | implied by the generated complex | manual GUI or rule-based (bifurcation / curvature) | **generated**, learned from real (graph, ostium) pairs |
| Output | merged surface mesh | labelled mesh | **voxel mask in scan coordinates** (vessel + aneurysm labels) |
| Downstream | CFD | morphology / realism study | **image synthesis** (and segmentation augmentation) |

Where we differ from SynVA:

- **The vessel is real and preserved exactly.** SynVA stitches meshes. We only need a
  mask, so the merge is a voxel **union**: real vessel voxels plus sac voxels (§5.5).
- **The ostium is generated, not picked.** SynVA chooses it by hand or by rule; we learn
  where and how sacs attach from real aneurysms.
- **The condition is a skeleton, not a surface patch.** It needs no healthy surface under
  the sac (§3.1), and it is cheap to extract at inference.
- **Image space is the target.** The sac must be realistic once voxelised at the scan grid
  (0.5 mm isotropic in the image project; aneurysms 2–13 mm, median 5.4 mm). Sub-voxel
  surface detail does not matter.

## 3. Why the condition is a 1D graph

### 3.1 A healthy surface under the sac cannot be recovered

Three attempts, all on local copies of ImperialNHS cases, and all dropped:
- **Hole fill (Geomagic or similar):** the fill is visible as a patch.
- **Centerline-guided rebuild** (sac removed, tube rebuilt along a bridged centerline and
  blended into the real wall): watertight, but traces remain. The rebuilt part is smoother
  than the rest, and bifurcations end up with facets.
- **Underlying reason:** an aneurysm deforms the local vessel around it, so the "healthy"
  wall under the sac is not in the data.

### 3.2 What the skeleton keeps, and what is repaired

We assume the **centerline of the parent branches does not change** when a sac grows, and
we condition on the skeleton of the local vessel tree instead of its surface. Only the
**radius** near the sac is unreliable (VMTK's inscribed sphere grows into the sac, up to
2.3× in our cases, and centerlines can run into the sac), so the radius there is
interpolated (§4.2).

At inference the host vessel is already healthy: only the skeleton extraction runs, about
1 s per vessel (VMTK centerlines 0.7 s on a 10k-face mesh).

## 4. The 1D graph

### 4.1 Format (`skeleton_graph.npy`, from `geometry/extract_skeleton.py`)

`extract_skeleton(mesh, endpoints)` -> dense graph (points every 0.1 mm):
- `nodes (N, 3)`: vessel ends, bifurcation points, and joints between a branch and a
  bifurcation piece.
- `edges`: per edge `u, v`, `kind` (`branch` or `bifurcation`), `pts`, `radius` (the
  condition: interpolated, then smoothed), `radius_interp`, `radius_raw`, `aneurysm`
  (per point: radius was interpolated), `extracted` (per point: False only for an added
  node point).

It is a local copy of AneuSeg's VMTK steps (same endpoints, source = endpoint 0,
resampled to 0.1 mm, smoothed 100 × 0.1). **Unlike `merge_centerline_v3`, the bifurcation
pieces** (`vmtkBranchExtractor` groups with `Blanking = 1`, 1–1.7 mm each) **are kept**,
so the graph is one connected tree.

VMTK computes one line per source→target pair, so near a bifurcation several lines run side
by side, a few tenths of a mm apart. Where a piece comes from a different line than the
branch it meets, the line is cut back to its point closest to the node (its first points
overlap the other branch). The remaining offset (≤ 0.54 mm on 9 cases) is then filled at
0.1 mm spacing (`extracted = False`). Finally, positions are smoothed over the graph
(σ = 0.5 mm, vessel ends fixed; the originals are kept as `pts_extracted`). That brings
the sharpest bend over 0.3 mm from 87–120° down to 15–23° (37° in EerG).

**Model input file: `skeleton_graph_points.npz`.** This is the same graph flattened into one
point graph; save and load it with `save_point_graph` / `load_point_graph`. Every dense
point is a node, and a branch is a chain between bifurcations or vessel ends (each
bifurcation piece joins the branch it leads into). Branch ids are given breadth-first from
the VMTK source end, so the root branch is 1. Arrays:
- `pos (N, 3)`, `radius (N,)`, `edge_index (2, E)` (both directions);
- `branch` (1..B), `branch_id` (the same, but 0 inside the aneurysm region), and
  `branch_onehot (N, B+1)`;
- the masks `aneurysm`, `is_junction`, `is_end` and `extracted`;
- `meta` (smoothing scales, units, frame).

Branch ids are only labels within one case, so the one-hot width differs per case. The
aneurysm label (0) is on the target side, so keep it out of the condition.

### 4.2 Building the graph for a real aneurysm (training data)

Per case (`vessel_edit/skeleton_pipeline.py`):
1. Copy the case's `label.nrrd`, meshes and `endpoints_manual.npy` locally. The geometry
   dataset is read-only.
2. Remove the sac (from `label.nrrd`, label 2) plus a 1 mm collar from
   `reconstruction4clipping.ply`, and patch the vessel (`vessel_edit/heal_vessel.py`) so
   VMTK's centerline is not pulled into the sac. If the sac sits at the end of that mesh,
   use the larger `reconstruction.ply`. This step is offline only, 1–8 min per case.
3. `extract_skeleton` on the patched mesh. On each centerline, the **aneurysm region**
   (points whose inscribed sphere touches the sac or collar, widened by 0.75 r) gets its
   radius replaced by a C1 Hermite curve between the healthy anchors.
4. Smooth the radius of the whole graph (`smooth_mm`, default 1 mm). The smoothing is
   diffusion over the graph, continuous through the nodes. The small fluctuations are
   segmentation noise. Measured on 9 cases (roughness = scatter of the radius around its
   1 mm moving average):

   | smoothing | outside the region | interpolated region |
   |---|---|---|
   | none | 10–19 µm | 0.1–3.7 µm |
   | σ = 1 mm | 2.5–7.6 µm | 0.1–3.3 µm |
   | σ = 3 mm | 1.1–2.3 µm | 0.2–1.7 µm |

   **Smoothing alone does not fully hide the region:** an interpolated stretch has no fine
   variation at all. Still open: radius jitter as training-time augmentation, or "decoy"
   interpolations at random sites (§8).

The graph is built from VMTK's own topology (line and tract order, no distance
tolerance). All 9 test cases give one connected tree, with no cycles, and every node has
one radius shared by all its edges.

Note: in the mesh files the frame is `origin + ijk · spacing` (no direction matrix).

## 5. Design

### 5.1 Pipeline at inference

1. Healthy host vessel (segmentation) -> surface mesh -> `extract_skeleton` -> 1D graph.
2. **Ostium model:** 1D graph -> one or more ostia (site + shape).
3. **Sac generator:** a sac attached to the generated ostium (§5.4).
4. Voxelise the sac and take the union with the host mask (§5.5).

### 5.2 Ostium: ground truth, representation, and the soft constraint

Code: `geometry/ostium.py` (`fitted_ostium`, `encode_ostium`, `decode_ostium`, `flip_code`).
Experiment: `vessel_edit/exp_ostium_fourier.py`, with results in
`runtime_skeleton/ostium_fourier/`.

#### 5.2.1 Ground truth

1. **Place the fit in image space.** The case's GHD fit is placed by inverting its Stage-1
   alignment: $V_{img} = (V_{fit} - t)\,R / s$, using R, s, t from `alignment_result.npy`.
   On 9 cases it lies 0.11–0.16 mm (median) from the big mesh.
2. **Dome.** The reviewed per-vertex dome label of the fit (`AneuG_morpho/<case>.npy :: manual_dome`),
   the same label the morphology sensor is trained on. The dome patch = faces whose three
   vertices are all dome, largest patch kept, holes filled. The sac cut out of the big mesh
   is this same patch (big-mesh faces lying on it), so ring, sac and patch share one
   definition.
3. **Raw rim.** The boundary of the dome patch, walked as one closed loop of fit
   vertices. It zigzags because a per-vertex label can only follow mesh edges.

#### 5.2.2 Representation: a Fourier curve anchored to the graph

The ostium is a closed curve in 3D. It is described in a small orthonormal coordinate system
attached to the 1D graph, as a sum of H = 5 Fourier harmonics (ellipses).

**Notation**

| symbol | meaning |
|---|---|
| $g_i \in \mathbb{R}^3,\ \rho_i$ | position and radius of graph point i (image space, mm) |
| $\gamma = \{\gamma_0, \dots, \gamma_{L-1}\}$ | the raw rim: an ordered closed loop of points (image space, mm) |
| M = 128 | number of resampled points on the ring |
| H = 5 | number of harmonics kept |
| $\sigma \in \{+1, -1\}$ | tangent direction choice |
| $u \in [0, 1)$ | position along the ring, as a fraction of its perimeter |

**A. The coordinate system of the ostium**

1. **Centre C.** Resample $\gamma$ uniformly by arc length into M points $r_0, \dots, r_{M-1}$
   (with $r_m$ at arc length $m/M$ of the perimeter). Then
   $$C = \frac{1}{M}\sum_{m=0}^{M-1} r_m .$$
2. **Site P.** The graph point nearest to the centre:
   $$P = \arg\min_i |g_i - C| .$$
3. **Graph tangent at P.** The graph is oriented away from its root. Take the point 3 steps
   back towards the root, $g_{P^-}$, and 3 steps forward, $g_{P^+}$ (forward only while the
   path continues uniquely; at a junction or a vessel end, $g_{P^+} = g_P$). Then
   $$\tau = \frac{g_{P^+} - g_{P^-}}{|g_{P^+} - g_{P^-}|}, \qquad t = \sigma\,\tau .$$
4. **Axes** (orthonormal and right-handed):
   $$n = \frac{C - P}{|C - P|}, \qquad e_1 = \frac{t - (t\cdot n)\,n}{|t - (t\cdot n)\,n|}, \qquad e_2 = n \times e_1 .$$
   - n points from the vessel towards the ostium.
   - $e_1$ runs along the vessel.
   - $e_2$ goes around it.

   Because P is the graph point nearest to C, n is already nearly perpendicular to τ (88–89°
   on all 9 cases), so $e_1 \approx t$.
5. **Scale.** The radius of the **local vessel**, not of the ostium:
   $$s = \operatorname{median}\{\rho_i : |g_i - g_P| < 3\ \text{mm}\} .$$
   It is known before the ostium exists (it is on the condition side), so the ostium's own
   size stays in the code.
6. **The transform.** With the rotation matrix built from the axes as rows,
   $$\mathbf{R} = \begin{pmatrix} e_1^\top \\ e_2^\top \\ n^\top \end{pmatrix} \in SO(3),$$
   - image → local: $x = \mathbf{R}\,(r - C)/s$;
   - local → image: $r = C + s\,\mathbf{R}^\top x$.

   This is an ordinary Euclidean coordinate system: moved to C, rotated to $(e_1, e_2, n)$,
   and measured in units of s. In it the ring is centred at the origin and has size about one.

**B. Encoding (ring → code)**

1. **Local coordinates of the raw rim:** $x_j = \mathbf{R}\,(\gamma_j - C)/s$, for j = 0 … L−1.
2. **Direction.** Use the signed area of the loop in the $e_1$–$e_2$ plane,
   $$A = \tfrac12 \sum_j \big(x_{j,1}\,x_{j+1,2} - x_{j+1,1}\,x_{j,2}\big) .$$
   If A < 0, reverse the loop. The ring then runs counter-clockwise about n, seen from the
   sac.
3. **Start.** Find the edge $(x_j, x_{j+1})$ that crosses the half-plane
   $\{x_2 = 0,\ x_1 > 0\}$ going counter-clockwise ($x_{j,2} < 0 \le x_{j+1,2}$). The start
   point is the crossing $x^\ast = x_j + \lambda (x_{j+1} - x_j)$, with
   $\lambda = -x_{j,2} / (x_{j+1,2} - x_{j,2})$. This is the rim point in the $e_1$ direction
   from C. Rotate the loop so that it begins at $x^\ast$.
4. **Resample** uniformly by arc length from $x^\ast$: $x(u_m)$, $u_m = m/M$, m = 0 … M−1.
5. **Fourier coefficients** (the discrete Fourier transform), for k = 1 … H:
   $$c_k = \frac{1}{M}\sum_{m=0}^{M-1} x(u_m)\, e^{-2\pi i k u_m} \in \mathbb{C}^3,
     \qquad a_k = 2\,\mathrm{Re}\,c_k, \qquad b_k = -2\,\mathrm{Im}\,c_k \qquad (a_k, b_k \in \mathbb{R}^3).$$
   For evenly spaced samples, this gives exactly the least-squares fit of the H-harmonic curve
   to the M points, because sines and cosines of different k are orthogonal. Nothing is
   optimised.
6. **Code:**
   - shape $\{a_k, b_k\}_{k=1}^{H}$: 6H = 30 numbers;
   - offset $(C - g_P)/s$: 3 numbers, in image axes;
   - the site P;
   - σ.

**C. Decoding (code → ring)**

1. $C = g_P + s \cdot \text{offset}$. Rebuild $n, e_1, e_2, \mathbf{R}$ exactly as in A.3–A.6,
   from C, the graph and σ.
2. The local curve is a sum of H ellipses:
   $$\hat x(u) = \sum_{k=1}^{H}\big(a_k \cos 2\pi k u + b_k \sin 2\pi k u\big), \qquad u \in [0, 1).$$
   - **One harmonic is an ellipse.** In the coordinates (along $a_k$, along $b_k$), the point
     $a_k\cos\theta + b_k\sin\theta$ is $(\cos\theta, \sin\theta)$, the unit circle. A linear
     map of a circle is an ellipse lying in the plane of $a_k$ and $b_k$; its semi-axes are
     the singular values of $[a_k\ b_k]$.
   - Harmonic k is traced k times per turn of u.
   - Harmonic 1 is the basic oval: size, elongation, tilt.
   - Harmonic 2 adds a two-fold wobble, such as the saddle where the neck wraps around a
     cylindrical vessel.
   - Harmonics 3–5 add finer detail.
3. Back to image space: $\hat r(u) = C + s\,\mathbf{R}^\top \hat x(u)$.

**Truncation is the smoothing.** Only $k \le H$ is kept, so the decoded ring is smooth, and
the mesh-edge zigzag of the raw rim (the higher harmonics) is not part of the target.

**D. Why the code is unique**

- The DFT is a fixed linear map from the M samples to the coefficients.
- Every other freedom of describing the same curve is fixed by a rule:
  - position (C);
  - rotation (n from P, $e_1$ from the graph tangent);
  - size (s);
  - speed (arc length);
  - direction (counter-clockwise about n);
  - start ($x^\ast$ on $+e_1$).
- Measured on 9 cases: shifting the raw loop's start or reversing it changes the
  coefficients by at most 1.2e-3 (resampling noise).

**E. Direction flip (training-time augmentation)**

At inference the inlet cannot be identified, so σ is arbitrary. At training time σ is drawn
at random each time a case is loaded, and the raw rim is encoded with that σ.
- $\sigma \to -\sigma$ gives $e_1 \to -e_1$, $e_2 \to -e_2$, n unchanged. That is a 180°
  rotation about n, not a mirror image, so counter-clockwise stays counter-clockwise.
- The start moves to the opposite crossing.
- **σ is given to the model as an input**, so the target is unambiguous and the model learns
  to place the same physical ring under either frame.
- Measured: encoding with −σ and decoding moves the ring by 0.01–0.04 mm (mean).

**F. What one ostium is, as the model sees it**

| item | type | role |
|---|---|---|
| P | index into the point graph | site; a distribution over graph points |
| offset = $(C - g_P)/s$ | 3 reals | centre (image axes, in units of s) |
| σ | ±1, **input** | which tangent direction the frame uses |
| $\{a_k, b_k\}_{k=1}^{5}$ | 30 reals | ring shape in the $(e_1, e_2, n)$ frame, in units of s |

Cost: encoding takes 1.7 ms per ostium (largest graph, 2,333 points), so the code is computed
on the fly in the dataset. Only the raw rim is stored.

#### 5.2.3 Soft constraint when generating the sac (model 2)

Model 2 generates GHD tokens conditioned on the graph and the ostium. Its shape goes through
the (frozen) morphology sensor, which gives each vertex a dome probability $p_i$. The
generated rim should match the conditioning ring. The rim is never extracted as a discrete
loop, so everything stays differentiable (option (a)):
- **Soft rim.** For every mesh edge (i, j):
  - crossing point $x_{ij} = x_i + \lambda(x_j - x_i)$, with
    $\lambda = \mathrm{clip}\!\big(\tfrac{0.5 - p_i}{p_j - p_i}, 0, 1\big)$;
  - boundary weight $w_{ij} = p_i(1-p_j) + p_j(1-p_i)$.
- **Target.** Dense samples $y_m = \hat r(u_m)$ of the conditioning ring. These are the
  already-smoothed H-harmonic curve, so the generated side needs no smoothing: a zigzag
  costs only about 0.2–0.3 mm.
- **Loss (weighted chamfer).**
  - Rim → target: $\sum w_{ij}\, d(x_{ij}, Y)^2 / \sum w_{ij}$.
  - Target → rim: $\frac{1}{M}\sum_m \min_{ij}\big(|y_m - x_{ij}|^2 + \beta(1 - w_{ij})\big)$,
    so that low-confidence edges cannot cover the target.

  It is differentiable through the vertex positions (the GHD tokens) and through the
  sensor probabilities. β and the weight of each term are to tune.

#### 5.2.4 Reconstruction experiment (9 cases)

`runtime_skeleton/ostium_fourier/`:
- `summary.txt` and `results.json`: the numbers;
- `error_vs_H.png`;
- `<case>_fourier.png`: the raw rim against H = 3, 4, 5 in three views.

| test | result |
|---|---|
| E1, H-harmonic ring vs raw rim (mean / max mm) | H=3: 0.08–0.20 / 0.32–0.64; **H=4: 0.08–0.18 / 0.27–0.57; H=5: 0.07–0.14 / 0.27–0.46**; H=8: 0.06–0.10 / 0.22–0.35 |
| E2, shift the start / reverse the loop | coefficients change ≤ 1.2e-3 (resampling only): canonical |
| E3, flip σ, then decode | ring moves 0.01–0.04 mm mean (max 0.26, Pztke); $e_1 \cdot e_1' = -1.00$ |
| E4, jitter the rim by σ_noise = 0.1 mm | ring moves 0.05–0.07 mm; $e_1$ rotates ≤ 5° (EerG 9°, next to a junction); P shifts by a few 0.1 mm-spaced graph points, which is harmless because the frame moves smoothly |

Most of the raw rim's zigzag is in the n direction (out of plane). H = 3–4 smooth it out,
and H = 5 starts to follow the larger wiggles (visible in 8dVI). **Chosen: H = 5** (30
numbers): it keeps the basic structure and stays smooth.

### 5.3 Training pairs: (1D graph, ostium)

From every real aneurysm: the graph from §4.2 and the real ostium. The ostium is the
boundary between sac faces and vessel faces on the mesh (the neck rim we already locate),
expressed in the graph's frame. **Next step: process the geometries to build these pairs.**
- Sources: ImperialNHS (368 folders: 159 bifurcation, 111 sidewall, 38 sidewall near a
  bifurcation, 60 type 3), then AneuX (323 cases).
- Shortcut risks to control:
  - **Radius smoothness** around the aneurysm (handled by §4.2 step 4).
  - **Patched-surface traces:** the graph does not see the surface, but positions near the
    sac come from the patched mesh.
  - **Crop extent:** the meshes are region-grown from the sac, so the sac sits near the
    centre of every crop. The model could learn "the ostium is in the middle". The graph
    should come from a crop that does not centre on the sac, for example the
    whole-vessel segmentation, or random crops.

### 5.4 The sac

Option B, kept from earlier: generate the AneuG complex and keep only its dome (morphology
sensor), placed on the generated ostium. With an explicit ostium, conditioning the
complex on the ostium (or a sac model conditioned on ostium + local graph) replaces the
earlier latent-inversion idea. To decide after the ostium model.

### 5.5 Merging without changing the vessel

Final mask = host vessel mask ∪ voxelised sac. The sac reaches slightly into the wall at
the neck, so the union has no gap, and must not grow back through the lumen. The aneurysm
label is the sac voxels outside the original vessel.

### 5.6 Model design (agreed)

Two generative models, sharing one graph encoder. Each module's input and output:

| # | module | input | output |
|---|---|---|---|
| 0 | **Data + augmentation** (training) | `vessel_graph.npz` (+ the GHD fit for model 2) | the trimmed, rotated graph and the targets below |
| 1 | **Graph encoder** | the point graph (§1 below) | per-point features H (N×d), coarse tokens, a global feature g |
| 2 | **Site head** (model 1a) | H, g | a distribution over graph points, p(site) |
| 3 | **Ostium head** (model 1b) | the features at P, g, the signed tangent t at P, the local radius s | the 33-number ostium code y → the decoded ring |
| 4 | **Ostium tokens** | y and the decoded ring | a token set describing the ostium |
| 5a | **Type head** (model 2) | graph tokens, ostium tokens (c2) | p(aneurysm type), 3 classes |
| 5b | **Complex head** (model 2) | c2 + an embedding of the type | x = [z (16), affine in the ostium frame] |
| 6 | **GHD-VAE decoder** (frozen) | z, type | GHD coefficients φ → complex mesh, placed via the affine and the ostium frame |
| 7 | **Morphology sensor** (frozen) | the placed complex | per-vertex dome probability → the sac (dome) |

**0. Data and augmentation** (on the fly, per sample)
1. Load the graph and the raw rim.
2. **Trim off-centre.** Pick a crop centre Q on the graph, near the site but not at it, and a random crop size, from local up to the whole graph. Make sure every graph point within **6 mm (Euclidean) of the ostium** is kept. Keep the piece connected to Q.
3. **Rotate** by a random SO(3) rotation: graph, rim and the GHD placement together.
4. Draw **σ = ±1**.
5. **Jitter the site:** P' = P plus a very small move along the graph.
6. **Re-encode** the rim at P' with the tangent σ·τ, giving the target y (33 numbers).
7. **Normalise:** centre on the crop's centroid; positions in units of 10 mm.
8. For model 2: z_gt (the VAE encoder's mean of the case's fitted φ) and the affine_gt of the fit relative to the ostium frame.
   - Because that affine is relative to the ostium frame, it is unchanged by the global rotation.
   - Under a σ flip it changes by the known 180° turn about n.

**1. Graph encoder**
- **Per-point inputs:**
  - position;
  - radius;
  - the sign-free tangent (t ⊗ t);
  - the junction and end flags.

  Not inputs: `branch_id`, the aneurysm mask, `geo_dist_from_site`.
- **Level 0** (all points, 0.1 mm): graph convolutions along the edges → F0.
- **Level 1** (farthest-point sampling at ~1 mm, pooling along the graph): positional encodings (Laplacian eigenvectors **plus their eigenvalues**, random-walk encodings; random eigenvector sign flips) → GPS blocks (local message passing + full attention) → F1.
- **Level 2** (~4 mm): the same → F2. Coarse-level edges connect sampled points that are neighbours along the original graph.
- **Upsampling:** F2 → F1 → F0 by interpolation + skip connection + graph convolution → H (N×d).
- **Global:** a pooled feature **g** (d), pooled over the coarsest level's tokens (attention
  pooling, or mean + max). Where H describes each point's surroundings, g summarises the whole
  cropped graph (vessel calibre, number of bifurcations, branch length and curvature), so
  decisions at one point can use the whole tree.

**2. Site head (model 1a)**
- In: H, g. Out: one score per point → softmax → p(site).
- **Train:** cross-entropy against a Gaussian soft label (~1 mm along the graph) around the true P. This worked in the morphology sensor.
- **Inference:** sample P.

**3. Ostium head (model 1b): what it predicts**

| numbers | what | units / frame | what follows from it |
|---|---|---|---|
| 3 | offset (C − P)/s | local vessel radii, image axes | centre C = P + s·offset; distance from the centerline = its length; **n = offset/\|offset\|** (n is not a separate output) |
| 30 | Fourier $a_k, b_k$ (3D vectors), k = 1…5 | radii, in the frame (e1, e2, n) | the ring shape |

- The frame comes from n plus the input tangent t: $e_1$ = t minus its n part, $e_2 = n \times e_1$.
- Ring: $\hat r(u) = C + s\,\mathbf{R}^\top \sum_k (a_k\cos 2\pi k u + b_k \sin 2\pi k u)$.
- In the training data n is perpendicular to the vessel (88–89°), because P is the graph
  point nearest C. A generated offset may tilt slightly along the tangent; the decoder has an
  option to remove that tangential part.
- The site P comes from the site head, not from this head.

**Ostium head details**
- **In:** H[P], g, the **signed tangent t = σ·τ at P** (it fixes the frame direction), s.
- **Out:** y = [(C − P)/s (3), Fourier {a_k, b_k}, k = 1…5 (30)], standardised per dimension.
- **Generator:** conditional **flow matching**, a velocity network v_θ(y_t, t, cond).
  - Train: $\|v_\theta(y_t, t, c) - (y_1 - y_0)\|^2$, with $y_t = (1-t)y_0 + t y_1$.
  - Sample: integrate dy/dt = v_θ from noise.
- **Option, as an ablation:** a one-shot MLP regressor (mean-squared error).
- **Train:** the true P with a very small jitter (teacher forcing).
- **Decode:** C = P + s·offset → frame (n, e1, e2) from P, t, C → ring points.

**4. Ostium tokens**
- The decoded ring (e.g. 32 points, expressed in the ostium frame / s).
- One token embedding the full code y.

**Model 2 target, written out.** The placed fit is
$V_{img} = k \cdot e^{\ell} \cdot V_{can}(\phi)\,Q + b$, where:
- $V_{can}$ is the GHD reconstruction of φ in normalised canonical space;
- $e^{\ell}$ is the fit's scale, generated by the VAE together with φ;
- $Q = R_w^\top R_{al}$;
- $k = \text{norm\_canonical}/s_{al}$;
- $b = (t_{vec}\cdot\text{norm\_canonical} - t_{al})\,R_{al}/s_{al}$.

In the ostium frame (C, $\mathbf{R}$, s):
$V_{loc} = \mathbf{R}(V_{img} - C)/s = (k e^{\ell}/s)\, V_{can}\,(Q\mathbf{R}^\top) + \mathbf{R}(b - C)/s$.

So the affine is:
- rotation $Q\mathbf{R}^\top$, as a 6D vector (two columns; Gram–Schmidt when decoding);
- log-scale $\log(k/s)$;
- translation $\mathbf{R}(b - C)/s$.

That is 10 numbers, plus z (16): **x is 26 numbers**. It is unchanged by the global
rotation; a σ flip or the site jitter changes $\mathbf{R}$ and is recomputed.

**5. Type head and complex head (model 2)**
- The aneurysm type is a **generative target** too. Sampling order: site → ostium → **type** → [z, affine].
- **Shared condition c2:** graph tokens near the site and the global feature; the ostium tokens (by cross-attention).
- **Type head:** c2 → p(type) (3 classes). Train: cross-entropy. Inference: **sample** the type from the softmax (not argmax).
- **Complex head:** c2 + emb(type) → x = [z (16), affine (rotation, log-scale, translation)], **generated jointly by one flow**, so every sampled shape comes with a matching pose.
  - It must be conditioned on the type: the Stage-1 VAE is type-conditional (`decode(z, type)`, per-type `norm_can`), so z means a different shape under each type. Sampling the type and z independently would give mismatched pairs.
  - Train: teacher-forced with the true type (as the ostium is). Inference: the sampled type.
- **Option:** a one-shot MLP.

**6. Placement**
- z → frozen GHD-VAE decoder → φ (144×3) → GHD reconstruction → mesh in the canonical space.
- Apply the affine, then the ostium frame (C, R, s), to get the complex in image space.

**7. Sac**
- The placed complex goes through the frozen morphology sensor, which gives each vertex a dome probability.
- The dome patch is the sac. It is voxelised and merged by union with the host mask (§5.5).

**Losses**

| model | loss |
|---|---|
| 1 | site cross-entropy + flow-matching loss on y (standardised per dimension) |
| 1, auxiliary | **ring point loss** (`--w_ring`, 0.1): the predicted and true codes decoded to 64 ring points relative to P, in units of s; point i against point i (both start on +e1, counter-clockwise); mean squared distance. Needed because the standardised loss weights all 33 numbers equally, while one standardised unit of offset moves the ring ~1.2 mm and one unit of a k = 3–5 harmonic only ~0.03–0.07 mm |
| 2 | type cross-entropy + flow-matching loss on [z, affine] given the true type |
| 2, auxiliary, **low weight** | rim chamfer: the sensor's soft rim (edge crossings p = 0.5, weighted by w) against the conditioning ring (the smooth H = 5 curve) |
| 2, auxiliary, **low weight** | optional geometric loss against the placed ground-truth fit, vertex by vertex (fixed GHD topology), on the dome or the whole mesh |

The auxiliary losses are computed on the **one-step estimate** $\hat x_1 = x_t + (1-t)\,v_\theta(x_t,t,c)$ (and $\hat y_1$ for the ring loss), weighted by t. In MLP mode they apply to the direct prediction.

**Graph-token positions** (`--norm_tokens`, on by default): each of the 32 graph tokens of the complex head gets its position relative to the ostium, rel = R(p − C)/s, as [rel/|rel| (3), log(1 + |rel|) (1)] instead of the raw rel and |rel| (raw |rel| averages ~8 and reaches ~80, against ~1 for every other input). The ostium head's log(s) input is centred, (log s + 2.1)/0.3. `--norm_tokens 0` restores the raw inputs, for ablation.

**Inference:**
1. Artery mask → mesh → `extract_skeleton` → graph.
2. Encoder.
3. Sample P.
4. Sample y → ring.
5. Sample the type, then [z, affine] given the type → complex (decoded with that type).
6. Sensor → dome.
7. Voxelise the dome and merge it by union with the host mask (in scan space).

### 5.7 Code

| file | what |
|---|---|
| `geometry/extract_skeleton.py` | mesh + endpoints -> 1D graph |
| `geometry/heal_vessel.py` | cut the sac (from the GHD fit) and patch the vessel |
| `geometry/ostium.py` | placed fit, dome rim, ostium Fourier code (encode / decode, explicit site, flip) |
| `geometry/complex_target.py` | a fit as [z, affine in the ostium frame] and back |
| `dataset/preprocess_vessel_graph.py` | per case -> `assembled/<ds>/<case>/vessel_graph/vessel_graph.npz` |
| `dataset/run_preprocess_vessel_graph.sh` | the full run (N workers, memory gate) |
| `dataset/ostium_dataset.py` | the training dataset: trimming, rotation, sigma, site jitter, coarse levels + PE, targets |
| `models/ostium_model.py` | encoder, site / ostium / complex heads, flow matching (or MLP) |
| `scripts/train/train_ostium_model.py` | training (never GPU 2) |
| `scripts/ostium/sanity_data.py`, `scripts/ostium/sanity_model.py` | data and model sanity checks -> `runtime_debug/ostium_model/` |
| `scripts/ostium/visualize_predictions.py` | prediction renders for checking by eye |

## 6. Requirements from the image-generation side

- **Output:** the binary vessel mask (vessel ∪ aneurysm) in the host scan's geometry, and
  optionally an aneurysm channel (label 2).
- **The vessel mask outside the sac must be the host's own** GT or nnU-Net mask.
- **Domain gap:** our sacs come from voxelised surfaces (smoother, different partial
  volume) while the image model trains on manual labels. Agree who handles it.
- **Volume:** many varied sacs per healthy scan, at plausible sites. This is what the
  ostium model provides.

## 7. Evaluation (sketch)

- **Ostium model:** site distribution against real aneurysm locations (bifurcation versus
  sidewall rates, distance to the nearest bifurcation, vessel radius at the site), ostium
  size and shape distributions, and held-out likelihood of real ostia.
- **Vessel preservation:** zero changed voxels outside the sac, by construction.
- **Sac realism:** MMD / coverage / 1-NNA / Fréchet against real sacs, morphology
  markers, neck geometry.
- **Downstream:** image realism and nnU-Net mask fidelity on the synthesised images;
  whether synthetic aneurysms improve detection or segmentation.

## 8. Open questions

1. ~~Number of Fourier harmonics H~~ **Decided: H = 5** (§5.2.4).
2. Ostium model family: site distribution over graph points + offset + Fourier code given σ.
3. Graph crop: region-grown local crop (sac-centred, a shortcut) or whole-vessel graph?
4. The leftover radius cue (§4.2): add jitter to every radius at training time, use decoy
   interpolations at random non-ostium sites, or both?
5. Output labels: binary vessel mask only, or vessel + aneurysm?
6. Branch name and scope.
