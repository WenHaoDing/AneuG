# Methodology: vessel-conditioned synthesis of intracranial aneurysms on healthy vessels

For the agent writing the paper. This describes **how the method works, from raw data to the
final voxel mask**, so it can be written up as a methods section. It is not a user guide
(for that, see `HANDOFF_synthetic_aneurysm_masks.md`). Every number below comes from the code
or from measurements on our data. The file paths in the appendix let you check any detail.

Status as of 14 Nov 2026. Several model variants were trained as ablations (§9). Which one
is reported as "the" model is the project lead's decision. §8 describes the configuration
trained most recently.

---

## 0. Overview

**Goal.** Given a healthy cerebral artery segmentation from a real scan (no aneurysm), add a
realistic aneurysm sac at a plausible site and return a voxel label map in the scan's own
grid:
- 0: background;
- 1: the artery, **unchanged to the voxel**;
- 2: the generated aneurysm.

These masks condition a separate 3D image-synthesis model (diffusion inpainting of TOF-MRA
patches), which turns them into synthetic scans with matching labels. The method is
therefore **mask-level, anatomy-preserving aneurysm synthesis**.

**Key idea.** The parent vessel is never generated: it is taken from the scan. The
generative model is conditioned on the **1D skeleton of the local vessel tree**: centerline
points every 0.1 mm, each with a radius, plus their connectivity. From this graph it
generates, in order:
1. a **site** on the graph;
2. the **ostium** (neck ring) at that site;
3. the **aneurysm type**;
4. the **shape and pose** of an aneurysm–vessel complex, in a frame anchored to the ostium.

A frozen decoder chain turns the complex into a mesh. A frozen morphology sensor cuts the
dome out of it. The dome is attached to the real artery wall and voxelised.

```
healthy artery mask ──► surface ──► 1D vessel graph (centerline + radius)
                                         │
                       ┌─────────────────┴──────────────────┐
                       │  Graph encoder (GINE + GPS, 3 levels)│
                       └─────────────────┬──────────────────┘
          site head ──► site P ──► ostium head ──► ostium code y (33) ──► ring
                                                      │
                     complex condition (tokens: ring, code, nearby graph)
                                                      │
                          type head ──► type ──► complex head ──► x = [z (16), affine (10)]
                                                      │
          frozen: GHD-VAE decoder ──► GHD reconstruction ──► placement in the ostium frame
                                                      │
          frozen: morphology sensor ──► per-vertex dome probability ──► dome patch
                                                      │
          attach to the artery wall, voxelise, union with the host mask ──► label map 0/1/2
```

Optional conditioning (morphology-conditioned variant, §6.6): the requested **dome volume**
and/or **aspect ratio**. Either can be left out.

---

## 1. Background components (from our earlier work, used frozen)

The method reuses three pretrained components from AneuG (`aneug_synva_methods_summary.md`
§1). None of them is trained here.

**1.1 Graph Harmonic Deformation (GHD).**
- An aneurysm–vessel *complex* (sac plus short stubs of the parent and branch vessels) is
  represented by deforming a canonical template mesh.
- There is one template per aneurysm type. Type 0 is bifurcation, type 1 sidewall, and
  type 2 sidewall near a bifurcation.
- The deformation is a combination of the lowest 144 eigenvectors of the template's
  cotangent Laplacian, with 3D coefficients: φ ∈ ℝ^{144×3}.
- A fixed template gives every reconstructed complex the same topology, so vertices
  correspond across shapes.
- Every real case has a GHD fit: φ, a scale e^ℓ, and a rigid alignment back to the image
  frame (Stage-1 alignment).

**1.2 Stage-1 GHD-VAE.**
- It is a type-conditional VAE over [φ, scale]: 432 + 1 numbers, standardised per
  dimension.
- Settings: hidden width 256, latent z ∈ ℝ^16, KL weight ×2.
- It also has a WGAN-GP critic, applied to shapes decoded from prior samples z ~ N(0, I)
  with weight 0.3, to improve sample diversity.
- `decode(z, type)` → (φ, scale). The latent of a real case is the encoder mean of its
  fitted φ.

**1.3 Morphology sensor.**
- It is a graph transformer on the GHD mesh (GPS, width 128, 4 layers). It is trained on
  reviewed per-vertex dome labels of 523 real cases, 23 of them brushed by hand.
- For a given (φ, type) it outputs:
  - a per-vertex **dome probability** p_i;
  - one **cap distribution** per vessel opening of the type, which locates the vessel
    ends (caps) of the complex.

**What is new in this work:** everything that conditions on the vessel graph:
- graph extraction;
- the ostium representation;
- the generative model;
- the self-supervised constraints;
- the morphology conditioning;
- the attachment to a real artery.

---

## 2. Data

**2.1 Sources.**
- Aneurysm cases from two collections:
  - ImperialNHS: in-house, pseudonymised;
  - AneuX: public.
- Each case provides:
  - a surface mesh of the region-grown artery around the aneurysm;
  - manually checked vessel endpoints (seeds for centerline extraction);
  - the GHD fit with its alignment;
  - the reviewed dome label (§1.3).

**2.2 Usable cases: 522.** 241 ImperialNHS and 281 AneuX; one AneuX case failed
preprocessing.

| type | ImperialNHS | AneuX | total |
|---|---|---|---|
| 0 bifurcation | 113 | 153 | 266 |
| 1 sidewall | 96 | 76 | 172 |
| 2 sidewall near a bifurcation | 32 | 52 | 84 |

**2.3 Split.**
- 15 validation cases, drawn at random **at patient level** (seed 0): 7 ImperialNHS and
  8 AneuX.
- Every case of a drawn patient is held out, so no vessel tree is shared with training.
  This leaves **507 training cases**.
- Patient keys:
  - ImperialNHS: the scan ID;
  - AneuX: the case name with its cut, side and ID suffixes removed.

**2.4 Mesh conventions.**
- The surfaces were made by marching cubes of the segmentation at level 0.7, then 25
  Laplacian iterations.
- Coordinates are mm in the frame `origin + ijk · spacing`, with no direction matrix.
- The 1D graph, the ostium and the generated meshes all live in this frame, so the output
  maps back onto the scan's voxels exactly.

---

## 3. Building the training pairs: (healthy-looking vessel graph, real ostium)

A real aneurysm case has a sac, but the model must be conditioned on a *healthy-looking*
vessel, as at inference. A healthy surface under the sac cannot be recovered:
- we tried hole filling and a centerline-guided rebuild, and both left visible traces;
- more fundamentally, an aneurysm deforms the vessel around it.

We therefore **condition on the skeleton, not the surface**. We assume the centerline of the
parent branches is not changed by the sac. Only the radius near the sac is unreliable, and
it is repaired.

**3.1 Ground-truth sac and ostium (per case).**
1. **Place the GHD fit in image space** by inverting its Stage-1 alignment:
   V_img = (V_fit − t) R / s. On 9 test cases the placed fit lies 0.11–0.16 mm (median) from
   the original mesh.
2. **Dome patch:** the fit's faces whose three vertices are labelled dome. Keep the largest
   patch and fill its holes.
3. **Raw rim:** the patch boundary, walked as one closed loop of fit vertices. It zigzags
   along mesh edges; the Fourier truncation in §4 removes this.
4. **The sac on the big mesh** is the set of big-mesh faces lying on this dome patch. Ring,
   sac and patch therefore share one definition.

**3.2 Removing the sac and patching the vessel** (offline only, 1–8 min per case).
1. **Endpoint filter:** drop endpoints that would seed the centerline on the sac. These are
   endpoints inside the closed dome or within 0.5 mm of it, or whose nearest mesh face is a
   sac face.
2. **Cut** the sac faces plus a 1 mm geodesic collar (neck bulge, rim defects). Keep the
   largest remaining island.
3. **Heal:** VMTK centerlines of the original mesh bridge the gap. Tubes swept along them
   are blended with the cut surface ("(original − zone) ∪ tubes", implicit blend, then
   marching cubes). Without this, the centerline would be pulled into the sac.

**3.3 Skeleton extraction** (`extract_skeleton`, the same code used at inference).
1. **VMTK centerlines** from one endpoint (the source) to every other endpoint. They are
   resampled to 0.1 mm and smoothed (100 iterations × 0.1).
2. **Radius:** VMTK's maximum inscribed sphere radius.
3. **Topology** from VMTK's branch extractor (line and tract order, no distance tolerance):
   - The short bifurcation pieces are **kept**, so the graph is one connected tree.
   - Where two VMTK lines meet a node a few tenths of a mm apart, the line is cut back and
     the gap (≤ 0.54 mm on 9 cases) is filled at 0.1 mm spacing.
4. **Position smoothing over the graph:** σ = 0.5 mm, with vessel ends fixed. This rounds
   the corners where lines meet: the sharpest bend over 0.3 mm drops from 87–120° to
   15–23°.
5. **Radius repair across the aneurysm (training only).** The aneurysm region is:
   - the centerline points whose inscribed sphere comes within tolerance of the removed sac
     and collar;
   - plus the points the sac projects onto (within 2.5 r);
   - with gaps shorter than r filled;
   - widened by 0.75 r.

   Its radius is replaced by a **C¹ Hermite curve** between the healthy anchors (slopes from
   a 1 mm line fit). Before the repair, the inscribed sphere grows into the sac, up to 2.3×.
6. **Radius smoothing over the whole graph:** diffusion with σ = 1 mm, continuous through
   the nodes. This removes segmentation noise and makes the repaired stretch less
   conspicuous. Roughness outside the region falls from 10–19 µm to 2.5–7.6 µm.

**3.4 The point graph (the model's input).**
- Every 0.1 mm point is a node, and edges join neighbours along the centerline.
- Graphs have 237–3864 points (median 1149).
- The model reads only **positions (mm), radii (mm) and edges**. Everything else is derived
  from them.
- Labels that would reveal the answer are stored for diagnostics only and never given to
  the model: the aneurysm region, branch IDs, and the distance from the site.

**3.5 The site.**
- The ground-truth site P is the graph point nearest the ostium centre C (the mean of the
  arc-length-resampled rim).
- The ostium normal n = (C − P)/|C − P| is then nearly perpendicular to the centerline: 88–89°
  on 9 cases.

---

## 4. Ostium representation: a Fourier curve in a graph-anchored frame

The ostium is a closed 3D curve. It is encoded as a fixed-length vector in a coordinate
frame attached to the graph, so the target is invariant to global position and rotation, and
canonical (unique).

**4.1 Frame at a site P** (given the ring centre C and a tangent sign σ = ±1).
- **Tangent.** τ is the central difference over 3 graph points back (towards the root, the
  first degree-1 point) and 3 forward. The forward points are used only while the path
  continues uniquely; at a junction or a vessel end only the backward direction is used.
  The signed tangent is t = σ τ.
- **Axes.**
  - n = (C − P)/|C − P|;
  - e₁ = normalise(t − (t·n) n);
  - e₂ = n × e₁.

  Let **R** be the rotation matrix with rows (e₁, e₂, n).
- **Scale.** s = the median radius of the graph points within 3 mm of P. This is the
  *local vessel* radius, known before the ostium exists, so the ostium's own size stays in
  the code.
- **Transforms.** Image → local: x = R (r − C)/s. Local → image: r = C + s Rᵀ x.

**4.2 Encoding.**
1. Map the raw rim to local coordinates.
2. Orient it counter-clockwise about n, using the signed area in the e₁–e₂ plane.
3. Start it at the crossing of the half-plane {x₂ = 0, x₁ > 0}, i.e. the rim point in the
   +e₁ direction.
4. Resample it uniformly by arc length (M = 128).
5. Keep the first H = 5 discrete Fourier harmonics:

   c_k = (1/M) Σ_m x(u_m) e^{−2πi k u_m}, a_k = 2 Re c_k, b_k = −2 Im c_k (a_k, b_k ∈ ℝ³).

   For uniform samples this is the exact least-squares fit of an H-harmonic curve; nothing
   is optimised.

**Code y ∈ ℝ³³:**
- the offset (C − P)/s (3 numbers, image axes);
- {a_k, b_k}, k = 1…5 (30 numbers, in the (e₁, e₂, n) frame, units of s).

The site P and the sign σ go with the code but are not part of it.

**4.3 Decoding.**
1. C = P + s · offset; n is the direction of the offset.
2. The frame follows from n and the input tangent t.
3. The ring is r̂(u) = C + s Rᵀ Σ_k (a_k cos 2πku + b_k sin 2πku).
4. Each harmonic is an ellipse:
   - k = 1 is the basic oval (size, elongation, tilt);
   - k = 2 is the saddle where the neck wraps around a cylinder;
   - k = 3–5 add finer detail.

**Truncation is the smoothing:** the mesh-edge zigzag of the raw rim lives in the higher
harmonics and is not part of the target.

**4.4 Validation of the representation** (9 cases):
- **Reconstruction error vs the raw rim** (mean / max mm):

  | H | mean | max |
  |---|---|---|
  | 3 | 0.08–0.20 | 0.32–0.64 |
  | 4 | 0.08–0.18 | 0.27–0.57 |
  | **5 (chosen)** | **0.07–0.14** | **0.27–0.46** |
  | 8 | 0.06–0.10 | 0.22–0.35 |
- **Canonical:** shifting the start or reversing the loop changes the coefficients by
  ≤ 1.2e-3.
- **Flipping σ, then decoding** moves the ring by 0.01–0.04 mm (mean).
- **Rim jitter of 0.1 mm** moves the ring by 0.05–0.07 mm and rotates e₁ by ≤ 5°.
- **Cost:** encoding takes 1.7 ms per ostium, so codes are computed on the fly during
  augmentation.

**4.5 Tangent sign.** At inference the flow direction is unknown, so σ is drawn at random:
- In training, σ is random per sample and the rim is re-encoded with it.
- σ → −σ is a 180° rotation about n (not a mirror), so orientation is preserved.
- The signed tangent is a model **input**, so the target is unambiguous.

---

## 5. Training-time data pipeline (per sample, on the fly)

1. **Off-centre crop.** The training meshes are region-grown from the sac, so the sac sits
   near the centre of every whole graph: a shortcut. To control it:
   - With probability 0.2, the whole graph is kept.
   - Otherwise:
     - draw a crop radius R, log-uniform in [8 mm, the graph's extent from the site];
     - draw a crop centre Q at graph distance U(0, 0.8 R) from the site;
     - keep the geodesic ball around Q.
   - Every graph point within 6 mm (Euclidean) of the rim and within 15 mm along the graph
     of the site is always kept; the ball grows to include them. The crop is therefore one
     connected piece.
2. **Rotation.** The graph, rim and GHD placement are rotated together. Two settings are
   available:
   - a uniformly random SO(3) rotation;
   - (used in recent runs) a rotation about a random axis by an angle U(0°, 15°).

   A no-rotation variant is an ablation.
3. **Tangent sign** σ = ±1 at random.
4. **Site jitter (teacher forcing with a slightly-off site).** P′ is a random graph point
   within 0.3 mm along the graph of the true P. The rim is re-encoded at P′, giving the
   target y.
5. **Normalisation.** Centre on the crop's centroid; positions and radii in units of 10 mm.
6. **Site label.** A Gaussian over graph distance from the true P, σ_site = 1 mm,
   normalised.
7. **Complex target x ∈ ℝ²⁶.** The placed fit is V_img = k e^ℓ V_can(φ) Q + b. In the
   ostium frame (C, R, s) this becomes:

   V_loc = (k e^ℓ / s) V_can (Q Rᵀ) + R (b − C)/s.

   The **affine** (10 numbers) is:
   - the rotation Q Rᵀ as a 6D vector (two columns, Gram–Schmidt on decoding);
   - log(k/s);
   - the translation R (b − C)/s.

   **z** (16) is the VAE encoder mean of the case's φ. Because the affine is relative to the
   ostium frame, it is unchanged by the global rotation. It is recomputed after the σ flip
   and the site jitter, which change R.
8. **Graph inputs** (§6.1), including the coarse levels and positional encodings.
9. **Target standardisation.** y and x are standardised per dimension, with statistics from
   two augmented passes over the training set.

In validation mode none of these augmentations are applied: whole graph, no rotation,
σ = +1, no jitter.

---

## 6. Model

The width is d = 128 throughout, for 2.87 M trainable parameters (encoder 1.19 M). There are
two generator backbones, an ablation axis: **conditional flow matching** or a
**deterministic MLP** (§6.5).

**6.1 Inputs.**
- **Node features (12):**
  - position (3);
  - radius (1);
  - the sign-free tangent as the 6 unique entries of t ⊗ t;
  - a junction flag (degree ≥ 3);
  - an end flag (degree 1).
- **Edge features (4):** the displacement vector and its length.
- **Coarse levels:**
  - farthest-point sampling to about 1 mm (level 1) and about 4 mm (level 2);
  - every finer point is assigned to its nearest seed **along the graph** (a geodesic
    Voronoi partition), which is used both to pool down and to broadcast up;
  - coarse edges join seeds whose regions touch.
- **Positional encodings per coarse level** (recomputed for every crop):
  - 8 Laplacian eigenvectors with random sign flips in training;
  - their 8 eigenvalues;
  - 8 random-walk return probabilities.

**6.2 Graph encoder (U-shaped, 3 levels).**
- **Level 0** (all points): input MLP, then 3 GINE blocks (residual + LayerNorm), giving F₀.
- **Levels 1 and 2:**
  - mean-pool the finer features into the seeds;
  - MLP([feature, position]) + MLP(positional encodings);
  - 2 GPS layers each (local GINE message passing plus 4-head full attention over the
    seeds of the graph).

  This gives F₁ and F₂.
- **Up path** (2 → 1 → 0): broadcast the coarse features back along the geodesic
  assignment, concatenate the skip features, apply an MLP and 2 GINE blocks. This gives the
  per-point features **H** (N × d).
- **Global feature g:** MLP of [attention pooling, mean pooling] over level 2. It summarises
  the whole crop: calibre, bifurcations, length and curvature.

**6.3 Site head.**
- logit_i = MLP([H_i, g]), with a softmax over each graph's points.
- **Loss:** soft cross-entropy against the Gaussian site label.

**6.4 Ostium head.**
- **Condition:** c₁ = MLP([H[P′], g, t, (log s − μ)/σ]), with μ = −2.1 and σ = 0.3 over the
  dataset (s in units of 10 mm). The signed tangent t fixes the frame direction.
- **Output:** the 33-number code y (standardised).
- **Teacher forcing:** in training it sees the true site, with the 0.3 mm jitter.

**6.5 Generator networks.** The ostium and complex heads use the same network type:
- a residual MLP, width 256, 4 pre-norm residual blocks;
- a 32-dimensional sinusoidal time embedding.

There are two modes:
- **Flow matching:**
  - noise y₀ ~ N(0, I);
  - path y_t = (1 − t) y₀ + t y₁, with t ~ U(0, 1);
  - loss ‖v_θ(y_t, t, c) − (y₁ − y₀)‖²;
  - sampling by 32 Euler steps from noise.

  The auxiliary losses (§7) use the one-step estimate ŷ₁ = y_t + (1 − t) v_θ, weighted by t.
- **MLP:**
  - direct regression ŷ = f(c), with mean-squared error;
  - auxiliary losses on the prediction, with weight 1.

  Given the condition, the output is **deterministic**. Sample variety then comes from:
  - the sampled site;
  - the sampled type;
  - the random crop and σ at inference;
  - in the morphology-conditioned variant, the requested condition.

**6.6 Complex head and type head.**
- **Token set** for a 2-layer pre-norm transformer encoder (4 heads, feed-forward 2d), with
  a learned token-kind embedding added to each token:
  - a learned CLS token;
  - one **code token**, MLP(y);
  - 32 **ring tokens**, MLP([ring point in the ostium frame / s, cos 2πu, sin 2πu]), for 32
    points of the decoded ring;
  - 32 **graph tokens**: the 32 level-1 seeds nearest to C. Each is MLP([F₁, rel/|rel|,
    log(1 + |rel|)]), with rel = R (p − C)/s.

    Raw distances reach about 80 s, which is why the direction-plus-log form is used; the
    raw form is an ablation.
- **c₂** = MLP([CLS output, g]).
- **Type head:** MLP(c₂) → p(type) over 3 classes. Loss: cross-entropy. At inference the
  type is **sampled** from the softmax.
- **Complex head:** the generator (§6.5) on c₂ + emb(type) → x = [z (16), affine (10)].
  - It is conditioned on the type because the VAE decoder is type-conditional (z means a
    different shape under each type).
  - In training it gets the true type (teacher forcing).
  - Sampling order: **site → ostium → type → [z, affine]**.

**6.7 Placement (frozen, differentiable).**
1. z → VAE decoder (with the type) → φ, scale e^ℓ.
2. GHD reconstruction → V_can.
3. Affine → V_loc = e^{ℓ_aff} e^ℓ V_can M + translation (units of s).
4. Image space: V = C + s Rᵀ V_loc.
5. The morphology sensor on (φ, type) gives p_i and the cap distributions. The **dome
   patch** is p > 0.5.

**6.8 Morphology-conditioned variant.**
- **Condition:** m = [log dome volume (mm³), aspect ratio].
  - The aspect ratio is dome height / **mean neck radius**, i.e. twice the clinical
    height / neck diameter.
  - m is standardised with the training statistics: log V 3.99 ± 0.98 (median volume
    52.6 mm³) and AR 2.22 ± 0.76.
- **How m enters:** it is embedded by two small MLPs and **added** to:
  - c₁ (so the neck can follow the requested size);
  - c₂ (so the type and the dome follow it).
- **Labels:** the markers of each real case's fitted complex, measured with the same dome
  marker calculator used in training (below).
- **Marker calculator** (differentiable in the vertices and in p):
  - The dome is cut exactly at p = 0.5 on the piecewise-linear mesh, and closed by a fan
    from the neck centre o (the mean of the crossing points).
  - **Volume:** Σ det(a − o, b − o, c − o)/6 over the dome (sub)faces.
  - **Neck normal:** the direction of the dome's total area vector.
  - **Height:** a smooth maximum of (v − o)·n over the dome points.
  - **Neck radius:** the mean distance of the crossing points to o.
- **VAE floor:** decoding a real case's own z changes its markers by a median of 5% in
  volume (90th percentile 31%) and 0.05 in AR. This bounds the attainable accuracy.
- **Optional conditions (condition dropout, p = 0.2).** In training, each component of m is
  independently replaced by "not given" with probability 0.2:
  - its standardised input is set to 0, the training mean;
  - it is excluded from the morph loss.

  At inference the user can therefore give the volume only, the AR only, or neither.
  A request for exactly the mean value is indistinguishable from leaving it out.

---

## 7. Training objective

**7.1 Supervised terms at the true site** (teacher-forced).

| term | definition | weight (§8 config) |
|---|---|---|
| site | soft cross-entropy (§6.3) | 1 |
| ost | flow-matching or MSE loss on the standardised y | 1 |
| ring | rings decoded from the predicted and true codes (64 points each, relative to the site, units of s), compared point to point; mean squared distance. Needed because the standardised loss weights the 33 numbers equally, while one standardised unit of offset moves the ring ~1.2 mm and one unit of a k = 3–5 harmonic only ~0.03–0.07 mm | 0.1 |
| cx | flow-matching or MSE loss on the standardised x, given the true type | 1 |
| type | cross-entropy | 0.1 |
| geo | mean vertex distance between the decoded predicted complex and the true fit (same GHD topology), in the ostium frame | 0.01 |
| rim | soft-rim chamfer (below) between the sensor rim of the predicted complex and the true smooth ring | 0.01 |
| cap | each cap centre of the predicted complex (cap distribution × vertices) to its nearest graph point, squared, units of s | 0.2 |
| tube | each parent-vessel vertex of the predicted complex (p < 0.2, outside any cap region) to the graph's tube wall, min_i(\|v − c_i\| − r_i), squared, units of s | 0.6 |
| morph | Huber loss on the standardised [log V, AR] of the predicted dome against the case's real m | 0.1 |

Cap and tube terms ignore anything whose nearest graph point is a graph end: beyond a crop
cut the vessel is simply missing. They tie the generated vessel stubs to the actual vessel
tree: the stubs should lie on the graph's tubes and the caps on its centerlines.

**Soft rim** (differentiable; the rim is never extracted as a loop). For each mesh edge
(i, j):
- crossing point x_ij = x_i + λ (x_j − x_i), with λ = clip((0.5 − p_i)/(p_j − p_i), 0, 1);
- weight w_ij = p_i (1 − p_j) + p_j (1 − p_i).

The chamfer against the target ring points Y is:

  Σ w_ij d(x_ij, Y)² / Σ w_ij + (1/|Y|) Σ_m min_ij (|y_m − x_ij|² + 0.25 (1 − w_ij)).

The penalty in the second term keeps low-confidence edges from covering the target.

**7.2 Self-supervised terms at sampled sites.**

The supervised terms only ever see the one true site per case. At inference the model
generates anywhere on a vessel. So at every step, one extra site per graph is drawn:
- **Site sampling** ("mix"): uniformly over the points at least 5 mm from any graph end for
  half of the graphs in a batch; from the site head's own distribution for the other half.
- **At that site the model generates:**
  - the ostium (no gradient);
  - the type (sampled);
  - the complex.

  For the complex:
  - MLP mode: the prediction.
  - Flow mode: the full 32-step path is run without gradient, then the one-step estimate is
    taken from a random step k with gradient, weighted by t_k.
- **Losses**, which need no ground truth:

  | term | definition | weight |
  |---|---|---|
  | selfrim | soft-rim chamfer between the generated dome's sensor rim and the **generated** ring (detached): the dome must be built on the ostium it was given | 0.2 |
  | cap_s, tube_s | the cap and tube terms of §7.1 for the generated complex, placed in the generated ostium frame | 0.2, 0.6 (shared weights with cap, tube) |
  | morph_s | morph loss of the generated dome against a **random target** m: a training case's m plus Gaussian jitter of 0.25 condition-stds, with the same condition dropout | 0.1 |

  The ostium frame and ring are detached here, so these terms train the encoder and the
  complex branch, not the ostium head.
- **morph_s makes the conditioning self-supervised.** The model must hit requested volumes
  and AR at arbitrary sites, not only reproduce each case's own markers.
- **Morph-loss sidedness** is an ablation:
  - two-sided (penalise both too small and too large);
  - one-sided (penalise only too small, so a condition acts as a lower bound).

**Total loss:** the weighted sum of all terms, with w_cap and w_tube applied to (cap + cap_s)
and (tube + tube_s).

**7.3 Optimisation.**
- **Optimiser:** AdamW, weight decay 1e-4, gradient-norm clipping at 1.0.
- **Batch:** 32 graphs.
- **Learning rate:** one-cycle schedule, 5% warm-up then cosine decay, peak 6e-4.
- **Main runs:** 1000 epochs from scratch.
- **Fine-tunes:** loaded weights with a fresh optimiser and schedule, typically 500 epochs at
  a peak of 3e-4.
- **Encoder learning:** the encoder is trained end-to-end by all terms. The VAE decoder, GHD
  and sensor are frozen, but gradients pass through them.

---

## 8. Configuration of the most recent model

This is `abl_mlp_crop_rot15_selfrim_graph_mix_morph_w2_drop`, finished on 2 Nov 2026:
- **Backbone:** MLP heads, d = 128.
- **Augmentation:** off-centre crops; rotation ≤ 15° about a random axis; random σ; site
  jitter of 0.3 mm.
- **Supervised losses (§7.1):** weights as listed.
- **Self-supervised terms:** selfrim 0.2, cap/tube 0.2/0.6, mixed site sampling.
- **Morphology:** conditioning on, condition dropout 0.2, **two-sided** morph loss.
- **Training:** in three stages, each later stage starting from the previous weights with a
  fresh optimiser and schedule:
  1. 1000 epochs from scratch (`..._morph`), with selfrim/cap/tube at 0.1/0.1/0.3;
  2. 500 epochs at lr 3e-4 with those weights doubled to 0.2/0.2/0.6 (`..._morph_w2`);
  3. 500 epochs at lr 3e-4 adding condition dropout (`..._morph_w2_drop`).

A flow-matching counterpart with the same configuration was trained for 875 epochs, without
condition dropout, and is unfinished. Confirm the configuration to report with the project
lead.

---

## 9. Inference: from a healthy patch to a label map

**9.1 Input.** A healthy artery mask (ground truth or nnU-Net), typically a 96³ patch at
0.5 mm exported by the image-generation pipeline, with:
- its origin and spacing;
- a **core** box, the region the image model inpaints, where the aneurysm must lie.

The mask is turned into a 1D graph by the same steps as §3.3, with no radius repair:
marching cubes at 0.7 with 25 Laplacian iterations, VMTK, smoothing. The graph extraction
runs in the image-generation repository and must follow the contract in
`HANDOFF_inference_1d_graph.md`.

**9.2 Training-like crops.** Inference graphs are larger and branchier than training crops.
Every forward pass therefore runs on a crop drawn like a training crop around a point:
- The extent comes from the training graphs' own extents from their sites.
- The same p_whole, log-uniform radius and off-centre rule apply.
- Everything within 7 mm of the point is kept.

**9.3 Site selection.**
- **Site score per graph point:** the site head's probability on a crop, times the crop's
  point count (so uniform = 1). It is averaged over 24–32 crops, counting a point only when
  it lies ≥ 7 mm along the graph from a crop cut.
- **Sites are restricted** to graph points in the core and are drawn from score^{1/T}
  (temperature T).
- **Alternatives:**
  - **Prompt:** the graph point nearest a user point. Optionally, a site is drawn from
    p(site | prompt) ∝ p(site) · exp(−d²/2σ²), with d measured along the graph. This is the
    posterior a prompt-conditioned head would learn if trained with prompts within ~σ of the
    true site, so no retraining is needed.
  - **Uniform random:** a graph point away from the vessel ends.

**9.4 Generation at a site.**
1. Draw a training-like crop around the site and a random σ.
2. Ostium: y ~ ostium head, then the ring.
3. Type ~ softmax.
4. x ~ complex head given the type.
5. Decode the mesh and place it (§6.7).
6. The sensor gives the dome.
7. For the morphology variant: the requested m. A missing component is left out if the
   model was trained with dropout; otherwise it is drawn from the 25 real cases nearest in
   the given component.

**9.5 Attaching the dome to the real artery** (voxel space; the artery is never modified).
1. **Dome patch:** generated faces with p > 0.5, small holes filled. Its open boundary is
   the rim.
2. **Rim check:** the share of rim points inside the artery lumen is reported (a warning).
3. **Rim projection:** each rim point moves onto the artery wall along the ray from its
   nearest centerline point (near the site) through it, to where the ray leaves the artery.
   Rims floating off the wall move in, and rims inside the lumen move out. The sample is
   not built if the median move exceeds the rim radius, since the neck would change shape.
4. **Dome warp:** the rim displacement is spread over the dome, fading to zero at the tip.
5. **Close inwards:** the rim is capped by a cone to its centerline points, so the solid
   runs from the dome tip through the wall into the vessel.
6. **One structure:** F = artery ∪ solid.
   - Voxelise at 2× resolution and downsample by majority.
   - Apply a morphological closing within a 2-voxel band around the rim, so there is no
     crease or gap at the neck.

   This mirrors how the training masks were drawn: artery and aneurysm as one structure,
   with the aneurysm split off afterwards.
7. **Split:** label 2 = F minus the artery, inside the core only.
8. **Connectivity:** every label-2 piece must face-touch the artery. Detached slivers under
   10% of the sac are dropped; a larger detached piece stops the build.

**9.6 Acceptance checks.** The placed dome is closed by a flat cap at its rim. A sample is
**rejected** if:
- more than 80% of the dome lies inside the parent vessel;
- more than 10% of the dome lies inside *another* vessel, meaning:
  - artery voxels owned by centerline points far along the graph from the site;
  - or another connected piece of the mask;
- the sac shape is outside the real range (not dome-like).

The shape metrics are computed on the voxel sac:
- volume;
- elongation √(λ₁/λ₃);
- flatness √(λ₂/λ₃);
- thickness (inscribed-sphere diameter / principal length);
- solidity (volume / convex-hull volume).

Each limit is a margin past the real sacs' 1st/99th percentile.

A sample also fails to build when:
- there is no dome or no rim;
- the rim can't be projected onto the wall, or its move is too large;
- the sac is empty or detached.

Failed samples are retried with new draws, up to 4n attempts for n requested masks.

**9.7 Guarantees by construction.**
- No artery voxel changes: the output's label 1 is exactly the host mask.
- The aneurysm lies inside the core.
- The sac touches the artery along the neck.

---

## 10. Design rationale (for the discussion)

- **Skeleton, not surface, as the condition.** A healthy surface under a real sac is not
  observable. The centerline is approximately invariant to sac growth, and the radius is
  repaired. The condition is also cheap to extract at inference: about 1 s per vessel,
  0.7 s of it VMTK.
- **Generated, not picked, ostium.** Earlier toolkits (SynVA) chose the ostium by hand or by
  rule (bifurcation or curvature). Here the site and neck shape are learned from 522 real
  (vessel, aneurysm) pairs.
- **Canonical Fourier ostium code.** It is low-dimensional (33 numbers), unique, smooth by
  construction, and invariant to global pose and scale (units of the local vessel radius).
- **Frame anchored to the ostium for the complex.** The pose target is invariant to global
  rotation, and the generated complex automatically sits at the generated neck.
- **Shortcut controls:**
  - off-centre crops (the sac is central in region-grown meshes);
  - radius smoothing over the whole graph (the repaired radius would otherwise be
    unnaturally smooth);
  - no answer-revealing inputs;
  - patient-level split.
- **Self-supervised terms at sampled sites.** The supervised terms see only one site per
  case. At inference the model generates everywhere, so consistency (dome on ring, stubs on
  the vessel, requested morphology) is enforced at arbitrary sites without ground truth.
- **Teacher forcing with jitter.** Every stage is trained on the true upstream values with
  small perturbations (site jitter, σ flips), so it tolerates its own upstream samples at
  inference.

---

## 11. Ablation axes available

Each run is a named folder under `runtime_train/ostium_model/` with its `args.json`. These
are the switches:

| switch | values |
|---|---|
| generator backbone | flow matching vs MLP |
| graph crops | off-centre crops vs whole graph |
| rotation | none vs ≤ 15° vs uniform SO(3) |
| self-supervised terms | off vs on |
| site sampling for those terms | model / uniform / mix |
| selfrim / cap / tube weights | 0.1/0.1/0.3 vs 0.2/0.2/0.6 (doubled) |
| graph-token positions | normalised vs raw (`norm_tokens`) |
| morphology conditioning | off vs on |
| condition dropout | 0 vs 0.2 |
| morph loss | two-sided vs one-sided |

---

## 12. Numbers already measured (verify before citing)

- **Representation and data:** §4.4 and §3.3.
- **End to end, on 10 healthy test patches** from the image-generation repo (6 sites per
  patch, seed 0, 60 samples):

  | model | accepted | sac volume, median / 90th percentile |
  |---|---|---|
  | `abl_mlp_crop_norot` (epoch 1000) | 53/60 (88%) | 21.6 / 58.2 mm³ |
  | `abl_mlp_crop_rot15_selfrim_graph_mix` | 44/60 (73%) | 9.1 / 37.9 mm³ |

  The median real dome volume is about 53 mm³, so generated sacs are smaller than real ones.
  The core box also clips large sacs.
- **Morphology conditioning:** the training script's periodic `morph_sanity` evaluates a
  fixed grid of requested (volume, AR) on held-out cases. It reports:
  - the median volume error and the share within 25%;
  - the AR error and the share within 0.3;
  - the slope and R² of generated against requested values.

  The figures are in each run's `sanity_morph/` and `log.jsonl`. Re-measure on the final
  checkpoint.

---

## 13. Open points (state them as limitations or future work)

- **Domain gap:** the sacs come from voxelised generated surfaces, while the image model
  trains on manual labels (different smoothness and partial-volume behaviour).
- **Small sacs:** generated sacs are smaller than real ones (§12).
- **Residual radius cue:** the repaired radius stretch lacks fine variation even after
  smoothing. Radius jitter or decoy interpolations were considered but not implemented.
- **Training graphs come from region-grown meshes, inference graphs from patch masks.** The
  training-like crops (§9.2) narrow this gap, and the extraction must follow the same
  recipe.
- **Final configuration:** which configuration (flow or MLP, rotation, morphology) is
  reported is still to be decided.

---

## Appendix: where each part lives (for checking details)

| part | file |
|---|---|
| skeleton extraction | `geometry/extract_skeleton.py` |
| sac removal and vessel healing | `geometry/heal_vessel.py` |
| ostium ground truth, Fourier code, frame | `geometry/ostium.py` |
| complex target (z, affine in the ostium frame) | `geometry/complex_target.py` |
| per-case preprocessing | `dataset/preprocess_vessel_graph.py` |
| training dataset and augmentation | `dataset/ostium_dataset.py` |
| model | `models/ostium_model.py` |
| dome markers | `models/dome_markers.py` |
| morphology labels | `scripts/ostium/morph_reference.py` → `dataset/ostium_morph_reference.json` |
| training and losses | `scripts/train/train_ostium_model.py` |
| inference | `inference/ostium_inference.py` |
| attachment and voxel merge | `inference/merge.py`, `inference/sac_shape.py` |
| mask tool | `inference/synthetic_masks.py` |
| validation split | `dataset/ostium_val_cases.json` |
| original design notes | `TASK_vessel_conditioned_aneurysm.md` |
