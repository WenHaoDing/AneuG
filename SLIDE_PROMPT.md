# Prompt for generating two architecture slides

Copy everything below the line into your slide-generating tool.

---

Create **two presentation slides**, each a clean architecture diagram of one neural
network. Same visual language on both so they can be compared side by side.
Audience: machine-learning researchers. Landscape 16:9. Prefer a left-to-right
data flow. Label every tensor with its shape. Use one accent colour for anything
new or changed, grey for everything unchanged.

## Shared context (state briefly on slide 1 only)

The input is a **triangle mesh of a cerebral aneurysm**, reconstructed from 144x3
Graph Harmonic Deformation coefficients on a fixed canonical template. Vertex
count is fixed per aneurysm type: 4143 for bifurcation, 2590 for sidewall.
Vertices correspond across all meshes of a type. Node features are
**xyz + surface normal = 6 channels**; edges are mesh edges. During training each
mesh gets an independent random SO(3) rotation.

---

# SLIDE 1 — "MorphoFormer: deterministic morphology sensor"

## Flow

1. **Input graph** `[N, 6]` xyz + normal, N = 4143 or 2590
2. **Encoder (GCNGPSEncoder)**, 614,400 params
   - stem: `GCNConv(6 -> 128)` then `GCNConv(128 -> 128)`
   - then **4 x GPSConv(128, heads=4)** — local message passing + global attention
3. **Per-vertex features** `[B, N, 128]`
4. Branch into two routes:
   - **per-vertex route** — heads applied at every vertex
   - **pooled route** — masked mean over vertices -> **embedding** `[B, 128]`

## Heads (show the two routes clearly)

Per-vertex:
- `loc_head: Linear(128 -> 3)` -> **softmax over VERTICES** -> `loc_probs [B, 3, N]`
  (one distribution per branch slot; 3 slots, bifurcation uses 3, sidewall 2)
- **endpoint** = soft-argmax: `einsum(loc_probs, vertex_positions)` -> `[B, 3, 3]`
  (emphasise: the endpoint is a weighted average of REAL vertex positions,
  not a free regression)
- `dir_head: Linear(128 -> 3)` -> softmax over vertices -> pool features ->
  `tangent_head: MLP(128 -> 128 -> 3)` -> L2-normalised **tangent** `[B, 3, 3]`
- `dome_head: Linear(128 -> 1)` -> per-vertex sigmoid -> **dome mask** `[B, N]`

From the pooled embedding:
- `phi_head: MLP(128 -> 128 -> 432)` -> reconstructs the GHD coefficients
- `rotation_head: MLP(128 -> 128 -> 6)` -> the augmentation rotation, as the
  6D representation (first two columns of R)

## Callout box

> The pooled **embedding [B,128]** is the shape descriptor used for the
> FPD / KPD generation metrics. The phi and rotation heads exist to force it to
> encode whole-shape geometry, not just where the openings are.

## Footer

Total 721,728 parameters, of which 614,400 are the encoder.
Losses: endpoint MSE, tangent (1 - cosine), cap patch cross-entropy (weight 2.0),
dome BCE (1.0), phi MSE (0.5), rotation MSE (0.5).

---

# SLIDE 2 — "Probabilistic variant: cap regions as a distribution"

Reuse slide 1's diagram in grey. Draw **only the additions** in the accent colour.

## Motivation box (top)

> A human brushing a cap does not draw a repeatable boundary. A deterministic
> head trained on one label per mesh collapses that ambiguity to roughly the
> per-vertex mean, which is not a labelling anyone would draw. This variant
> samples plausible labellings instead.

## Additions

1. From **embedding [B,128]**: `prior_net: MLP(128 -> 128 -> 16)` -> `mu, logvar`
   of an **8-dimensional latent z**. *Used at inference.*
2. **Training only**, draw in a dashed box:
   - `label_encoder`: pool per-vertex features over each labelled region
     (3 cap slots + dome) -> concat `[B, 512]` -> `MLP(512 -> 128 -> 128)`
   - `posterior_net: MLP(256 -> 256 -> 16)` on `[embedding, label_summary]`
     -> `mu, logvar`. *Sees the actual label.*
3. **z [B, 8]** — sampled from the POSTERIOR during training, from the PRIOR at
   inference. Show this as a switch or two-way arrow.
4. **FiLM fusion** (this is the key panel):
   - `film: Linear(8 -> 256)`, zero-initialised -> `gamma, beta` each `[B, 128]`
   - `fz = feat_dense * (1 + gamma) + beta`  -> `[B, N, 128]`
   - only `loc_head` and `dome_head` read `fz`
5. `dir_head`, `tangent_head`, `phi_head`, `rotation_head` stay on the
   UNMODULATED features/embedding — draw them in grey.
6. Loss gains `+ beta * KL(posterior || prior)`

## The one box that must stand out

> **Why multiplicative, not concatenation.**
> Concatenating z onto each vertex feature adds the *same constant* to every
> vertex's logit. The cap head is a softmax **over vertices**, which is
> shift-invariant, so the constant cancels exactly and z has no effect at all —
> measured max |p(z1) - p(z2)| = 1.9e-09. FiLM scales the features instead, and
> because features differ per vertex, scaling moves vertices by different
> amounts. That survives the softmax.

## Sampling illustration (bottom right)

Small inset: the same mesh with 4 different z draws giving 4 cap regions of
182 / 198 / 199 / 151 vertices — sharing a core of 140, union 200, i.e. **70%
overlap**. Caption: "agreement in the middle, disagreement at the boundary."

## Footer

892,416 parameters, +170,688 over the deterministic model. Of those, 152,080
(`label_encoder` + `posterior_net`) exist only during training, so the deployed
model is just 18,608 parameters heavier. FiLM is zero-initialised, so at
initialisation the variant reproduces the deterministic model exactly for any z.
