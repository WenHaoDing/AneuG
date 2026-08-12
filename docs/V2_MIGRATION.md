# V2 repository layout

The v2 implementation is now organized by responsibility instead of being
kept in a version-named directory. Git history and release tags should carry
the version boundary; package paths describe what code does.

## Supported v2 training pipeline

These are the three core entry points identified for the v2 upgrade:

| Stage | Entry point | Main implementation |
| --- | --- | --- |
| GHD shape VAE | `scripts/train/train_ghd_vae.py` | `models/ghd_vae.py` |
| Branch transformer | `scripts/train/train_branch_transformer.py` | `models/branch_transformer.py` |
| Fourier branch MLP-VAE | `scripts/train/train_branch_mlp_vae.py` | `models/branch_mlp_vae.py` |

Run entry points from the repository root, for example:

```bash
python scripts/train/train_ghd_vae.py
python scripts/train/train_branch_transformer.py
python scripts/train/train_branch_mlp_vae.py
```

The scripts resolve the repository root from their own locations, so they may
also be launched from another working directory.

## Reusable code in existing packages

- `models/branch_transformer.py`: branch transformer and its conditioners.
- `models/branch_mlp_vae.py`: Fourier branch MLP-VAE and its conditioners.
- `models/ghd_vae.py`: conditional GHD VAE used by the supported stage-one script.
- `models/multi_canonical_ghd_reconstruct.py`: multi-canonical GHD reconstruction.
- `models/ghd_vqvae.py`: preserved experimental GHD VQ-VAE model.
- `utils/generate_synthetic.py`: reusable checkpoint loading and synthetic generation.
- `utils/mesh_fusion.py`: vessel mesh fusion utilities.
- `visualization/sanity_check.py`: training sanity-check rendering.
- `dataset/ghd_dataset_alias.py`: preserved compatibility aliases from the former v2 package.

`Aneu_GHD/` was deliberately left unchanged.

## Preserved supporting and unrelated scripts

No script from the former `v2/` directory was deleted. They were classified as
follows so they can be reviewed independently:

### Evaluation and measurement

- `scripts/evaluate/eval_branch_transformer.py`: evaluates transformer branch generation.
- `scripts/evaluate/eval_branch_mlp_vae.py`: renders fully generated Fourier MLP-VAE samples.
- `scripts/evaluate/eval_ablation.py`: sweeps stage-one and stage-two checkpoint combinations.
- `scripts/evaluate/measure_spread.py`: compares geometric spread in real and synthetic datasets.

### Generation and export

- `scripts/generate/generate_synthetic.py`: exports complete synthetic cases and metadata.

### Visualization

- `scripts/visualize/batch_visualize_baselines.py`: renders baseline VAE/VQ-VAE checkpoints.
- `scripts/visualize/batch_visualize_tune1.py`: renders the first GHD VAE tuning run.
- `scripts/visualize/batch_visualize_tune2.py`: renders the second GHD VAE tuning run.
- `notebooks/visualize_ghd_generation_v2.ipynb`: preserved interactive generation notebook.

### Experimental v2 training

- `scripts/experimental/train_branch_transformer_experimental.py`: earlier transformer training variant.
- `scripts/experimental/train_ghd_vae_tune1.py`: first hard-coded GHD VAE tuning variant.
- `scripts/experimental/train_ghd_vae_tune2.py`: second hard-coded GHD VAE tuning variant.
- `scripts/experimental/train_ghd_vqvae.py`: experimental vector-quantized GHD model.

These files remain available, but they are not presented as supported v2 entry
points until their value is reviewed.

## Archived v1 entry points

The former root-level v1 Python entry points are preserved under
`scripts/legacy/v1/`, and their notebooks are under `notebooks/v1/`. They were
archived without modernizing their code. The obsolete `v2/` source directory
was removed; new code imports directly from `models`, `dataset`, `utils`, or
`visualization` as appropriate.
