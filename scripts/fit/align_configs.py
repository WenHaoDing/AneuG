"""Named Stage-1 (alignment) config presets, keyed by name.

Companion to fit_configs.py's CONFIGS, but for alignment.py's own --w-*
weight flags rather than ghd_fit.py's --lambda-* ones. manage.py's requeue
combines an align config with a fit config for each redo -- they're
independent axes, so any pairing is valid (e.g. endpoint_focus alignment +
with_opening fitting).

Weights are relative to each other within one alignment_loss() call, not
absolute -- "default" is empty since run_case.py's own argparse defaults
(all weights 1.0) already match the config every existing production batch
used.
"""

ALIGN_CONFIGS = {
    "default": "",
    # Branch endpoints matter more than exact surface/centerline-curve shape
    # or dome position -- useful when a case's overall silhouette is noisy
    # but its true vessel endpoints (branch-ranking-confirmed) are reliable.
    "endpoint_focus": "--w-endpoint 5.0 --w-centerline 0.5 --w-surface 0.5 --w-dome 0.5",
    # Full centerline/skeleton shape matters more than surface chamfer or
    # dome position -- useful when the dome/opening data is noisy or
    # unreliable but the branch topology and curvature are trustworthy.
    "skeleton_focus": "--w-centerline 5.0 --w-surface 0.3 --w-dome 0.2",
}
