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
    # endpoint_focus, but with ONE branch singled out: its endpoint term counts
    # 3x what every other branch's does. For a case whose alignment is dragged
    # off by a single bad branch -- or one whose alignment hinges on a single
    # branch being right -- weighting all endpoints equally lets the majority
    # win. Unlike the other configs this one is not a fixed string: it carries
    # a {branch} placeholder that must be filled per case, which is what
    # ALIGN_NEEDS_BRANCH below marks. Use align_flags() rather than indexing
    # ALIGN_CONFIGS directly so an unfilled placeholder can never reach a
    # command line.
    "endpoint_focus_branch": ("--w-endpoint 5.0 --w-centerline 0.5 --w-surface 0.5 "
                              "--w-dome 0.5 --branch-focus-id {branch} "
                              "--branch-focus-endpoint-weight 3.0"),
}

# Align configs needing a per-case branch index substituted into {branch}.
ALIGN_NEEDS_BRANCH = {"endpoint_focus_branch"}


def align_flags(name, focus_branch=None):
    """Resolved --w-* flag string for one align config.

    focus_branch is the rank-ordered branch index (0 = upstream/dome-first)
    that gets the extra endpoint weight. Required for the configs in
    ALIGN_NEEDS_BRANCH, ignored by the others.
    """
    if name not in ALIGN_CONFIGS:
        raise ValueError(f"unknown align config {name!r}; known: {sorted(ALIGN_CONFIGS)}")
    flags = ALIGN_CONFIGS[name]
    if name in ALIGN_NEEDS_BRANCH:
        if focus_branch in (None, ""):
            raise ValueError(f"align config {name!r} needs a focus branch index")
        return flags.format(branch=int(focus_branch))
    return flags
