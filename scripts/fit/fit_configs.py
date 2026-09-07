"""Named GHD-fitting config presets, keyed by name.

SIX configs, deliberately. Everything else that used to live here
(with_geo_dist, no_occupancy, high_occupancy, and the 0.125-weight opening
variant) was found not to earn its keep and has been removed.

Unlike the earlier version, a config is NOT just a flag string appended to
run_case.py -- two of the four run a DIFFERENT SCRIPT with a different
output-path flag, and one needs an extra input. So each entry is a dict:

  script        : path, relative to the repo root, of the script to run
  out_flag      : "--save-root" (run_case.py: dataset dir, script appends the
                  case name) or "--out-dir" (the other two: full case dir)
  flags         : extra CLI flags
  supports_align: whether alignment.py's --w-* ALIGN_CONFIGS flags apply.
                  Only run_case.py exposes them; the ARAP and clone pipelines
                  run their own Stage-1 internally and accept no --w-* flags,
                  so pairing them with a non-default align config is refused.
  needs_old_case: clone only -- also needs --old-case-dir pointing at the old
                  project's already-fitted result for the same case.
  supports_eta_min : clone_fit.py has no --eta-min flag; the other two do.

manage.py's `requeue` looks up each manifest row's `refit_config` here.
"""

# Old GHD project's fitted results, the warm-start source for "clone".
# <OLD_FIT_ROOT>/<old dataset dir>/<case_name>/ghd_fitted.obj
from pathlib import Path

OLD_FIT_ROOT = "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final"

# The old project spelled AneuX as "AnueX". Geometry uses the correct spelling,
# so clone/clone_target need this mapping to find a case's old fitted mesh.
OLD_DATASET_DIR = {"ImperialNHS": "ImperialNHS", "AneuX": "AnueX"}

# Where each dataset's dome-region data lives. ImperialNHS has label.nrrd
# (voxel label==2); AneuX has NO label.nrrd at all -- 0/323 -- and instead
# ships a dome_sac.ply per case (323/323). Getting this wrong is SILENT:
# load_dome_points_from_nrrd returns None when the file is missing and callers
# "skip the dome loss term gracefully", so the whole dome term just vanishes
# from Stage-1 alignment with only a printed note.
DATASET_DOME_SOURCE = {"ImperialNHS": "nrrd", "AneuX": "mesh", "AneuX_stable": "mesh"}


def dome_source_for(dataset):
    return DATASET_DOME_SOURCE.get(dataset, "nrrd")


# What clone_fit.py actually reads out of an old case dir. ghd_fitted.obj alone
# is NOT enough: it also needs landmarks.npz (for aneu_type) and metrics.json
# (for the old s_can). Checking only the mesh routed AneuX/C0035_cut2 -- which
# has the mesh but no landmarks.npz -- to `clone`, where it died on a
# FileNotFoundError instead of falling back to `default`.
# ghd_fitted_uncapped.obj is NOT required: clone_reframe falls back to
# multi-start when the uncapped rings are unavailable.
_CLONE_REQUIRED = ("ghd_fitted.obj", "landmarks.npz", "metrics.json")


def old_case_dir(dataset, case_name):
    """Path to a case's old fitted result, or None if it isn't usable."""
    d = Path(OLD_FIT_ROOT) / OLD_DATASET_DIR.get(dataset, dataset) / case_name
    return d if all((d / f).exists() for f in _CLONE_REQUIRED) else None


def has_clone_source(dataset, case_name):
    return old_case_dir(dataset, case_name) is not None

# ── Stage-A MSE : mesh-health ratio ───────────────────────────────────────
# For arap_init and clone, Stage A is driven by a per-vertex MSE against the
# warm-start mesh, balanced against the mesh-health group (rigid, laplacian,
# normal-consistency, edge, thickness). The MSE weight is held at 1.0 and the
# WHOLE mesh-health group is scaled so that rigid = 1/ratio -- so "ratio" is
# the STARTING MSE:rigid balance, and it is the one knob that moves it.
#
# DIRECTION: rigid = 1/ratio, so a HIGHER ratio means WEAKER mesh health and
# MORE warping in Stage A; a LOWER ratio clamps the mesh harder. If a case
# warps too much at r5, go DOWN (r2, r1, r0.5), not up.
#
# The two pipelines have different base weights, so the same ratio implies a
# different scale factor in each; that is what stage_a_ratio_flags() resolves.
# Note clone_fit also decays rigid to 25% of its start over Stage A, so the
# ratio names the start, not the end.
STAGE_A_RATIOS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
# Fallback when a config declares no default of its own. Each config carries
# its own "default_stage_a_ratio" so the two pipelines can diverge once the
# sweeps say they should -- ARAP's historical setting is r20, clone_fit's was
# r0.5, and both are currently set to r5.
DEFAULT_STAGE_A_RATIO = 5.0

_ARAP_BASE = {"rigid": 0.05, "laplacian": 0.001, "consistency": 0.03, "edge": 0.01}
_CLONE_BASE_RIGID = 2.0


def stage_a_ratio_flags(config_name, ratio=None):
    """CLI flags setting the Stage-A MSE:mesh-health ratio for one config.

    Returns "" for configs with no Stage A (default, with_opening)."""
    cfg = CONFIGS.get(config_name)
    if not cfg or not cfg.get("supports_stage_a_ratio"):
        return ""
    if ratio in (None, ""):
        ratio = cfg.get("default_stage_a_ratio") or DEFAULT_STAGE_A_RATIO
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError(f"stage-A ratio must be positive, got {ratio}")
    if config_name == "arap_init":
        # scale the whole group so rigid lands on 1/ratio
        s = 1.0 / (ratio * _ARAP_BASE["rigid"])
        return ("--lambda-rigid-stage-a %g --lambda-laplacian-stage-a %g "
                "--lambda-consistency-stage-a %g --lambda-edge-stage-a %g"
                % (_ARAP_BASE["rigid"] * s, _ARAP_BASE["laplacian"] * s,
                   _ARAP_BASE["consistency"] * s, _ARAP_BASE["edge"] * s))
    if config_name == "clone":
        # clone_fit takes the scale directly and applies it to the whole group
        return "--mesh-health-scale-stage-a %g" % (1.0 / (ratio * _CLONE_BASE_RIGID))
    return ""

CONFIGS = {
    # 1. Plain fit: dome-region alignment on, occupancy at 1.0, no
    #    opening-index losses -- otherwise run_case.py's argparse defaults.
    #    The ONE departure from those defaults is the rigid warm-up: 1000
    #    iterations holding the mesh stiff (rigid 5 -> 2, checkpointed at
    #    5/4/3) BEFORE the n_iter fit begins, so a run is 1000 + n_iter
    #    iterations total. Validated over the 160-case hand-fixed batch
    #    (2026-09-01): 152/152 completed cases wrote all three checkpoints,
    #    and every failure was a Stage-1 input-data problem, none from this.
    #    requeue's --rigid-warmup-iters is appended after these flags, so it
    #    still overrides (argparse takes the last occurrence).
    "default": {
        "script": "ghd/fitting/run_case.py",
        "out_flag": "--save-root",
        "supports_dome_source": True,
        "flags": "--rigid-warmup-iters 1000",
        "supports_align": True,
        "needs_old_case": False,
        "supports_eta_min": True,
        "supports_stage_a_ratio": False,
        "default_stage_a_ratio": None,
    },

    # 2. Opening-index-dependent losses on, all four at 0.2, plus the same
    #    1000-iteration rigid warm-up as "default" (5 -> 2, checkpointed at
    #    5/4/3, ON TOP OF n_iter). Both run_case.py configs therefore now
    #    start stiff; only the ARAP and clone pipelines, which run their own
    #    Stage A, are left alone.
    "with_opening": {
        "script": "ghd/fitting/run_case.py",
        "out_flag": "--save-root",
        "supports_dome_source": True,
        "flags": ("--rigid-warmup-iters 1000 "
                  "--lambda-opening-chamfer 0.2 --lambda-roundness 0.2 "
                  "--lambda-normal-alignment 0.2 --lambda-geodesic 0.2"),
        "supports_align": True,
        "needs_old_case": False,
        "supports_eta_min": True,
        "supports_stage_a_ratio": False,
        "default_stage_a_ratio": None,
    },

    # 3. ARAP warm start. Stage B keeps ghd_fit.py's default loss battery
    #    (i.e. the "default" config) at lr 3e-3; Stage A runs the r5
    #    mesh-health ratio at lr 3e-3. Ring-normal blend 0.5 is the validated
    #    orientation correction. Stage B also gets the same 1000-iteration
    #    rigid warm-up as default/with_opening (5 -> 2, checkpointed at 5/4/3,
    #    additive to n_iter). Stage A is left alone -- it has its own
    #    mesh-health ratio, which is the knob for warping there.
    "arap_init": {
        "script": "ghd/registration/fit_with_arap_init.py",
        "out_flag": "--out-dir",
        "supports_dome_source": True,
        "flags": ("--use-ring-normal --ring-normal-blend 0.5 --lr-stage-a 3e-3 --lr 3e-3 "
                  "--rigid-warmup-iters 1000"),
        "supports_align": False,
        "needs_old_case": False,
        "supports_eta_min": True,
        "supports_stage_a_ratio": True,
        "default_stage_a_ratio": 20.0,
    },

    # 4. Clone from the old project's fitted mesh. Stage B keeps the default
    #    loss battery; Stage A is the r5 ratio (mesh-health scale 0.1 against
    #    clone_fit's own base) with node-MSE 1.0 and opening chamfer 1.0, and
    #    with surface chamfer and occupancy OFF -- the exact Stage-A setup the
    #    mse_weight_sweep tested. The clone target is reframed onto the real
    #    target by surface chamfer first (ghd/registration/clone_reframe.py).
    "clone": {
        "script": "ghd/fitting/clone_fit.py",
        "out_flag": "--out-dir",
        "supports_dome_source": True,
        "flags": ("--lambda-node-mse 1.0 --lambda-opening-chamfer-stage-a 1.0 "
                  "--lambda-chamfer-stage-a 0.0 --lambda-occupancy-stage-a 0.0"),
        "supports_align": False,
        "needs_old_case": True,
        "supports_eta_min": False,
        "supports_stage_a_ratio": True,
        "default_stage_a_ratio": 5.0,
    },

    # 5. Same as "clone" but with the Stage-A opening-chamfer term OFF, so
    #    node-MSE (plus volume) is the only thing pulling the canonical onto
    #    the clone. Worth having as a control: with surface chamfer and
    #    occupancy already off, opening chamfer at 1.0 is ~87% of Stage A's
    #    initial loss, and in early probes it moved while node-MSE did not --
    #    the two can pull against each other. This isolates that.
    "clone_no_opening": {
        "script": "ghd/fitting/clone_fit.py",
        "out_flag": "--out-dir",
        "supports_dome_source": True,
        "flags": ("--lambda-node-mse 1.0 --lambda-opening-chamfer-stage-a 0.0 "
                  "--lambda-chamfer-stage-a 0.0 --lambda-occupancy-stage-a 0.0"),
        "supports_align": False,
        "needs_old_case": True,
        "supports_eta_min": False,
        "supports_stage_a_ratio": True,
        "default_stage_a_ratio": 5.0,
    },

    # 6. Fit the CLONE MESH ITSELF rather than the real target. The old
    #    project's fitted mesh is aligned into this repo's frame (the same
    #    surface-chamfer reframing "clone" uses), Stage A is SKIPPED entirely,
    #    and the default Stage-B loss battery then fits that clean, watertight
    #    surface instead of the noisy real clip. No Stage A means no
    #    MSE:mesh-health ratio to choose.
    "clone_target": {
        "script": "ghd/fitting/clone_fit.py",
        "out_flag": "--out-dir",
        "supports_dome_source": True,
        "flags": "--skip-stage-a --fit-target clone",
        "supports_align": False,
        "needs_old_case": True,
        "supports_eta_min": False,
        "supports_stage_a_ratio": False,
        "default_stage_a_ratio": None,
    },
}


def resolve_flags(config_name, stage_a_ratio=None):
    """Every flag a config contributes, with the Stage-A ratio resolved.

    Single source of truth for "what does this config actually run", shared by
    manage.py's requeue and its `configs` listing.
    """
    cfg = CONFIGS[config_name]
    return " ".join(x for x in (cfg["flags"],
                                stage_a_ratio_flags(config_name, stage_a_ratio)) if x)


def output_dir(config_name, dataset, case_name, runtime_root, align_config="default"):
    """Where a (config, case) result lives: runtime/<config>/<dataset>/<case>.

    Named after the CONFIG, not the dataset, so every config's results for the
    whole corpus sit together and a case never overwrites itself across
    configs. A non-default align config is appended (e.g. clone_endpoint_focus).
    """
    folder = config_name
    if align_config and align_config != "default":
        folder = f"{config_name}_{align_config}"
    return Path(runtime_root) / folder / dataset / case_name


def build_fit_command(config_name, dataset, case_name, geometry_dir, runtime_root,
                      python, device, n_iter, align_flags="", stage_a_ratio=None,
                      align_config="default"):
    """The full shell command for one (config, case). Single source of truth --
    used by manage.py's requeue and by run_all.py, so they cannot drift.

    Returns (command_string, out_dir). Raises if the config needs an old fitted
    mesh and the case has none.
    """
    cfg = CONFIGS[config_name]
    out = output_dir(config_name, dataset, case_name, runtime_root, align_config)
    # run_case.py takes the PARENT dir and appends the case name itself;
    # the other two take the full case dir.
    out_arg = (f'--save-root "{out.parent}"' if cfg["out_flag"] == "--save-root"
               else f'--out-dir "{out}"')
    parts = [python, cfg["script"], f'--case-dir "{geometry_dir}"', out_arg]
    if cfg["needs_old_case"]:
        old = old_case_dir(dataset, case_name)
        if old is None:
            raise ValueError(f"{config_name} needs an old fitted mesh, but none for "
                             f"{dataset}/{case_name}")
        parts.append(f'--old-case-dir "{old}"')
    src = dome_source_for(dataset)
    if src != "nrrd":
        if cfg.get("supports_dome_source", True):
            parts.append(f"--dome-source {src}")
        else:
            print(f"  WARNING: {config_name} has no --dome-source; {dataset} needs "
                  f"'{src}', so its Stage-1 dome term will be SKIPPED for "
                  f"{case_name}.")
    parts += [f"--device {device}", f"--n-iter {n_iter}"]
    if cfg["supports_eta_min"]:
        parts.append("--eta-min 1e-4")
    if cfg["supports_align"] and align_flags:
        parts.append(align_flags)
    parts.append(resolve_flags(config_name, stage_a_ratio))
    return " ".join(x for x in parts if x).strip(), out
