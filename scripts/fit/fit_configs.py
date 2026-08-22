"""Named GHD-fitting config presets, keyed by name.

Each value is a string of extra CLI flags appended to the base
`run_case.py --case-dir ... --save-root ...` invocation. "default" is
intentionally empty -- run_case.py's own argparse defaults already match
the config the full ImperialNHS production batch was run with (dome-region
alignment on, occupancy loss on at 2.0, no opening-index-dependent losses).

Add new presets here as they're needed for redo cases; manage.py's
`requeue` subcommand looks up each manifest row's `refit_config` value in
this dict.
"""

CONFIGS = {
    "default": "",
    # Opening-index-dependent losses at 25% of their original 0.5 weight --
    # the full-weight version ("not working well" per user feedback) was
    # overpowering the fit; toned down rather than dropped entirely.
    "with_opening": (
        "--lambda-opening-chamfer 0.125 --lambda-roundness 0.125 "
        "--lambda-normal-alignment 0.125 --lambda-geodesic 0.125"
    ),
    # Geodesic (GeoDistance) term only, isolated from the other three
    # opening-dependent losses in "with_opening" -- weighted equal to the
    # main surface-point chamfer term (loss_chamfer, fixed weight 1.0 in
    # ghd_fit.py's total loss -- not itself a CLI flag), so it's a real
    # peer to the primary position-matching signal rather than a minor add-on.
    "with_geo_dist": "--lambda-geodesic 1.0",
    "no_occupancy": "--lambda-occupancy 0.0",
    "high_occupancy": "--lambda-occupancy 4.0",
}
