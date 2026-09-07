#!/bin/bash
# Fit the AneuX cases whose clip was cut BY HAND ("_manual"), with the two
# fixes those cases need:
#
#   1. CENTERLINE REPAIR (this script, step 1). A hand cut puts the mesh's
#      openings somewhere else than the automatic cut's -- 3.5-5.6 mm away on
#      p171 -- so the stock clipped centerline no longer matches the mesh it is
#      paired with. repair_manual_centerlines.py re-clips it onto the manual
#      mesh and saves clipped_centerline{_fallback}_manual.npy alongside;
#      nothing is overwritten, and resolve_case_paths() then prefers it.
#
#   2. NO-EXTRUSION CLOSURE (automatic, inside load_case_data). A hand cut is
#      oblique to the centerline, so extruded rings are built on planes tilted
#      away from the opening they stitch to. Those cases are closed where they
#      are instead, and their unreferenced vertices dropped.
#
# Both are keyed off resolve_case_paths()'s used_manual_mesh, so no other case
# in any dataset is affected.
#
#   bash scripts/fit/run_manual_cases.sh              # repair, then fit
#   bash scripts/fit/run_manual_cases.sh --dry-run    # show the plan only
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="/home/yaplab2/miniconda3/envs/new/bin/python"
CASES_FILE="scripts/fit/aneux_manual_cases.txt"
mapfile -t CASES < "$CASES_FILE"

echo "=== step 1/2: repair centerlines for ${#CASES[@]} manual-clip case(s) ==="
"$PY" scripts/fit/repair_manual_centerlines.py --dataset AneuX "$@"

case " $* " in *" --dry-run "*) echo "(dry run -- not fitting)"; exit 0;; esac

echo
echo "=== step 2/2: fit them ==="
exec "$PY" scripts/fit/run_all.py \
    --datasets AneuX \
    --only-cases "${CASES[@]}" \
    --gpus cuda:0:2 cuda:1:2 cuda:2:3 \
    --priority-config clone \
    --fallback-config default
