"""Find CFD cases that ran with the wrong outlet flow split, and record them in the registry.

THE PROBLEM. Bifurcated (two-outlet) cases impose a flow split between the two outlets,
read by the Fluent journal from flowsplit_ratio.txt: Q5/Q6 = (d5/d6)^2.45, with d the
equivalent diameter of each outlet cap. Some cases were run with split values that do
not match their flowsplit_ratio.txt. Example, 1KPkS2Uocy_aneurysm1: the run applied
0.7159 / 0.2841, the file says 0.8315 / 0.1685.

HOW IT IS DETECTED. Fluent's transcript (the `output` file in each case folder) records
the exact values the journal entered for each outflow zone:

    > /define/boundary-conditions/outflow (surface6 surface5)
    5 0.7158880386736757
    > /define/boundary-conditions/outflow (surface6 surface5)
    6 0.2841119613263243

Those applied values are compared with the case's own flowsplit_ratio.txt, and ANY
mismatch is wrong -- there is no tolerance. Every case whose log sets a split used the
same constant 0.7159 / 0.2841, so a case whose correct split merely lies close to it
still ran with the wrong file. The comparison allows only for float formatting
(MATCH_EPS), which matters because both files print the values to 16 digits. The log is
used rather than the solved velocity field because it is exact: integrating the outlet
velocities from blood_data.pt was tried on the example case and left 18% of the inflow
unaccounted for, far too coarse to tell a correct split from a wrong one.

SCOPE. Only cases with an `output` log are checked. Cases without one were run by
someone else, whose runs do not have this problem. Sidewall cases set no outflow split
(a single pressure outlet), so they are recorded as not applicable.

RESULT. These columns are added to _meta/case_registry.xlsx, replaced if already there:

    cfd_log                True if the case folder has an `output` log
    flowsplit_applied_5/6  weights the run actually used, normalized to sum to 1
    flowsplit_correct_5/6  values in flowsplit_ratio.txt
    flowsplit_abs_error    |applied - correct| for zone 5 (zone 6 is the complement)
    flowsplit_status       ok | wrong | no_log | no_outflow_split | unparsed | no_ratio_file
    flowsplit_wrong        True only for status 'wrong'

The registry is backed up next to itself before it is written.

    python fix_cfd/fix_cfd_cases.py                  # check and write the registry
    python fix_cfd/fix_cfd_cases.py --dry_run        # check only
"""

import argparse
import os
import re
import shutil
import time

import pandas as pd

DATASET_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
case_registry = os.path.join(DATASET_ROOT, "_meta", "case_registry.xlsx")

LOG_NAME = "output"
RATIO_NAME = "flowsplit_ratio.txt"
OUTFLOW_ZONES = (5, 6)
# Applied and correct values are both written to 16 significant digits, so a correctly
# applied split matches to ~1e-15. This only absorbs formatting, not a real difference.
MATCH_EPS = 1e-6
# The outflow commands sit in the setup section at the top of the transcript, before the
# solver's per-iteration output; nothing past this point is needed.
MAX_LOG_LINES = 5000

_VALUE_LINE = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def applied_flow_split(log_path):
    """Weights the journal entered for zones 5 and 6, or None if the log sets none.

    Each value is on the first non-empty line after an outflow command echo.
    """
    found = {}
    with open(log_path, "r", errors="replace") as f:
        pending = False
        for i, line in enumerate(f):
            if i >= MAX_LOG_LINES or len(found) == len(OUTFLOW_ZONES):
                break
            if "boundary-conditions/outflow" in line:
                pending = True
                continue
            if pending and line.strip():
                m = _VALUE_LINE.match(line)
                if m and int(m.group(1)) in OUTFLOW_ZONES:
                    found[int(m.group(1))] = float(m.group(2))
                pending = False
    if not found:
        return None
    if set(found) != set(OUTFLOW_ZONES):
        raise ValueError("log sets outflow weights for zones %s only" % sorted(found))
    total = sum(found.values())
    return {z: found[z] / total for z in OUTFLOW_ZONES}     # Fluent weights are relative


def correct_flow_split(ratio_path):
    vals = [float(x) for x in open(ratio_path).read().split()]
    if len(vals) != 2:
        raise ValueError("%s holds %d values, expected 2" % (ratio_path, len(vals)))
    return {5: vals[0], 6: vals[1]}


def check_case(case_dir):
    row = {"cfd_log": False, "flowsplit_applied_5": None, "flowsplit_applied_6": None,
           "flowsplit_correct_5": None, "flowsplit_correct_6": None,
           "flowsplit_abs_error": None, "flowsplit_status": "no_log", "flowsplit_wrong": False}
    log_path = os.path.join(case_dir, LOG_NAME)
    if not os.path.isfile(log_path):
        return row
    row["cfd_log"] = True
    try:
        applied = applied_flow_split(log_path)
    except ValueError:
        row["flowsplit_status"] = "unparsed"
        return row
    if applied is None:
        row["flowsplit_status"] = "no_outflow_split"
        return row
    ratio_path = os.path.join(case_dir, RATIO_NAME)
    if not os.path.isfile(ratio_path):
        row.update(flowsplit_status="no_ratio_file",
                   flowsplit_applied_5=applied[5], flowsplit_applied_6=applied[6])
        return row
    correct = correct_flow_split(ratio_path)
    err = abs(applied[5] - correct[5])
    wrong = err > MATCH_EPS
    row.update(flowsplit_applied_5=applied[5], flowsplit_applied_6=applied[6],
               flowsplit_correct_5=correct[5], flowsplit_correct_6=correct[6],
               flowsplit_abs_error=err, flowsplit_status="wrong" if wrong else "ok",
               flowsplit_wrong=wrong)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DATASET_ROOT)
    p.add_argument("--registry", default=case_registry)
    p.add_argument("--dry_run", action="store_true", help="report only; do not write the registry")
    args = p.parse_args()

    cases = sorted(d for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d)) and not d.startswith("_"))
    results = {c: check_case(os.path.join(args.root, c)) for c in cases}

    status = pd.Series({c: r["flowsplit_status"] for c, r in results.items()})
    print("case folders: %d" % len(cases))
    for s, n in status.value_counts().items():
        print("  %-17s %d" % (s, n))
    wrong = [c for c, r in results.items() if r["flowsplit_wrong"]]
    print("\nwrong flow split (applied differs from flowsplit_ratio.txt): %d" % len(wrong))
    for c in wrong:
        r = results[c]
        print("  %-32s applied %.4f / %.4f   correct %.4f / %.4f   off by %.1f pp"
              % (c, r["flowsplit_applied_5"], r["flowsplit_applied_6"], r["flowsplit_correct_5"],
                 r["flowsplit_correct_6"], 100 * r["flowsplit_abs_error"]))
    odd = [c for c, r in results.items() if r["flowsplit_status"] in ("unparsed", "no_ratio_file")]
    if odd:
        print("\nneeds a look (log could not be parsed, or no ratio file): %s" % ", ".join(odd))

    reg = pd.read_excel(args.registry)
    missing = sorted(set(cases) - set(reg["case"]))
    if missing:
        print("\ncase folders not in the registry, so not recorded there: %s" % ", ".join(missing))
    if args.dry_run:
        print("\n--dry_run: registry not written")
        return

    cols = list(next(iter(results.values())).keys())
    new = pd.DataFrame.from_dict(results, orient="index")[cols]
    reg = reg.drop(columns=[c for c in cols if c in reg.columns])
    reg = reg.merge(new, left_on="case", right_index=True, how="left")
    backup = "%s.bak_%s" % (args.registry, time.strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(args.registry, backup)
    reg.to_excel(args.registry, index=False)
    print("\nregistry updated: %s  (backup: %s)" % (args.registry, os.path.basename(backup)))


if __name__ == "__main__":
    main()
