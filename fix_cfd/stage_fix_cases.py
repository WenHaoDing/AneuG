"""Pull the ImperialNHS cases that need re-running out of the dataset, and stage them for CFD.

A case needs fixing when the registry flags it for either problem:

    roughness_flag   surface rougher than the physiological reference (check_roughness.py)
    flowsplit_wrong  run with the wrong outlet flow split              (fix_cfd_cases.py)

Both problems only exist in ImperialNHS cases. A flagged case from any other dataset means
the registry is not what this script expects, so it stops rather than moving anything.

For every case that needs fixing:

  1. MOVE its folder out of the dataset
         angioflowv2_merged/<case>  ->  angioflowv2_merged_fix_backup/<case>
     Both sit on the same disk, so this is a rename: instant, and nothing is copied.
  2. COPY its CFD source folder into a fresh staging area
         angioflow/cfd/ImperialNHS/<case>  ->  angioflow/cfd/ImperialNHS_fix/<case>

A manifest, fix_manifest.csv, is written into both new folders: one row per case with its
reasons, the values that flagged it, where it was moved from and to, and what happened. It
is the record of what left the dataset, and undoing a move is renaming the folder back.

NOTE ON RE-RUNNING THE CHECKS. Once cases are moved, fix_cfd_cases.py and
check_roughness.py no longer find them in the dataset folder, so a re-run of either would
blank their flags in the registry. The manifest keeps the list either way.

The script refuses to act on a half-finished registry: every ImperialNHS row must already
carry a roughness status and a flow split status.

    python fix_cfd/stage_fix_cases.py --dry_run     # show exactly what would happen
    python fix_cfd/stage_fix_cases.py               # do it

Re-running is safe: cases already moved or already copied are recognised and skipped.
"""

import argparse
import csv
import os
import shutil
import sys
import time

import pandas as pd

DATASET_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
REGISTRY = os.path.join(DATASET_ROOT, "_meta", "case_registry.xlsx")
BACKUP_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged_fix_backup"
CFD_SOURCE_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/ImperialNHS"
FIX_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/ImperialNHS_fix"

DATASET = "ImperialNHS"
MANIFEST = "fix_manifest.csv"


def _flag(series):
    """Registry booleans come back as True/False, 1/0 or NaN depending on the writer."""
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])


def select_cases(reg):
    # roughness_reference_model is written only by the current check_roughness.py; an
    # older registry still carries a full set of roughness flags from the previous
    # reference model, which would otherwise pass every check below.
    for col in ("roughness_flag", "roughness_status", "roughness_reference_model",
                "flowsplit_wrong", "flowsplit_status"):
        if col not in reg.columns:
            sys.exit("registry has no %s column: run check_roughness.py and fix_cfd_cases.py "
                     "first" % col)
    imp = reg[reg["dataset"] == DATASET]
    unfinished = imp[imp["roughness_status"].isna() | imp["flowsplit_status"].isna()]
    if len(unfinished):
        sys.exit("%d %s rows have no roughness or flow split status yet (e.g. %s): the checks "
                 "have not finished" % (len(unfinished), DATASET, unfinished["case"].iloc[0]))

    rough, split = _flag(reg["roughness_flag"]), _flag(reg["flowsplit_wrong"])
    chosen = reg[rough | split].copy()
    other = chosen[chosen["dataset"] != DATASET]
    if len(other):
        sys.exit("flagged cases outside %s, which should be impossible: %s"
                 % (DATASET, ", ".join(other["case"].head(10))))
    chosen["fix_reasons"] = [
        "+".join(r for r, on in (("roughness", a), ("flowsplit", b)) if on)
        for a, b in zip(rough[chosen.index], split[chosen.index])]
    return chosen


def same_filesystem(a, b):
    return os.stat(a).st_dev == os.stat(b).st_dev


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=REGISTRY)
    p.add_argument("--dry_run", action="store_true", help="print the plan; move and copy nothing")
    args = p.parse_args()

    reg = pd.read_excel(args.registry)
    chosen = select_cases(reg)
    counts = chosen["fix_reasons"].value_counts().to_dict()
    print("%d case(s) need fixing: %s" % (len(chosen), counts))

    if not args.dry_run:
        os.makedirs(BACKUP_ROOT, exist_ok=True)
        os.makedirs(FIX_ROOT, exist_ok=True)
        if not same_filesystem(DATASET_ROOT, BACKUP_ROOT):
            print("note: dataset and backup are on different filesystems; moves will copy")

    rows, problems = [], []
    for _, r in chosen.iterrows():
        case = r["case"]
        src = r["case_dir"] if isinstance(r["case_dir"], str) and r["case_dir"] else \
            os.path.join(DATASET_ROOT, case)
        backup = os.path.join(BACKUP_ROOT, case)
        cfd_src = os.path.join(CFD_SOURCE_ROOT, case)
        if not os.path.isdir(cfd_src) and isinstance(r.get("original_name"), str):
            cfd_src = os.path.join(CFD_SOURCE_ROOT, r["original_name"])
        fix_dst = os.path.join(FIX_ROOT, case)

        # 1. move out of the dataset
        if os.path.isdir(src) and os.path.exists(backup):
            move = "conflict: exists in both dataset and backup, left untouched"
        elif os.path.isdir(src):
            move = "moved"
            if not args.dry_run:
                shutil.move(src, backup)            # a rename on the same filesystem
        elif os.path.isdir(backup):
            move = "already moved"
        else:
            move = "missing: in neither dataset nor backup"

        # 2. copy the CFD source into the staging area
        if not os.path.isdir(cfd_src):
            copy = "missing: no CFD source folder"
        elif os.path.exists(fix_dst):
            copy = "already staged"
        else:
            copy = "copied"
            if not args.dry_run:
                shutil.copytree(cfd_src, fix_dst)

        if move.startswith(("conflict", "missing")) or copy.startswith("missing"):
            problems.append((case, move, copy))
        rows.append({
            "case": case, "fix_reasons": r["fix_reasons"],
            "roughness_worst_target_sd": r.get("roughness_worst_target_sd"),
            "roughness_reference_model": r.get("roughness_reference_model"),
            "flowsplit_applied_5": r.get("flowsplit_applied_5"),
            "flowsplit_correct_5": r.get("flowsplit_correct_5"),
            "moved_from": src, "moved_to": backup, "move": move,
            "cfd_source": cfd_src, "staged_to": fix_dst, "copy": copy,
        })

    summary = pd.DataFrame(rows)
    print("move: %s" % summary["move"].value_counts().to_dict())
    print("copy: %s" % summary["copy"].value_counts().to_dict())
    for case, move, copy in problems:
        print("  ! %s | %s | %s" % (case, move, copy))

    if args.dry_run:
        print("--dry_run: nothing moved or copied")
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    summary.insert(0, "staged_at", stamp)
    for root in (BACKUP_ROOT, FIX_ROOT):
        path = os.path.join(root, MANIFEST)
        # append on re-runs so the history of what moved when is kept
        summary.to_csv(path, mode="a", index=False, header=not os.path.exists(path),
                       quoting=csv.QUOTE_MINIMAL)
    print("manifest written to %s and %s" % (os.path.join(BACKUP_ROOT, MANIFEST),
                                            os.path.join(FIX_ROOT, MANIFEST)))


if __name__ == "__main__":
    main()
