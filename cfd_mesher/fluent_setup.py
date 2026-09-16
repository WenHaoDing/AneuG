"""Put the ANSYS Fluent run files into a volume-meshed case folder.

Every case needs the same UDF and inlet waveform, plus a journal that depends on how
many outlets the case has. They are copied from cfd_mesher/ansys_files/:

    parabolic_vinlet_udf_full.c   parabolic inlet velocity UDF      same for every case
    udf_waveform_5cycles.txt      inlet flow waveform the UDF reads  same for every case
    jnl_bifurcated                journal for 1 inlet + 2 outlets  -> copied as jnl_transient
    jnl_sidewall                  journal for 1 inlet + 1 outlet   -> copied as jnl_transient

WHICH JOURNAL. The bifurcated journal sets zones 5 AND 6 as outflows split by
flowsplit_ratio.txt; the sidewall journal sets only zone 5, as a pressure outlet. So the
choice is really "does this mesh have a second outlet", and it is decided from the
number of branches in the case's fusion npz -- the same record the volume mesher
labels boundary zones from. The case's recorded aneurysm type (meta.json) is checked
against it when present, and a disagreement is an error rather than a guess.

WHAT A JOURNAL NEEDS. Fluent reads everything by fixed name from the case folder, so
after copying, the folder is also checked for the files the mesher should have left:

    mesh.msh              read by both journals
    inlet_centroids.csv   read by the UDF
    flowsplit_ratio.txt   read by the bifurcated journal only

A case missing one is reported, not silently marked ready.

CASE LIST. A run over a whole folder also writes case_list.txt at its root: one ready
case folder name per line, sorted. The HPC job array (ansys_files/hpc_bash.sh) runs
line N as sub-job N, so case folders keep their own names. It is only written when
every folder was considered -- a run restricted with --cases would otherwise replace a
full list with a partial one.

Run on a folder of meshed cases (this is also called automatically by
generate_cfd_volume_meshes.py after each case is meshed):

    python -m cfd_mesher.fluent_setup                          # every case under the volume root
    python -m cfd_mesher.fluent_setup --root /path --cases A B --overwrite
"""

import argparse
import json
import os
import shutil

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ANSYS_FILES_DIR = os.path.join(PACKAGE_DIR, "ansys_files")

COMMON_FILES = ("parabolic_vinlet_udf_full.c", "udf_waveform_5cycles.txt")
JOURNALS = {"bifurcated": "jnl_bifurcated", "sidewall": "jnl_sidewall"}
JOURNAL_NAME = "jnl_transient"

# files Fluent reads from the case folder that the volume mesher, not this module, writes
REQUIRED_FROM_MESHER = {
    "bifurcated": ("mesh.msh", "inlet_centroids.csv", "flowsplit_ratio.txt"),
    "sidewall": ("mesh.msh", "inlet_centroids.csv"),
}

FUSION_INFO_FILENAME = "ghd_forward_fusion_info.npz"
# aneurysm_type -> configuration; types 1 and 2 share the sidewall canonical
TYPE_TO_CONFIG = {0: "bifurcated", 1: "sidewall", 2: "sidewall"}


def detect_configuration(case_dir, npz_filename=FUSION_INFO_FILENAME):
    """'bifurcated' or 'sidewall', from the branch count in the fusion npz.

    Three branches (inlet + 2 outlets) is bifurcated, two is sidewall. meta.json's
    aneurysm_type, when the folder has one, must agree.
    """
    import numpy as np

    npz_path = os.path.join(case_dir, npz_filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError("no %s in %s" % (npz_filename, case_dir))
    n = len(np.load(npz_path, allow_pickle=True)["cpcd_glo"])
    if n == 3:
        config = "bifurcated"
    elif n == 2:
        config = "sidewall"
    else:
        raise RuntimeError("%s has %d branches; expected 2 (sidewall) or 3 (bifurcated)"
                           % (npz_path, n))

    meta_path = os.path.join(case_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            atype = json.load(f).get("aneurysm_type")
        if atype is not None and TYPE_TO_CONFIG.get(int(atype)) != config:
            raise RuntimeError(
                "%s: fusion npz has %d branches (%s) but meta.json says aneurysm_type %s"
                % (case_dir, n, config, atype))
    return config


def install_fluent_files(case_dir, overwrite=False, npz_filename=FUSION_INFO_FILENAME,
                         ansys_dir=ANSYS_FILES_DIR):
    """Copy the Fluent files into one case folder. Returns a report dict.

    Existing copies are left alone unless `overwrite`. `ready` is True only when the
    folder then holds every file its journal and the UDF read.
    """
    config = detect_configuration(case_dir, npz_filename)
    copied, kept = [], []
    targets = [(name, name) for name in COMMON_FILES] + [(JOURNALS[config], JOURNAL_NAME)]
    for src_name, dst_name in targets:
        src = os.path.join(ansys_dir, src_name)
        if not os.path.exists(src):
            raise FileNotFoundError("missing ANSYS template %s" % src)
        dst = os.path.join(case_dir, dst_name)
        if os.path.exists(dst) and not overwrite:
            kept.append(dst_name)
            continue
        shutil.copy2(src, dst)
        copied.append(dst_name)

    missing = [f for f in REQUIRED_FROM_MESHER[config]
               if not os.path.exists(os.path.join(case_dir, f))]
    return {"case": os.path.basename(os.path.normpath(case_dir)), "configuration": config,
            "journal_source": JOURNALS[config], "copied": copied, "already_present": kept,
            "missing_mesher_outputs": missing, "ready": not missing}


def main():
    from .config import VOLUME_SAVE_ROOT

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=VOLUME_SAVE_ROOT,
                   help="folder of volume-meshed case folders")
    p.add_argument("--cases", nargs="*", default=None,
                   help="only these case names (default: every folder under --root)")
    p.add_argument("--npz_filename", default=FUSION_INFO_FILENAME)
    p.add_argument("--include_unmeshed", action="store_true",
                   help="also set up cases with no mesh.msh yet (default skips them, so a "
                        "run in progress is not reported as a wall of missing files)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace Fluent files already present")
    p.add_argument("--case_list", default="case_list.txt",
                   help="file written at --root listing every ready case, one per line, "
                        "for the HPC job array; '' to skip")
    args = p.parse_args()

    cases = args.cases or sorted(d for d in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, d)))
    counts = {"bifurcated": 0, "sidewall": 0}
    ready = []
    skipped_unmeshed, not_ready, errors = [], [], []
    for case in cases:
        case_dir = os.path.join(args.root, case)
        if not os.path.isdir(case_dir):
            continue
        if not args.include_unmeshed and not os.path.exists(os.path.join(case_dir, "mesh.msh")):
            skipped_unmeshed.append(case)
            continue
        try:
            report = install_fluent_files(case_dir, overwrite=args.overwrite,
                                          npz_filename=args.npz_filename)
        except Exception as exc:
            errors.append((case, repr(exc)))
            print("[%s] error: %r" % (case, exc))
            continue
        counts[report["configuration"]] += 1
        if report["ready"]:
            ready.append(case)
        else:
            not_ready.append((case, report["missing_mesher_outputs"]))
            print("[%s] %s, still missing %s" % (case, report["configuration"],
                                                report["missing_mesher_outputs"]))

    if args.case_list and args.cases is None:
        path = os.path.join(args.root, args.case_list)
        with open(path, "w") as f:
            f.write("".join(c + "\n" for c in sorted(ready)))
        print("case list: %d ready case(s) -> %s" % (len(ready), path))
    elif args.case_list:
        print("case list not written: --cases restricts this run to a subset")

    total = sum(counts.values())
    print("set up %d case(s): %d bifurcated, %d sidewall | not ready %d | errors %d | "
          "skipped (no mesh.msh yet) %d"
          % (total, counts["bifurcated"], counts["sidewall"], len(not_ready), len(errors),
             len(skipped_unmeshed)))


if __name__ == "__main__":
    main()
