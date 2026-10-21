"""Upload files that are new on disk to the whding123/AneuGv2 dataset, under geometry/.

The repo mirrors the local case folders: geometry/<case>/<file>. This script compares the
two and uploads only what the repo does not have yet -- it never re-uploads or overwrites a
file already there, and it never touches anything outside geometry/.

Excluded no matter what:
    mesh.msh, mesh.vtu        volume meshes, too large and not part of the dataset
    anything not in a case folder (logs, case lists, scripts at the cohort root)

Uploads are batched into commits of --batch files, since one huge commit is slow and
fails as a whole. Re-running after an interruption is safe: anything that made it into
the repo is no longer "new" and is skipped.

    python huggingface/upload_aneugv2_new_files.py             # dry run: list what is new
    python huggingface/upload_aneugv2_new_files.py --upload    # actually upload
"""
import argparse
import collections
import os
from pathlib import Path

REPO_ID = "whding123/AneuGv2"
REPO_TYPE = "dataset"
PREFIX = "geometry"
LOCAL_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2"
CASE_GLOB = "cohort_650_*"
EXCLUDE_NAMES = {"mesh.msh", "mesh.vtu"}


def local_files(root):
    """{path_in_repo: local_path} for every file in a case folder, minus exclusions."""
    out = {}
    for case_dir in sorted(Path(root).glob(CASE_GLOB)):
        if not case_dir.is_dir():
            continue
        for f in sorted(case_dir.iterdir()):
            if f.is_file() and f.name not in EXCLUDE_NAMES and not f.name.startswith("."):
                out[f"{PREFIX}/{case_dir.name}/{f.name}"] = f
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=LOCAL_ROOT)
    p.add_argument("--upload", action="store_true", help="without this, only a dry run")
    p.add_argument("--batch", type=int, default=400, help="files per commit")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these file names, e.g. --only cap_masks.npz uncap_sanity.png")
    a = p.parse_args()

    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    remote = set(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    local = local_files(a.root)
    new = {k: v for k, v in local.items()
           if k not in remote and (a.only is None or v.name in a.only)}

    by_name = collections.Counter(Path(k).name for k in new)
    size = sum(v.stat().st_size for v in new.values())
    print("repo %s: %d files | local (excluding %s): %d | NEW: %d files, %.1f MB"
          % (REPO_ID, len(remote), ", ".join(sorted(EXCLUDE_NAMES)), len(local), len(new),
             size / 1e6))
    for name, n in by_name.most_common():
        print("   %5d x %s" % (n, name))
    if not new:
        print("nothing new to upload")
        return
    if not a.upload:
        print("dry run -- re-run with --upload to send these")
        return

    items = sorted(new.items())
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in chunk]
        names = collections.Counter(Path(k).name for k, _ in chunk)
        api.create_commit(repo_id=REPO_ID, repo_type=REPO_TYPE, operations=ops,
                          commit_message="Add %s (%d files)"
                                         % (", ".join(sorted(names)), len(chunk)))
        print("committed %d / %d" % (min(i + a.batch, len(items)), len(items)), flush=True)
    print("done")


if __name__ == "__main__":
    main()
