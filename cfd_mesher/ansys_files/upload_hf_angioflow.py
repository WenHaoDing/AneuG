"""Upload processed CFD cases to the Hugging Face dataset whding123/Angioflow_fix.

Uploads <SAVE_ROOT>/<case>/ (blood_data.pt, wall_data.pt + metadata copied by export.py)
to <repo>/cfd_data/<case>/ for every case whose export finished, i.e. both .pt files exist.
Every folder under SAVE_ROOT is a case except _logs, _sanity and hidden folders, which
stay local.

By default nothing on the Hub is deleted: the upload is resumable and skips files that
are already there, so it can be re-run (or re-submitted after a walltime kill) as more
cases finish. Pass --wipe ONCE to delete and recreate the repo first, which drops all
history and LFS storage.

The write token is read from $HF_TOKEN only and must never be written into a script.
Pass it in when submitting, through PBS only (not on the login node):

    qsub -v HF_TOKEN=hf_... <job>.sh            # job runs: python upload_hf_angioflow.py
    python upload_hf_angioflow.py --dry-run     # list what would be uploaded
"""
from __future__ import annotations

import argparse
import os
import sys

import huggingface_hub
from huggingface_hub import HfApi

REPO_ID = "whding123/Angioflow_fix"
SAVE_ROOT = "/rds/general/user/jp3025/ephemeral/Angioflow/AneuGv2/processed"
PATH_IN_REPO = "cfd_data"  # cases land in <repo>/cfd_data/<case>/
REQUIRED = ("blood_data.pt", "wall_data.pt")


def complete_cases(root):
    """Case folders under root with both .pt files written."""
    cases, partial = [], []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith(("_", ".")) or not os.path.isdir(path):
            continue
        if all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED):
            cases.append(name)
        else:
            partial.append(name)
    return cases, partial


def wipe_repo(api, repo_id, private, assume_yes):
    """Delete the dataset repo (history and LFS included) and create it again empty."""
    if not assume_yes:
        if not sys.stdin.isatty():
            sys.exit("--wipe in a non-interactive job needs --yes as well")
        typed = input(f"This deletes ALL files and history of dataset {repo_id}.\n"
                      f"Type the repo id to confirm: ").strip()
        if typed != repo_id:
            sys.exit("confirmation did not match, nothing deleted")
    api.delete_repo(repo_id, repo_type="dataset", missing_ok=True)
    print(f"[wipe] deleted {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", private=private)
    print(f"[wipe] recreated {repo_id} (private={private})")


def dry_run(api, args, cases):
    """Report what a real run would do, without deleting or uploading anything.

    Getting this far already proves the token works and the node can reach the Hub.
    """
    from huggingface_hub.errors import RepositoryNotFoundError
    print(f"[dry] huggingface_hub {huggingface_hub.__version__}")
    try:
        on_hub = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
        print(f"[dry] {args.repo_id} exists, {len(on_hub)} file(s) on the Hub")
    except RepositoryNotFoundError:
        on_hub = set()
        print(f"[dry] {args.repo_id} does not exist, would be created")
    if args.wipe:
        print(f"[dry] --wipe: would DELETE {args.repo_id} with all {len(on_hub)} file(s) "
              f"and history, then recreate it empty")
        on_hub = set()
    for c in cases:
        files = sorted(os.listdir(os.path.join(args.root, c)))
        todo = [f for f in files if f"{PATH_IN_REPO}/{c}/{f}" not in on_hub]
        print(f"  {PATH_IN_REPO}/{c}: {len(files)} file(s), {len(todo)} to upload"
              + (f" ({', '.join(todo)})" if todo and len(todo) < len(files) else ""))
    print("[dry] nothing was changed on the Hub")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--root", default=SAVE_ROOT)
    p.add_argument("--wipe", action="store_true",
                   help="delete and recreate the repo before uploading (run once only)")
    p.add_argument("--yes", action="store_true", help="skip the --wipe confirmation prompt")
    p.add_argument("--private", action="store_true",
                   help="create the repo as private (only used with --wipe or if missing)")
    p.add_argument("--cases", nargs="*", help="upload only these case names")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set: pass it at submit time, e.g. qsub -v HF_TOKEN=hf_... "
                 "(never write it into a script)")

    cases, partial = complete_cases(args.root)
    if args.cases:
        unknown = sorted(set(args.cases) - set(cases))
        if unknown:
            sys.exit(f"not found / not complete under {args.root}: {unknown}")
        cases = [c for c in cases if c in set(args.cases)]
    if partial:
        print(f"[skip] {len(partial)} incomplete case(s): {', '.join(partial)}")
    if not cases:
        sys.exit("no complete cases to upload")

    size = sum(os.path.getsize(os.path.join(args.root, c, f))
               for c in cases for f in os.listdir(os.path.join(args.root, c)))
    print(f"[plan] {len(cases)} case(s), {size / 1e9:.1f} GB -> {args.repo_id}")

    if int(huggingface_hub.__version__.split(".")[0]) < 2:
        # before 2.0 only upload_large_folder is resumable, and it cannot target a subfolder
        sys.exit(f"huggingface_hub {huggingface_hub.__version__} is too old, need >= 2.0: "
                 f'pip install -U "huggingface_hub>=2"')

    api = HfApi(token=token)
    print(f"[auth] logged in as {api.whoami()['name']}")

    if args.dry_run:
        dry_run(api, args, cases)
        return

    if args.wipe:
        wipe_repo(api, args.repo_id, args.private, args.yes)
    else:
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    # Resumable, multi-commit upload into PATH_IN_REPO: a re-run after a kill skips files
    # already committed. Patterns match local paths (relative to --root).
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.root,
        path_in_repo=PATH_IN_REPO,
        repo_type="dataset",
        allow_patterns=[f"{c}/*" for c in cases],
        ignore_patterns=["_logs/*", "_sanity/*", ".cache/*"],
        commit_message=f"Upload {len(cases)} processed CFD cases to {PATH_IN_REPO}/",
    )
    print(f"[done] https://huggingface.co/datasets/{args.repo_id}/tree/main/{PATH_IN_REPO}")


if __name__ == "__main__":
    main()
