#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=24gb
#PBS -lwalltime=24:0:0

# Export Fluent cases to .pt and upload them to Hugging Face in the same job.
# The job copies export_n_upload.py from SCRIPT_DIR into $TMPDIR and runs it there.
# The run configuration is passed with qsub -v; anything left out keeps the default at
# the top of export_n_upload.py.
#
#   HPC_USER       user in /rds/general/user/<HPC_USER>/ephemeral/Angioflow/<PROJECT>
#   PROJECT        project folder; CASE_ROOT and SAVE_ROOT (<CASE_ROOT>/processed) follow
#   CASE_PATTERN   e.g. 'cohort_*'; none = every case folder
#   REPO_ID        e.g. whding123/Angioflow_fix
#   OVERWRITE_HF   true / false: replace case folders already on the Hub
#   SANITY         true / false: render mesh PNG + WSS/velocity GIFs
#   UPLOAD_SANITY  true / false: also upload the renders to cfd_data/sanity/<case>/
#
#   export HF_TOKEN=hf_...              # this shell only, never in a file
#   qsub -v HF_TOKEN export_n_upload.sh
#   qsub -v "HF_TOKEN,PROJECT=AneuGv2,CASE_PATTERN=cohort_*,REPO_ID=whding123/AneuGv2,OVERWRITE_HF=true" export_n_upload.sh
#   qsub -v "HF_TOKEN,HPC_USER=wd123,PROJECT=AneuX_fix,CASE_PATTERN=none,REPO_ID=whding123/Angioflow_fix,SANITY=true,UPLOAD_SANITY=true" export_n_upload.sh
#   qsub -v "HF_TOKEN,HPC_USER=wd123,PROJECT=Angioflow_fix,CASE_PATTERN=none,REPO_ID=whding123/Angioflow_fix,SANITY=true,UPLOAD_SANITY=true" export_n_upload.sh

# Quote the -v list (a bare * would be expanded by your shell) and keep commas out of the
# values: qsub splits the list on them. Re-submit after a walltime kill: exported cases
# and cases already on the Hub are skipped.

SCRIPT_DIR=$HOME/Angioflow/export   # where export_n_upload.py lives

echo "start dir: $(pwd)  PBS_O_WORKDIR: $PBS_O_WORKDIR  TMPDIR: $TMPDIR"
cp "$SCRIPT_DIR/export_n_upload.py" "$TMPDIR/" || exit 1
cd "$TMPDIR"

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set: submit with  qsub -v HF_TOKEN export_n_upload.sh"
    exit 1
fi

bool_flag() {   # bool_flag <name> <value> -> --name / --no-name
    case "${2,,}" in
        true|1|yes)  echo "--$1" ;;
        false|0|no)  echo "--no-$1" ;;
        *)           echo "bad value for $1: '$2' (use true or false)" >&2; exit 1 ;;
    esac
}

args=()
[ -n "$HPC_USER" ] && args+=(--user "$HPC_USER")
[ -n "$PROJECT" ]  && args+=(--project "$PROJECT")
[ -n "$REPO_ID" ]  && args+=(--repo-id "$REPO_ID")
if [ -n "$CASE_PATTERN" ]; then
    [ "${CASE_PATTERN,,}" = "none" ] && args+=(--case-pattern=) \
                                     || args+=(--case-pattern "$CASE_PATTERN")
fi
if [ -n "$OVERWRITE_HF" ]; then f=$(bool_flag overwrite-hf "$OVERWRITE_HF") || exit 1; args+=($f); fi
if [ -n "$SANITY" ];       then f=$(bool_flag sanity "$SANITY") || exit 1; args+=($f); fi
if [ -n "$UPLOAD_SANITY" ]; then f=$(bool_flag upload-sanity "$UPLOAD_SANITY") || exit 1; args+=($f); fi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate new

echo "python export_n_upload.py ${args[*]}"
python export_n_upload.py "${args[@]}"

# qstat -u $USER
