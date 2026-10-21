#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=16gb
#PBS -lwalltime=24:0:0

# Upload processed AneuGv2 cases to HF dataset whding123/AneuGv2, under cfd_data/<case>/.
# Default: upload only.
#
#   qsub -v MODE=dry upload_hf.sh       # dry run: checks token + Hub access, lists what
#                                       # would be uploaded; changes nothing
#   qsub -v MODE=drywipe upload_hf.sh   # dry run of the wipe: also reports what --wipe
#                                       # would delete; changes nothing
#   qsub -v MODE=wipe upload_hf.sh      # ONE TIME: delete + recreate repo, then upload
#   qsub upload_hf.sh                   # every later run: resumable upload, no wipe
#
# Re-submit (without MODE) after a walltime kill or when more cases finish; files already
# on the Hub are skipped. The token is passed at submit time, never stored in a file:
#   qsub -v HF_TOKEN=hf_... upload_hf.sh   (MODE goes in the same list: -v HF_TOKEN=hf_...,MODE=dry)

cd $HOME/Angioflow/AneuGv2/0000_array_log

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate new

# needs huggingface_hub >= 2.0 in env "new" (upload_hf.py checks and exits if older)

case "${MODE:-upload}" in
    dry)     python upload_hf.py --dry-run ;;
    drywipe) python upload_hf.py --dry-run --wipe ;;
    wipe)    python upload_hf.py --wipe --yes ;;
    upload)  python upload_hf.py ;;
    *)       echo "unknown MODE=$MODE (use dry, drywipe, wipe, or leave unset)"; exit 1 ;;
esac

# qstat -u $USER
