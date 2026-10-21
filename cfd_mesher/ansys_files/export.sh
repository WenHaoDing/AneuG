#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=24gb
#PBS -lwalltime=12:0:0

cp -r $HOME/Angioflow/export/export_n_upload.py $TMPDIR/

cd $PBS_O_WORKDIR

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate new

## Verify install:
python export.py

# qstat -u $USER