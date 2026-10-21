#!/bin/bash
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:05:00
#PBS -J 1-3

# Dry run of hpc_bash_dispatch.sh: prints what each sub-job would see, runs nothing.
#
#   qsub -J 3-5 test_array.sh
#   qsub -J 3-5 -v CASE_LIST=$HOME/Angioflow/AneuGv2/case_list.txt test_array.sh
#
# Each sub-job writes test_array.sh.o<jobid>.<index> in the folder qsub was run from.

Case_List=${CASE_LIST:-$HOME/Angioflow/AneuGv2/case_list.txt}

echo "array index : ${PBS_ARRAY_INDEX}"
echo "job id      : ${PBS_JOBID}"
echo "user        : ${USER}"
echo "start dir   : $(pwd)"
echo "PBS_O_WORKDIR: ${PBS_O_WORKDIR}"
echo "TMPDIR      : ${TMPDIR}"
echo "case list   : ${Case_List}"

if [ ! -f "$Case_List" ]; then
    echo "case list not found"
    exit 1
fi
Sim_Folder_Name=$(sed -n "${PBS_ARRAY_INDEX}p" "$Case_List" | tr -d '\r[:space:]')
echo "case        : ${Sim_Folder_Name:-<none on line ${PBS_ARRAY_INDEX}>}"
