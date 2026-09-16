#!/bin/bash 

#asking for n nodes, 20 proc each 
#PBS -l select=1:ncpus=64:mem=72gb
#PBS -l walltime=24:00:00
#PBS -J 1-100

# CASE FOLDERS KEEP THEIR OWN NAMES. Sub-job N runs the case named on line N of
# case_list.txt, instead of a folder that had to be renamed case_N. Write the list once
# on the login node, in the run directory, before submitting:
#
#   cd $HOME/AneuG_CFD/AneuGv2
#   for d in */; do [ -f "$d/jnl_transient" ] && echo "${d%/}"; done > case_list.txt
#   qsub -J 1-$(wc -l < case_list.txt) hpc_bash.sh
#
# -J on the qsub line overrides the #PBS -J directive above. Submitting in chunks works
# against the same list: qsub -J 1-100, then -J 101-200, and so on. Do not rewrite the
# list while jobs from it are queued, or indices will point at different cases.

#Define the simulation name & directories name
Project_Name=AneuG_CFD
Run_Name=AneuGv2
ToolBox_Name=toolbox_transient_aneux
Journal_Name=jnl_transient
nodes=1
cpus=64
Log_File_Name=log
Results=Results
Time_Stamp=timeStamp.log
Save_dir=/rds/general/user/wd123/ephemeral/$Project_Name/$Run_Name
Case_List=$HOME/$Project_Name/$Run_Name/case_list.txt

# line PBS_ARRAY_INDEX of the list; tr drops a Windows line ending if the list was edited there
Sim_Folder_Name=$(sed -n "${PBS_ARRAY_INDEX}p" "$Case_List" | tr -d '\r[:space:]')

# An empty name would make every path below point at the run directory itself -- the
# copy would pull in all cases, and the skip check would look in the wrong place -- so
# stop before anything touches the filesystem.
if [ -z "$Sim_Folder_Name" ]; then
    echo "No case on line ${PBS_ARRAY_INDEX} of $Case_List"
    exit 1
fi
if [ ! -d "$HOME/$Project_Name/$Run_Name/$Sim_Folder_Name" ]; then
    echo "Case folder $Sim_Folder_Name (line ${PBS_ARRAY_INDEX}) not found"
    exit 1
fi
echo "Array index ${PBS_ARRAY_INDEX} -> $Sim_Folder_Name"

######################################################################
np=$(($nodes*$cpus))

# module load ansys/19.4-fluids
module load Ansys/2022R2_Fluids

export ANSYSLIC_DIR=1055@ansys.cc.ic.ac.uk
export ANSYSLMD_LICENSE_FILE=$ANSYSLIC_DIR
export MPI_TMPDIR=$TMPDIR

# control flow: whether target case has been run
if [ -d "$Save_dir/$Sim_Folder_Name" ] && [ -f "$Save_dir/$Sim_Folder_Name/$Results/ensight_files.geo" ]; then
    echo "Target case has been run"
    exit 0
else
    echo "Target case has not been run"
fi

cd $PBS_O_WORKDIR; 
cp -r $HOME/$Project_Name/$Run_Name/$Sim_Folder_Name $TMPDIR/

cd $TMPDIR/$Sim_Folder_Name
date >> $HOME/$Project_Name/$Run_Name/$Sim_Folder_Name/$Time_Stamp

fluent 3ddp -gu -t${np} -pshmem -ssh -i $TMPDIR/$Sim_Folder_Name/$Journal_Name > output

mkdir -p $Save_dir/$Sim_Folder_Name/$Results
find "$TMPDIR" -type f -name "output*" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "flowsplit_ratio.txt" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "inlet_centroids.csv" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "*.obj" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "ensight_files*" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "mesh.msh" -exec cp {} "$Save_dir/$Sim_Folder_Name" \;
find "$TMPDIR" -type f -name "*forward_fusion_info.npz" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;
find "$TMPDIR" -type f -name "branch_ranking.npy" -exec cp {} "$Save_dir/$Sim_Folder_Name/$Results" \;

date>> $Save_dir/$Sim_Folder_Name/$Time_Stamp


#rm -r $WORK/$Project_Name/$Run_Name/$Sim_Folder_Name/*.msh
#rm -r $WORK/$Project_Name/$Run_Name/$Sim_Folder_Name/*.vtu

# qstat -u $USER
# qsub array_transient_aneux.sh
