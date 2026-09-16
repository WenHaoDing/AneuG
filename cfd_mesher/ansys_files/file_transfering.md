

<!-- rsync -ahz --partial --info=progress2 \
    /media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2 \
    wd123@dtn-c.cx3.hpc.ic.ac.uk:/rds/general/user/wd123/home/Angioflow/ -->



# 1. Copy the CFD cases to your hpc home drive (run the following bash on ws2, change <your_username> to your username)
```bash
rsync -ahz --partial --info=progress2 \
    /media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2 \
    <your_username>@dtn-c.cx3.hpc.ic.ac.uk:/rds/general/user/<your_username>/home/Angioflow/
```

# 2. Access your HPC drive in VSCODE.
## 2.1 Use SSH in VSCODE to open your hpc drive
ssh location: login.cx3.hpc.imperial.ac.uk
## 2.2 Find this file:
/rds/general/user/<your_username>/home/Angioflow/AneuGv2/0000_array_log/hpc_bash_dispatch.sh
## 2.3 Update this .sh file
- On line 6, change to the start and end id you picked. 
- On lin 29, change <your_username> to your user name.


# 3. Run array jobs on hpc
```bash
cd /rds/general/user/<your_username>/home/Angioflow/AneuGv2/0000_array_log
qsub hpc_bash_dispatch.sh
```

# 4. Check running status
```bash
qstat -u $USER
```
After submitting the jobs for a few hours, you can run this bash command to check running status. If you see 'B', that means things are fine and cases are running.

# 5. Extract data.
First, you will need to create a conda env called 'new' in order to run the data extraction script. You need: numpy, torch, and ensightreader.
Follow instructions on: https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/conda/
It is recommended that you contact wenhao to let him do it for you.
Then, you need to change <your_username> in /rds/general/user/<your_username>/home/Angioflow/AneuGv2/0000_array_log/main.py to your user name.
Then, you can run
```bash
cd /rds/general/user/<your_username>/home/Angioflow/AneuGv2/0000_array_log
qsub export.sh
```
This will create a folder where you can find the extracted data:
/rds/general/user/<your_username>/home/Angioflow/AneuGv2/processed




# 1. Copy the CFD cases to your hpc home drive (run the following bash on ws2, change <your_username> to your username, I suggest running this in a tmux session, as it can take hours)
```bash
rsync -ahz --partial --info=progress2 \
    /media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2 \
    <your_username>@dtn-c.cx3.hpc.ic.ac.uk:/rds/general/user/<your_username>/home/Angioflow/
```

# 2. Open a tmux session on ws2, and SSH your hpc home drive
```bash
tmux new -s <your_tmux_session_name>
```
This creates a tmux session, then in the tmux session, run SSH command:
```bash
ssh <your_username>@login.cx3.hpc.imperial.ac.uk
```
Once this is done, could you kindly inform me the name of your tmux session?
