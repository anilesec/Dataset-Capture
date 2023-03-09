#!/bin/bash
#SBATCH -n 1
#SBATCH --constraint=enroot 
#SBATCH --partition gpu
#SBATCH --cpus-per-task=1

# mandatory for enroot
source /etc/proxyrc;
mkdir /tmp/slurm_`id -u`
chmod 700 /tmp/slurm_`id -u`
export XDG_RUNTIME_DIR=/tmp/slurm_`id -u` 

# enroot start --root  --mount /scratch --mount /gfs-ssd alphapose -- python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/outlier_remove.py \
#  --inp_pcds_dir /scratch/1/user/aswamy/data/hand-obj/20220909152911/fgnd_xyz_wc \
#  --seq_dir /scratch/1/user/aswamy/data/hand-obj/20220909152911 > /gfs-ssd/user/$(whoami)/test.log 2>&1

enroot start --root  --mount /scratch --mount /gfs-ssd alphapose -- python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/L515_data_interface.py \
 --inp_seq_dir $1 --sqn $2 > /gfs-ssd/user/$(whoami)/test.log 2>&1
