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

inp_seq_dir=$1
sqn=$2
enroot start --root  --mount /scratch --mount /gfs-ssd alphapose -- python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/L515_data_interface.py \
 --inp_seq_dir $inp_seq_dir --sqn $sqn > /scratch/1/user/$(whoami)/logfiles/L515_data_interface_"$sqn".log 2>&1
