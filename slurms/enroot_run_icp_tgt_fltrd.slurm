#!/bin/bash
#SBATCH -n 1
#SBATCH --constraint=enroot 
#SBATCH --partition gpu
#SBATCH --cpus-per-task=8

# mandatory for enroot
source /etc/proxyrc;
mkdir /tmp/slurm_`id -u`
chmod 700 /tmp/slurm_`id -u`
export XDG_RUNTIME_DIR=/tmp/slurm_`id -u` 

sqn=$1
enroot start --root  --mount /scratch --mount /gfs-ssd alphapose -- python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/run_icp_tgt_fltrd.py > /scratch/1/user/$(whoami)/logfiles/run_icp_tgt_fltrd_"$sqn".log 2>&1
