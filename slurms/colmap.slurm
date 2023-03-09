#!/bin/bash
#SBATCH -n 1
#SBATCH --constraint=enroot
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --cpus-per-task=6
#SBATCH --exclude=tholus-27,mensa-01,chasma-01,chasma-02,chasma-03 
# mandatory for enroot
mkdir /tmp/slurm_`id -u`
chmod 700 /tmp/slurm_`id -u`
export XDG_RUNTIME_DIR=/tmp/slurm_`id -u`
# create the enroot environment
export ENROOT_DATA_PATH=/dev/shm/${{SLURM_JOB_ID}}
enroot create  /scratch/1/project/localization/enroot/sqsh/kapture-internal.sqsh --name localization+kapture-internal+20220410
source /etc/proxyrc
OUTPUT_DIR=/scratch/1/user/aswamy/logfiles
hostname >> ${{OUTPUT_DIR}}/slurm.log 2>&1
nvidia-smi >> ${{OUTPUT_DIR}}/slurm.log 2>&1
DATASET_PATH=/scratch/1/user/aswamy/data/colmap-hand-obj/$1
# enroot start --mount /home:/home --mount /gfs:/gfs --mount /gfs-ssd:/gfs-ssd --mount /scratch:/scratch --mount /nfs:/nfs --mount /local:/local localization+kapture-internal+20220410 bash /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/colmap_launch_gpu.sh 20220705173214 > /scratch/1/user/$(whoami)/logfiles/colmap_20220705173214.log 2>&1

enroot start --mount /home:/home --mount /gfs:/gfs --mount /gfs-ssd:/gfs-ssd --mount /scratch:/scratch \
 --mount /nfs:/nfs --mount /local:/local localization+kapture-internal+20220410 \
 bash /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/colmap_launch_gpu.sh $1 > \
 /scratch/1/user/$(whoami)/logfiles/colmap_$1.log 2>&1