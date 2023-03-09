#!/bin/bash
#SBATCH --output=/scratch/1/user/fbaradel/logs/slurm/%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fabien.baradel@naverlabs.com
#SBATCH --cpus-per-task 4
#SBATCH --mem 10G

cd /home/fbaradel/Hands/ROAR
source posebert/setup.sh

python "${@}"

#### Other useful params
#####SBATCH --output=/tmp-network/user/fbaradel/logs/slurm/%j.log
#### #SBATCH --cpus-per-task 1
#### #SBATCH -p gpu-mono
#### #SBATCH --gres=gpu:1
#### #SBATCH --mem 30G
#### #SBATCH --constraint="gpu_v100|gpu_p100|gpu_p40|gpu_32g|gpu_a100|gpu_40g"