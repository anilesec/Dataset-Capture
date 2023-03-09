#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task 6
#####SBATCH --gres=gpu:1
#SBATCH --mem 64g
####SBATCH -p gpu
####SBATCH --constraint='gpu_32g|gpu_16g|gpu_22g'
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=anilkumar.swamy@naverlabs.com
#SBATCH --output=/scratch/1/user/aswamy/logfiles/L515_data_interface_%j.log
#SBATCH --error=/scratch/1/user/aswamy/logfiles/L515_data_interface_%j.err
#SBATCH --chdir=/gfs-ssd/user/aswamy/github_repos/Dataset-Capture
source ~/.bashrc
conda activate datacap
arg1=$1 # inp seq dir L515
arg2=$2 # sqn
arg3=$3 # start ind 
arg4=$4 # end ind
echo "python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/L515_data_interface.py --inp_seq_dir $arg1  --save_base_dir /scratch/1/user/aswamy/data/hand-obj --sqn $arg2"

python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/L515_data_interface.py --inp_seq_dir $arg1  --save_base_dir /scratch/1/user/aswamy/data/hand-obj --sqn $arg2
echo "Done!!"

