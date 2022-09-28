#!/bin/bash

mkdir -p "/scratch/1/user/aswamy/github_repos/Fast-Robust-ICP/data/$1/data_for_reg"
mkdir -p "/scratch/1/user/aswamy/data/hand-obj/$1/icp_res/0000000000"

touch /scratch/1/user/aswamy/data/hand-obj/$1/icp_res/0000000000/f_trans.txt

rsync --progress -auvz /scratch/1/user/aswamy/data/hand-obj/$1/gt_mesh/xyz/tgt_pcd.ply /scratch/1/user/aswamy/github_repos/Fast-Robust-ICP/data/$1/data_for_reg/
#rsync --progress -auvz /scratch/1/user/aswamy/data/hand-obj/$1/fgnd_xyz_wc_wn_wo_outlr/* /scratch/1/user/aswamy/github_repos/Fast-Robust-ICP/data/$1/data_for_reg/


