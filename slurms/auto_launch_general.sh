#!/bin/bash


dirs="20220913154643 20220913153520 20220913151554 20220913145135 20220913144436 20220912165455 20220912164407 20220912161700 20220912161552 20220912160620 20220912155637 20220912152000 20220912151849"

for d in $dirs
  do
    echo "sbatch /gfs-ssd/user/aswamy/github_repos/hand-obj-recon/slurms/general_cpu.slurm $d"
  done