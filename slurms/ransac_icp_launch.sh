#!/bin/bash
dirs1="20220805164755 20220829155218 20220902153221 20220824154342 20220830162330 20220809161015 20220905151029 20220905155551 20220905153946 20220824104203 20220913145135 20220812180133 20220907153810 20220905154829 20220905111237 20220902154737 20220913151554 20220909114359 20220912144751 20220909145039 20220909140016 20220902163904 20220811170459 20220705173214 20220912155637 20220824142508 20220905140306 20220824160141 20220805165947 20220902114024"
dirs2="20220830161218 20220902111535 20220902104048 20220909151546 20220824152850 20220912161700 20220909111450 20220824150228 20220913153520 20220824105341 20220811172657 20220912160620 20220909113237 20220823113402 20220902111409 20220809163444 20220819155412 20220824181949 20220909142411 20220912151849 20220902151726 20220811165540 20220811163525 20220907155615 20220909134639 20220909120614 20220912143756 20220905105332 20220902170443 20220905112733"
dirs3="20220913144436 20220823115809 20220902110304 20220902163950 20220912164407 20220819162529 20220823114538 20220905142354 20220812170512 20220809171854 20220829154032 20220912165455 20220913154643 20220811171507 20220909115705 20220824155144 20220830163143 20220909152911 20220824144438 20220902164854 20220905112623 20220907152036 20220905141444 20220812174356 20220912161552 20220909141430 20220824180652 20220909121541 20220819164041 20220912142017"
dirs4="20220912152000 20220809170847 20220824102636 20220902115034 20220812172414 20220811154947"

# for colmap ransac ipc
for d in $dirs4;
  do
    echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_vh_recon.slurm  $d"
    sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_vh_recon.slurm  $d
  done




# for VH recon eval
# for d in $dirs4;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_vh_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_vh_recon.slurm  $d
#   done



# for briach recon eval
# for d in $dirs1;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_recon.slurm  $d
#   done


# for briac dope recon eval
# for d in $dirs1;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d
#   done
# for d in $dirs2;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d
#   done
# for d in $dirs3;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d
#   done
# for d in $dirs4;
#   do
#     echo "sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d"
#     sbatch /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/slurms/eval_briac_dope_recon.slurm  $d
#   done


