#!/bin/bash

# sh /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/maintainence/tar_and_store.sh   /scratch/1/user/aswamy/data/L515_seqs/20220805 /gfs/team/cv/Users/aswamy/dataset_backup/hand-obj

# list all dirs of given path
search_dir="$1"
dst_dir="$2"

# src_dir="/scratch/1/user/aswamy/data/hand-obj"  # source root dir
# dst_dir="/gfs/data/handobject/hand-obj" # target root dir
# dirs="20220801 20220804 20220805 20220809 20220811 20220812 20220819 20220822 20220823 20220824 20220826 20220829 20220830 20220902 20220905"  # list of folders to zip and copy
dst_dir="/gfs/data/handobject/hand-obj"
src_dir="/scratch/1/user/aswamy/data/hand-obj"
dirs="20220907152036 20220907153810 20220907155615 20220905112623"
for d in $dirs
    do 
        echo "tar cvf $src_dir/$d.tar $src_dir/$d"
        #tar cvf $src_dir/$d.tar $src_dir/$d

        echo "rsync --progress -auvz $src_dir/$d.tar $dst_dir/"
        rsync --progress -auvz $src_dir/$d.tar $dst_dir/
    done
# for entry in "$search_dir"/*
#     do
        # echo "tar cvf $entry.tar $entry"
        # tar cvf $entry.tar $entry
        
        # echo "rsync --progress -auvz  $entry.tar $2/"
        # rsync --progress -auvz  $entry.tar $2
    # done

# # find all dirs of given path and create .tar for each dir 
# echo "find $1 -maxdepth 1 -mindepth 1 -type d -exec tar cvf {}.tar {}  \;"
# find $1 -maxdepth 1 -mindepth 1 -type d -exec tar cvf {}.tar {}  \;

# # find all tar files and sync to dest folder
# echo "find $1 -maxdepth 1 -type f -name  '*.tar' -exec rsync --progress -auvz {} $2 \;"
# find $1 -maxdepth 1 -type f -name  '*.tar' -exec rsync --progress -auvz {} $2 \;

