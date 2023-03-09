#!/bin/bash

# list all dirs of given path
search_dir="$1"
for entry in "$search_dir"/*
do
  echo "sbatch slurms/L515_data_interface.slurm $entry"
  sbatch slurms/L515_data_interface.slurm $entry 
done
