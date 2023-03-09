#!/bin/bash

DATASET_PATH=/scratch/1/user/aswamy/data/colmap-hand-obj/$1

colmap feature_extractor --database_path $DATASET_PATH/database.db  --image_path $DATASET_PATH/images --ImageReader.mask_path $DATASET_PATH/masks --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.camera_params "899.783,900.019,653.768,362.143"

colmap exhaustive_matcher --database_path $DATASET_PATH/database.db --SiftMatching.use_gpu 1 

#colmap vocab_tree_matcher --database_path $DATASET_PATH/database.db --SiftMatching.use_gpu 1 --SiftMatching.gpu_index 0

colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse
