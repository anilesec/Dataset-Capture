# PoseBERT

## Setup the environnement
The commad above will launch the appropriate conda anv and set the correct environnement variables.
```
source posebert/setup.sh
```

## Skeleton normalization with pytorch
We re-impelement the skeleton normalization (center, extract relative pose and root rotation).
```
ipython posebert/skeleton.py
```
This should create an image img.jpg where you see the input skeleton and the relative hand skeleton using numpy and torch

## Rendering
Simple renderer and we also make sure that we are able to project 3d points into the 2d image plane
```
ipython posebert/renderer.py
```
This should create an image img.jpg where you see the mesh and the mesh plus the projected 3d joints with the bones.

## Rendering of SHOWme video
```
# Debug
ipython posebert/renderer.py -- "show_video(seqname='20220705173214', video_dir='/scratch/1/user/fbaradel/showme/gt_mesh', image_size=640, t_start=0, t_end=32)"

# Jobs
sbatch -p gpu --gres=gpu:1 --mem 5G  posebert/bash.sh posebert/renderer.py "show_video(seqname='20220705173214', video_dir='/scratch/1/user/fbaradel/showme/gt_mesh', image_size=640, t_start=0, t_end=10000)"
```

## MANO annots
Extracting trajectories from InterHands
```
# Debug
ipython posebert/dataset.py -- "preprocess_mano(split='train', fps=30, debug=True, max_interval=16, min_seq_len=32)"

# Jobs
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/dataset.py "preprocess_mano(split='train', fps=30, debug=False, max_interval=8, min_seq_len=256)"
#python posebert/dataset.py "preprocess_mano(split='train', fps=30, debug=False, max_interval=16, min_seq_len=256)"
python posebert/dataset.py "preprocess_mano(split='test', fps=30, debug=False, max_interval=16, min_seq_len=256)"
python posebert/dataset.py "preprocess_mano(split='val', fps=30, debug=False, max_interval=16, min_seq_len=256)"
```

## Dataset
Implementation of a dataset for Anil project.
We sample a hand_pose from Interhand and we synthetically generate a sequence of global_orient and translation.
```
ipython posebert/dataset.py -- "test()"
```
This should generate a video video.mp4 where you see the mesh and the projected 3d joints in the image plane.

## PoseBERT model
Small adaptation of the initial PoseBert model.
We give j3d as input (translation included) and output a single hand_pose (average of the 6D representation), and sequences of global_orient and translation.
```
ipython posebert/model.py
```
This should generate an image img.jpg showing the output and the relative j3d hand pose.

## Demo pipeline
TODO load Anil pkl files and estimate the transl
```

```

## Training pipeline
```
# Debug
CUDA_VISIBLE_DEVICES=1 ipython posebert/train.py -- --overfit 1 --iter 1000 -j 0 --noise_3d 0.0001 --noise_transl 0.001 --masking 20 --random 10 --seq_len_anil 10 --mask_block 1 --noise_2d_loc 500 --noise_2d_disp 100 --random_block 0

# Simple auto-encoders
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --input_type j2d --name ae_j2d
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --input_type j3d --name ae_j3d
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --input_type j3dj2d --name ae_j3dj2d

# Denoising auto-encoder
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --noise_3d 0.0001 --noise_transl 0.001 --noise_2d_loc 500 --noise_2d_disp 100 --masking 30 --random 20 --input_type j3dj2d --name dae_j3dj2d_bigNoise
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --noise_3d 0.00001 --noise_transl 0.0001 --noise_2d_loc 500 --noise_2d_disp 100 --masking 30 --random 20 --input_type j3dj2d --name dae_j3dj2d
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --noise_3d 0.00001 --noise_transl 0.0001 --noise_2d_loc 50 --noise_2d_disp 50 --masking 20 --random 10 --input_type j3dj2d --name dae_j3dj2d_smallNoise

# Denoising auto-encoder with small noise
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --noise_3d 0.0000001 --noise_transl 0.000001 --noise_2d_loc 5 --noise_2d_disp 5 --masking 10 --random 5 --input_type j3dj2d --name dae_j3dj2d_smallNoiseBis

# More aggresive noise
ckpt=/tmp-network/user/fbaradel/projects/ROAR/logs/dae_j3dj2d_bigNoise/checkpoints/last.pt
sbatch -p gpu --gres=gpu:1 --mem 20G  posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 128  --noise_3d 0.0001 --noise_transl 0.001 --noise_2d_loc 500 --noise_2d_disp 100 --masking 30 --random 20 --input_type j3dj2d --name dae_j3dj2d_bigNoise_aggresive
# --pretrained_ckpt $ckpt

# Per block and very aggresive and longer sequences
ckpt=/tmp-network/user/fbaradel/projects/ROAR/logs/dae_j3dj2d_bigNoise/checkpoints/last.pt
sbatch -p gpu --gres=gpu:1 --mem 15G --cpus-per-task 4 posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 256  --noise_3d 0.0005 --noise_transl 0.005 --noise_2d_loc 600 --noise_2d_disp 150 --masking 40 --random 30 --mask_block 1 --input_type j3dj2d --name dae_j3dj2d_maskPerBlock
# --pretrained_ckpt $ckpt

# Similar as above but with 2d only
ckpt=/tmp-network/user/fbaradel/projects/ROAR/logs/dae_j3dj2d_bigNoise/checkpoints/last.pt
sbatch -p gpu --gres=gpu:1 --mem 15G --cpus-per-task 4 posebert/bash.sh posebert/train.py  --iter 10000 -j 8 --n_visu_to_save 1 --seq_len 256  --noise_3d 0.0005 --noise_transl 0.005 --noise_2d_loc 600 --noise_2d_disp 150 --masking 40 --random 30 --mask_block 1 --input_type j2d --name dae_j2d_maskPerBlock
# --pretrained_ckpt $ckpt

# Eval
ckpt=/tmp-network/user/fbaradel/projects/ROAR/logs/dae_j3dj2d_bigNoise/checkpoints/last.pt
ipython posebert/train.py -- --overfit 1 --seq_len 128 --save_dir logs --name debug_eval --n_visu_to_save 1 --pretrained_ckpt $ckpt --seq_len_anil 128 --eval_anil_only 1 --eval_anil 1 --eval_only 1
```

## Demo code for Anil
```
# Simple python script
ipython posebert/demo.py --  --ckpt /scratch/1/user/fbaradel/ROAR/logs/dae_j3dj2d_bigNoise_aggresive/checkpoints/last.pt --all_seq 1 --start 0 --seq_len 1000000
python posebert/demo.py  --ckpt /scratch/1/user/fbaradel/ROAR/logs/dae_j3dj2d_bigNoise_aggresive/checkpoints/last.pt --seqname 20220705173214 --take_left_hands_into_account 0 --start 100 --seq_len 128 --outdirname res_debug --all_seq 1
python posebert/demo.py  --ckpt /scratch/1/user/fbaradel/ROAR/logs/dae_j3dj2d_bigNoise_aggresive/checkpoints/last.pt --seqname 20220705173214 --take_left_hands_into_account 0 --start 100 --seq_len 128 --outdirname res_debug --all_seq 0 --render 1
ipython posebert/demo.py --  --dope_only 1 --shape dope --debug 1 --method dope --refine_posebert 0
ipython posebert/demo.py --  --dope_only 1 --shape dope --debug 1 --method dope_median_filtering --refine_posebert 0
ipython posebert/demo.py --  --dope_only 1 --shape dope --debug 1 --method posebert --refine_posebert 0
ipython posebert/demo.py --  --dope_only 1 --shape dope_median --debug 1 --method posebert_shape --refine_posebert 0
ipython posebert/demo.py --  --dope_only 1 --shape dope_median --debug 1 --method posebert_shape --refine_posebert 1

# Different methods
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method dope --refine_posebert 0 --pnp kabsch
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method dope_median_filtering --refine_posebert 0 --pnp kabsch
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method dope --refine_posebert 0 --pnp pnp
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method dope_median_filtering --refine_posebert 0 --pnp pnp

sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method dope_median_filtering_2d --refine_posebert 0
sbatch -p gpu --gres=gpu:1 --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope --debug 0 --method posebert --refine_posebert 0
sbatch -p gpu --gres=gpu:1 --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope_median --debug 0 --method posebert_shape --refine_posebert 0
sbatch -p gpu --gres=gpu:1 --mem 5G --cpus-per-task 1 posebert/bash.sh posebert/demo.py --dope_only 1 --shape dope_median --debug 0 --method posebert_shape --refine_posebert 1

# Re-order
ipython posebert/demo.py --  --order_sequence 1 --res_dir /scratch/1/user/fbaradel/ROAR/relative_camera_poses/dope_dope_kabsch_refinePoseBERTFalse
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh  posebert/demo.py --order_sequence 1 --res_dir /scratch/1/user/fbaradel/ROAR/relative_camera_poses/dope_dope_kabsch_refinePoseBERTFalse
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh  posebert/demo.py --order_sequence 1 --res_dir /scratch/1/user/fbaradel/ROAR/relative_camera_poses/dope_median_filtering_dope_kabsch_refinePoseBERTFalse
sbatch -p cpu --mem 5G --cpus-per-task 1 posebert/bash.sh  posebert/demo.py --order_sequence 1 --res_dir /scratch/1/user/fbaradel/ROAR/relative_camera_poses/posebert_dope_kabsch_refinePoseBERTFalse

```
