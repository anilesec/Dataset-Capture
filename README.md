# Dataset-Capture

## Setup the environnement
Activate appropiate conda env:
```
conda activate /nfs/tools/humans/conda/envs/datacap
```


## Pre-process(save rgb, pcds, etc.) the captured raw data from L515 sensor
Run the following script:
```
python scripts/process_raw_data.py --inp_seq_dir /tmp-network/user/aswamy/L515_seqs/2022061420220614171547 --save_base_dir /tmp-network/dataset/hand-obj --save_rgb  --save_pcd --seg_pcd --depth_thresh 0.8
```
P.S: read arguments help to pass appropriate commands

Note: 
*  "--save_depth" (for saving depth maps) and "--save_masks" (for saving forground masks) options are not 100% correct. They are not priority at the moment, but I will fix them later.

* --depth_thresh needs to be set for each sequence depending the distance to background while capturing, otherwise can lead to wrong pcd forground
