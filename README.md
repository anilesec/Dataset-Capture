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

$~$
$~$
$~$


# Evaluation
$~$
$~$
$~$


- **3D joints evaluation of estimations over our annotations**

|               | MPJPE (mm) | MPJPE_PA (mm) |MPJPE* (mm) | MPJPE_PA* (mm)
|---------------|-------|----------|--------|----------|
| dope          | 39.028 |  19.167 | 38.905   | 19.0439
| hocnet        | 42.38  | 21.928  | 42.217   |21.8191 |        
| posebert      |        |         |        |       |
| frankmocap    |        |         |         |       |

Note: 
 -  models are evaluated over ~85K frames
 - '*' indicates evaluation with dispalced 3d joints
 - Dope(85357); Hocnet(87540)


$~$
$~$
$~$


-  **Noise vs err on ann jts3d**


|noise | rot_err(deg) | tran_err(mm) | rep_err(pix)|
|-----|-------|----------|----------|
| 1mm |0.745558 | 7.403 |1.93 | 
| 2mm |1.515083 | 19.799 |3.86 | 
|3mm | 2.025938| 20.191| 5.778|
|4mm|2.898119| 36.861 |7.69|
|5mm|3.006424|30.012 | 9.54|



$~$
$~$
$~$

- **Rel-pose estimation evaluation** 

| sqn           | rep_err (pix)| RRRE (deg)  | RRTE (m)    
|---------------|-------|----------|-----------|
|20220705173214 | 3.373 | 28.517615 | 0.238575|
|20220907152036| 3.1897 | 23.406612|0.172552|
|20220907153810| 3.78 | 12.989140 | 0.124432 | 
|20220907152036| 3.189| 23.406612 | 0.172552 |
|20220907153810| 3.7752| 12.98914 | 0.124432 |










