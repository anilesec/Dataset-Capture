import numpy as np
import os, sys, glob
from ipdb import set_trace as bb
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image
from evaluation.viz_utils import *
from tqdm import tqdm
import pathlib
from evaluation.eval_utils import *

osp = os.path
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

def trnsfm_jts3d_to_all_frms(jts3d_pth, all_frms_poses_pths, save_dir):
    jts3d_ann = np.loadtxt(jts3d_pth)
    for idx, posp in tqdm(enumerate((all_frms_poses_pths))):
        pose = np.linalg.inv(np.loadtxt(posp))
        fname = osp.basename(osp.dirname(posp))

        jts3d_frm_save_pth = osp.join(save_dir, f'{fname}.txt')
        os.makedirs(osp.dirname(jts3d_frm_save_pth), exist_ok=True)

        jts3d_frm = tform_points(T=pose, X=jts3d_ann)
        np.savetxt(jts3d_frm_save_pth, jts3d_frm)
    
    return None


if __name__ == "__main__":
    print(f"This scripts transforms the annotated 3d joints to all frames using ann poses")
    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    print(f"All seq ids: \n {all_sqns}")

    for sqn in tqdm(all_sqns):
        print(f"seq: {sqn}")

        # get 3d jts
        jts3d_pth = pathlib.Path(f'{RES_DIR}/{sqn}/jts3d.txt')

        missing_jts3d_seqs = []
        if not jts3d_pth.exists():
            print(f'Missing jts3d.txt for sqn:{jts3d_pth}')
            missing_jts3d_seqs.append(sqn)
            continue

        # get all anno poses
        all_frms_poses_pths = sorted(glob.glob(f'{RES_DIR}/{sqn}/icp_res/*/f_trans.txt'))

        save_dir = osp.join(RES_DIR, sqn, 'jts3d')

        # transform jts3d ann to all frames
        trnsfm_jts3d_to_all_frms(jts3d_pth, all_frms_poses_pths, save_dir)
        print(f"saved here: {save_dir}")
    print(f'Missing jts3d sequences: {missing_jts3d_seqs}')
    print('Done!!')
