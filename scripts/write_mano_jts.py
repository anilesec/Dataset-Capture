import os
import re
from ipdb import set_trace as bb
import numpy as np
import pprint
osp =  os.path
import pathlib
from tqdm import tqdm
import glob
import cv2
import sys
from evaluation.eval_utils import *
from evaluation.viz_utils import *

def get_mano_jts_from_json(json_fpth):
    import json
    with open(json_fpth, 'rb') as f:
        data = json.load(f)

    return np.array(data['mano_joints']) / 1000. # convert to m

if __name__ == "__main__":
    RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')
    MANO_RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj/mano_fitting_res_latest')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqn", type=str, help='seq no.')
    args = parser.parse_args()

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        all_sqns = [args.sqn]

    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)

        # load mano anno
        mano_ann_pth = osp.join(MANO_RES_DIR, sqn, 'registration.json')
        jts3d_ann = get_mano_jts_from_json(mano_ann_pth)

        rgb_pths = glob.glob(osp.join(RES_DIR, sqn, 'rgb/*.png'))
        for imgp in rgb_pths:
            frm_name = osp.basename(imgp).split('.')[0]
            posp = osp.join(RES_DIR, sqn, f'icp_res/{frm_name}/f_trans.txt')
            pose = np.linalg.inv(np.loadtxt(posp))
            jts3d_ann_trnsfmd2frm = trnsfm_points(trnsfm=pose, pts=jts3d_ann)

            save_pth = osp.join(RES_DIR, sqn, f'mano_jts/{frm_name}.txt')
            os.makedirs(osp.dirname(save_pth), exist_ok=True)
            np.savetxt(save_pth, jts3d_ann_trnsfmd2frm)
    print('Done!!')




