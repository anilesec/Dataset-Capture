from dis import dis
import os
import re
from this import d
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


if __name__ == "__main__":
    RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

    import argparse

    parser = argparse.ArgumentParser("project GT pcd to all frames of a given seq")
    parser.add_argument('--sqn', type=str, default=None,
                        help='seq name')

    args = parser.parse_args()
    print("args:", args)

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))
    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        all_sqns = [args.sqn]

    for sqn in all_sqns:
        print(f'sqn: {sqn}')
        # load GT 2d jts to compute hand bbox
        print('Loading annotation 2d & 3d jts...')
        sqn_dir = osp.join(RES_DIR, sqn)
        all_jts2d_ann, all_jts3d_ann = load_ann_jts(sqn_dir)

        all_jts2d_ann_disp, all_jts3d_ann_disp = load_ann_jts(sqn_dir, disp=True)
        
        all_imgs_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'rgb/*.png')))
        
        print('Loading all images...')
        all_imgs = np.array(
                [cv2.imread(imp)[:, :, ::-1] for imp in tqdm(all_imgs_pths)]
            )

        # print('Loading dope dets...')
        # all_dope_pkls_dir = osp.join(RES_DIR, sqn, 'dope_dets')
        # all_jts2d_dope, all_jts3d_dope = load_dope_poses(dets_pkls_dir=all_dope_pkls_dir)

        # print('Loading hocnet jts...')
        # all_hocnet_jts3d_txts = sorted(glob.glob(osp.join(RES_DIR, sqn, 'jts3d_hocnet/*.txt')))
        # all_jts3d_hocnet = np.array(
        #     [np.loadtxt(pth) for pth in tqdm(all_hocnet_jts3d_txts)]
        # )

        print('Creating viz...')
        save_pth = os.path.join('./out/vid_jts3d_ann_disp.mp4')
        # bb()
        create_juxt_vid(filepath=save_pth, inp_imgs=all_imgs, jts_order='OURS',
                        all_2d_jts=all_jts2d_ann_disp, all_3d_jts_rt=all_jts3d_ann_disp,
                        all_3d_jts_cam=all_jts3d_ann, all_3d_jts_prcst_algnd=None)
    
        # bb()
    