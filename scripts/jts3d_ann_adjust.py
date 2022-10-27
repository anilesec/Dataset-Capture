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
import open3d as o3d


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

    for sqn in tqdm(all_sqns):
        print(f'sqn: {sqn}')

        # load gt pcd
        gt_pcd_pth = osp.join(RES_DIR,  sqn, 'gt_mesh/xyz/tgt_pcd.ply')
        pcd = o3d.io.read_point_cloud(gt_pcd_pth)
        # bb()
        verts = np.array(pcd.points)
        normals = np.array(pcd.normals)

        # load jts3d
        jts3d_pth = osp.join(RES_DIR, sqn, 'jts3d.txt')
        jts3d = np.loadtxt(jts3d_pth)
        # bb()
        jts3d_inds = []
        for j in jts3d:
            ind = verts.tolist().index(j.tolist())
            jts3d_inds.append(ind)
        jts3d_nrmls = verts[jts3d_inds]

        # compute new points with offset displacement
        disp = 0.005 # 5mm
        jts3d_disp = jts3d - disp * jts3d_nrmls

        # save displaced jts3d
        jts3d_disp_save_pth = osp.join(RES_DIR, sqn, 'jts3d_disp.txt')
        np.savetxt(jts3d_disp_save_pth, jts3d_disp)

        print(f"saved here: {jts3d_disp_save_pth}")
    print('Done!')








