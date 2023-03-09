import os
import glob
from evaluation.eval_recon import saveas_pkl
import numpy as np
import os, sys, glob
from ipdb import set_trace as bb
import argparse
import pprint, pickle
from tqdm import tqdm
import pathlib
osp = os.path

from evaluation.pnps import *
from evaluation.eval_utils import *
from evaluation.viz_utils import *

def ours2hocnet(**kwargs):
        return np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None,
                        help='seq id')
    args = parser.parse_args()

    RES_DIR = '/scratch/1/user/aswamy/data/hand-obj'

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        all_sqns = [args.sqn]

    from tqdm import tqdm
    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)

        j3d_gt = np.loadtxt(osp.join(RES_DIR, sqn, 'jts3d.txt'))[ours2hocnet()]
        j3d_frm0_cam = np.loadtxt(osp.join(RES_DIR, sqn, 'dope_jts3d_cam_frm1.txt'))

        j3d_frm0_cam_algnd2gt, trnsfm = compute_similarity_transform(j3d_frm0_cam.T, j3d_gt.T, True, mu_idx=0, scale=1.0)

        # trnsfm_mat = np.eye(4)
        # trnsfm_mat[:3, :3] = trnsfm['rot']
        # trnsfm_mat[:3, 3] = trnsfm['tran'].flatten()

        # new_verts = trnsfm['scale'] * trnsfm['rot'].dot(np.array(rm.points).T) + trnsfm['tran']
        trnsfm_save_pth = osp.join(RES_DIR, sqn, 'dope_frm1_to_gt_trnsfm.pkl')
        saveas_pkl(trnsfm, trnsfm_save_pth)
        print('saved:', trnsfm_save_pth)
    print('done')

        


