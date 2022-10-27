# Convert Midas depth into masks 
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import glob, copy
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
# import open3d as o3d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument('--sqn', type=str, required=True,
                        help='seq name')

    args = parser.parse_args()

    print("args:", args)
    seq = args.sqn
    base_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}'
    imgs_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}/rgb'
    dimgs_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}/midas_depth'
    dimgs_pths = sorted(glob.glob(osp.join(dimgs_dir, '*.png')))
    imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.png')))
    for ithresh in np.linspace(0.1, 0.6, 6):
        for dimp, imp in tqdm(zip(dimgs_pths, imgs_pths)):
            # get mask
            dimg = plt.imread(dimp)
            mask = (dimg > ithresh).astype(np.uint8)
            mask = np.expand_dims(mask, axis=2)
            save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}')
            fname_mask = osp.join(save_dir, osp.basename(dimp))
            os.makedirs(osp.dirname(fname_mask), exist_ok=True)
            mask_3ch = np.repeat((mask * 255).astype(np.uint8), repeats=3, axis=2)
            cv2.imwrite(fname_mask, mask_3ch)

            # get masked image
            img = plt.imread(imp)
            mskd_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}_mskd_img')
            fname_mskd_img = osp.join(save_dir, osp.basename(imp))
            os.makedirs(osp.dirname(fname_mskd_img), exist_ok=True)
            cv2.imwrite(fname_mskd_img, (mskd_img[:, :, ::-1]*255).astype(np.uint8))
            # bb()
