from evaluation.eval_utils import RES_DIR
import numpy as np
import matplotlib.pyplot as plt
import glob, copy
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2


def getLargestCC(segmentation):
    from skimage.measure import label   
    "get largest connected components"
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largestCC
    

# select the largest companent and writing it back
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument('--sqn', type=str, default=None,
                        help='seq name')

    parser.add_argument('--start_ind', type=int, default=None,
                    help='start index of a seq')
    parser.add_argument('--end_ind', type=int, default=None,
                    help='End index of a seq')

    args = parser.parse_args()
    RES_DIR = "/scratch/1/user/aswamy/data/hand-obj/"
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    print(f"All seq ids: \n {all_sqns}")

    if args.sqn is not None:
        all_sqns = [args.sqn]

    for sqn in all_sqns[args.start_ind : args.end_ind]:
        imgs_pths = sorted(glob.glob(f'/scratch/1/user/aswamy/data/hand-obj/{sqn}/slvless_img/*.png'))

        for imp in tqdm(imgs_pths):
            im = cv2.imread(imp)
            im_norm = 255. * im / (np.sum(im,axis=-1)[:, :, None] + 1e-6)
            im_mask = (im_norm.sum(2) > 0.0).astype(np.uint8)
            # bb()
            lcc = getLargestCC(im_mask)
            im_lcc = cv2.bitwise_and(im, im, mask=lcc.astype(np.uint8))
            cv2.imwrite(imp, im_lcc)
        # bb()