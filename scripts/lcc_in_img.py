import numpy as np
import matplotlib.pyplot as plt
import glob, copy
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2

# select the largest companent and writing it back
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument('--sqn', type=str, required=True,
                        help='seq name')

    parser.add_argument('--start_ind', type=int, default=None,
                    help='start index of a seq')
    parser.add_argument('--end_ind', type=int, default=None,
                    help='End index of a seq')

    args = parser.parse_args()

    def getLargestCC(segmentation):
        from skimage.measure import label   
        "get largest connected components"
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

        return largestCC

    imgs_pths = sorted(glob.glob(f'/scratch/1/user/aswamy/data/hand-obj/{args.sqn}/slvless_img/*.png'))

    for imp in tqdm(imgs_pths[args.start_ind : args.end_ind]):
        im = cv2.imread(imp)
        im_norm = 255. * im / (np.sum(im,axis=-1)[:, :, None] + 1e-6)
        im_mask = (im_norm.sum(2) > 0.0).astype(np.uint8)
        # bb()
        lcc = getLargestCC(im_mask)
        im_lcc = cv2.bitwise_and(im, im, mask=lcc.astype(np.uint8))
        cv2.imwrite(imp, im_lcc)
        # bb()