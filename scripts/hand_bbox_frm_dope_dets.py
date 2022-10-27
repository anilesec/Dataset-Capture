import os, sys, glob
from webbrowser import get
from evaluation.viz_utils import saveas_json
import numpy as np
import pathlib
osp = os.path
from evaluation.eval_utils import *
import cv2
from tqdm import tqdm
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

def get_bbox_from_jts2d(jts2d, xoffset=75, yoffset=150):
    mins = jts2d.min(axis=0)
    maxs = jts2d.max(axis=0)
    xmin, ymin = mins[0], mins[1]
    xmax, ymax = maxs[0], maxs[1]
    w = min(xmax - xmin, 1280)
    h = min(ymax - ymin, 720)
    bbox = [max(xmin - xoffset, 0), max(ymin - xoffset, 0), w + yoffset, h + yoffset]

    return bbox


if __name__ == "__main__":
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
        
        all_imgs_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'rgb/*.png')))
        # bb()
        assert len(all_imgs_pths) == len(all_jts2d_ann), f"no of 2d jts{len(all_jts2d_ann)} and imgs {len(all_imgs_pths)} are not same"

        for imgp, j2d in tqdm(zip(all_imgs_pths, all_jts2d_ann)):
            
            hbbox = get_bbox_from_jts2d(j2d)
    
            img = cv2.imread(imgp)
            bbox_vis = np.array(hbbox, int)
            bbox_vis[2:] += bbox_vis[:2]
            cvimg = cv2.rectangle(img, tuple(bbox_vis[:2]), tuple(bbox_vis[2:]), (255, 0, 0), 3)
    
            # save img with bbox
            img_bbox_savepth = imgp.replace('rgb', 'hand_bbox')
            os.makedirs(osp.dirname(img_bbox_savepth), exist_ok=True)
            cv2.imwrite(img_bbox_savepth, cvimg)
            # bb()
            # save bbox
            bbox_save_pth = img_bbox_savepth.replace('png', 'txt')
            np.savetxt(bbox_save_pth, np.array(hbbox))
            
    print('Done')













        

            
            


    

