import os
import numpy as np 
# import cv2 
import glob
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sqn', type=str, default=None, help='')
    parser.add_argument('--sqn_dir', type=str, default=None, help='')
    args = parser.parse_args()

    base_dir = "/morpheo-nas/aerappan/briac_baseline"
    
    # all_sqns = os.listdir(base_dir)
    
    imgs_pths = sorted(glob.glob(os.path.join(args.sqn_dir, 'rgb/*.png')))
    masks_pths = sorted(glob.glob(os.path.join(args.sqn_dir, 'slvless_img_bin/*.png')))


    print(masks_pths)

    assert len(imgs_pths) == len(masks_pths)

    for idx, imgp in tqdm(enumerate(imgs_pths)):
        new_name = "{:010d}.png".format(idx)
        # print(os.path.join(os.path.dirname(imgp),new_name))
        cmd = 'mv -v {} {}'.format(imgp, os.path.join(os.path.dirname(imgp),new_name))
        os.system(cmd)

    for idx, mgp in tqdm(enumerate(masks_pths)):
        new_name = "{:010d}.png".format(idx)
        # print(os.path.join(os.path.dirname(imgp),new_name))
        cmd = 'mv -v {} {}'.format(mgp, os.path.join(os.path.dirname(mgp),new_name))
        os.system(cmd)
    print('Done')   

