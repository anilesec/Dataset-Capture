import os, cv2, glob
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb
from PIL import Image
import PIL
import numpy as np

def get_bin_mask(im, th=0, val=255):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_bin = cv2.threshold(im_gray, th, val, cv2.THRESH_BINARY)

    return im_bin

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--imgs_dir', type=str, default=None, help='imgs dir')
    args = parser.parse_args()

    imgs_pths = sorted(glob.glob(osp.join(args.imgs_dir, '*.png')))

    for imgp in tqdm(imgs_pths):
        img = cv2.imread(imgp)
        img_bin = get_bin_mask(im=img, th=0, val=255)
        result = Image.fromarray((img_bin).astype(np.uint8))
        result.save(imgp)
        # bb()
