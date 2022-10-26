import os, cv2, glob
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb

RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

def get_bin_mask(im, th=0, val=255):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im_bin = cv2.threshold(im_gray, th, val, cv2.THRESH_BINARY)

    return im_bin

if __name__ == "__main__":
    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    print(f"All seq ids: \n {all_sqns}")

    for sqn in tqdm(all_sqns):
        print(f"seq: {sqn}")

        if sqn == '20220907152036':
            # get hand-obj img from slvless_img 
            slvless_imgs_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'slvless_img/*.png')))

            for slvless_imgp in tqdm(slvless_imgs_pths):
                slvless_img = cv2.imread(slvless_imgp)
                slvless_img_bin = get_bin_mask(im=slvless_img, th=0, val=255)
                
                fn_slvless_img_bin = slvless_imgp.replace('slvless_img', 'slvless_img_bin')
                os.makedirs(osp.dirname(fn_slvless_img_bin), exist_ok=True)
                cv2.imwrite(fn_slvless_img_bin, slvless_img_bin)



        

