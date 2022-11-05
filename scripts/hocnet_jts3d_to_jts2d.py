
import os, glob
from ipdb import set_trace as bb
import numpy as np
from evaluation.viz_utils import *
from evaluation.eval_utils import * 

if __name__ == "__main__":
    print(f"This scripts transforms the annotated 3d joints to all frames using ann poses")
    import argparse
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None, help='seq id')
    
    args = parser.parse_args()
    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    print(f"All seq ids: \n {all_sqns}")

    if args.sqn is not None:
        all_sqns = [args.sqn]

    for sqn in tqdm(all_sqns):
        print(f"sqn: {sqn}")

        all_jts3d_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'jts3d_hocnet/*.txt')))
        
        for jts3dp in tqdm(all_jts3d_pths):
            img_pth = pathlib.Path((jts3dp.replace('jts3d_hocnet', 'rgb')).replace('txt', 'png'))
            jts3d_ann_pth = pathlib.Path(jts3dp.replace('jts3d_hocnet', 'jts3d'))
            if not img_pth.exists():
                print(f'RGB Missing: {img_pth}')
            if not jts3d_ann_pth.exists():
                print(f'Ann Missing: {jts3d_ann_pth}')
            
            jts3d_ann = np.loadtxt(str(jts3d_ann_pth))
            jts3d = np.loadtxt(jts3dp)
            img = cv2.imread(str(img_pth))

            # translate hocnet jts3d by GT jts3d root coord
            jts3d_center = jts3d - jts3d[0] 
            jts3d_tran = jts3d_center + jts3d_ann[0]

            proj_trnsfm = CAM_INTR @ np.eye(4)[:3, :]
            jts2d = project(P=(CAM_INTR @ np.eye(4)[:3, :]), X=jts3d_tran)

            img_jts2d = draw_projtd_handjts(img, jts2d.reshape(1, 21, 2), 'CP', line_type=cv2.LINE_AA, colors=((0, 0, 255), (0, 0, 255)))
            save_dir = osp.join(RES_DIR, sqn, 'jts2d_hocnet')
            os.makedirs(save_dir, exist_ok=True)
            fn_img_jts2d = osp.join(save_dir, osp.basename(img_pth))
            cv2.imwrite(fn_img_jts2d, img_jts2d)

            fn_jts2d = osp.join(save_dir, osp.basename(img_pth).replace('png', 'txt'))

            np.savetxt(fn_jts2d, jts2d)

        print(f'Saved here: {save_dir}')

            
            
            





