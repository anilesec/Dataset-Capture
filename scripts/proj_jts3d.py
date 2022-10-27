# import open3d as o3d
import numpy as np
import os, sys, glob
from ipdb import set_trace as bb
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image
from evaluation.viz_utils import *

osp = os.path
# read_o3d_pcd = o3d.io.read_point_cloud

intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
            )
RES_DIR = '/scratch/1/user/aswamy/data/hand-obj/'    

def tform_points(T, X):
    """
    X: Nx3
    T: 4x4 homogeneous
    """
    X = np.vstack((X.T, np.ones(len(X))))
    X = T @ X
    X = X[:3].T
    return X

def project(P, X):
    """
    X: Nx3
    P: 3x4 projection matrix, ContactPose.P or K @ cTo
    returns Nx2 perspective projections
    """
    X = np.vstack((X.T, np.ones(len(X))))
    x = P @ X
    x = x[:2] / x[2]
    return x.T

def project_ontoimg(img, pts, pose, intrinsics, disp=False, fn=None):
    import matplotlib.pyplot as plt    
    proj_mat = intrinsics @ pose[:3, :]
    img_pts = project(P=proj_mat, X=pts)
    if disp:
        import matplotlib.pyplot as plt    
        plt.imshow(img)
        plt.scatter(img_pts[:, 0], img_pts[:, 1], s=1, c='r', alpha=0.02)
        plt.savefig(fn)
        plt.show()

    return img_pts

def plot_onto_img(img, pts):
    proj_img = cv2.circle(img, pts[0], 0, [0, 0, 255], -1)
    return proj_img


def proj_jts3d_onto_frms(jts3d, all_frms_poses_pths, imgs_dir, save_dir, intrinsics):
    for idx, posp in tqdm(enumerate((all_frms_poses_pths))):
        frm_no = osp.basename(osp.dirname(posp))[-4:]
        imgp = osp.join(imgs_dir, f'{int(frm_no):010d}.png')  # try to have frame id between 0 to 999
        # bb()
        if not Path(imgp).is_file():
            print(f"Warning: For pose file {posp} corresponding image path {imgp} does not exists!! No Projection!!")
            continue
        img = cv2.imread(imgp)
        
        pose = np.linalg.inv(np.loadtxt(posp)) # inv because we need tgt to src for proj

        jts2d = project(P=(intrinsics @ pose[:3, :]), X=jts3d)

        # bb()
        img_jts2d = draw_projtd_handjts(img, jts2d.reshape(1, 21, 2), 'OURS', line_type=cv2.LINE_AA, colors=((0, 0, 255), (0, 0, 255)))

        os.makedirs(save_dir, exist_ok=True)
        fn_img_jts2d = osp.join(save_dir, osp.basename(imgp))
        cv2.imwrite(fn_img_jts2d, img_jts2d)

        fn_jts2d = osp.join(save_dir, osp.basename(imgp).replace('png', 'txt'))

        np.savetxt(fn_jts2d, jts2d)

    print(f'Saved here: {save_dir}')

    return None


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser("project each frame 3d jts to all frames of a given seq")
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
    # bb()
    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        
        all_sqns = [args.sqn]

    for sqn in all_sqns:
        print(f'sqn:{sqn}')
        jts3d = np.loadtxt(f'{RES_DIR}/{sqn}/jts3d_disp.txt')

        all_frms_poses_pths = sorted(glob.glob(f'{RES_DIR}/{sqn}/icp_res/*/f_trans.txt'))

        imgs_dir = f'{RES_DIR}/{sqn}/rgb'
        assert Path(imgs_dir).is_dir(), f"rgb dir does not exist {imgs_dir}" 

        save_dir = f'{RES_DIR}/{sqn}/proj_jts_disp'

        out = proj_jts3d_onto_frms(jts3d, all_frms_poses_pths, imgs_dir, save_dir, intrinsics)
    
        print('Done!')



if __name__ == "__main__" and False:
    import argparse
    import pathlib

    parser = argparse.ArgumentParser("project GT jts3d to all frames of a given seq")
    parser.add_argument('--sqn', type=str, default=None,
                        help='seq name')
    parser.add_argument('--sqn_res_dir', type=str, default=None,
                        help='seq results dir')
    args = parser.parse_args()
    print("args:", args)
    
    if args.sqn_res_dir is not None:
        args.sqn = osp.basename(pathlib.Path(args.sqn_res_dir))

    ICP_DIR = '/scratch/1/user/aswamy/github_repos/Fast-Robust-ICP'
    RES_DIR = '/scratch/1/user/aswamy/data/hand-obj/'

    # bb()
    jts3d = np.loadtxt(f'{RES_DIR}/{args.sqn}/jts3d.txt')
    
    all_frms_poses_pths = sorted(glob.glob(f'{RES_DIR}/{args.sqn}/icp_res/*/f_trans.txt'))
    # bb()
    
    imgs_dir = f'{RES_DIR}/{args.sqn}/rgb'
    assert Path(imgs_dir).is_dir(), f"rgb dir does not exist {imgs_dir}" 

    save_dir = f'{RES_DIR}/{args.sqn}/proj_jts'

    out = proj_jts3d_onto_frms(jts3d, all_frms_poses_pths, imgs_dir, save_dir, intrinsics)
   
    print('Done!')