from re import A
import open3d as o3d
import numpy as np
import os, sys, glob
import polyscope as ps
from ipdb import set_trace as bb
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

osp = os.path
read_o3d_pcd = o3d.io.read_point_cloud

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


def proj_gt_ont_frms(tgt_pcd_pth, all_frms_poses_pths, imgs_dir, save_dir, intrinsics):
    assert Path(tgt_pcd_pth).is_file(), f"tgt path does not exist{tgt_pcd_pth}"
    tgt_pcd = read_o3d_pcd(tgt_pcd_pth)

    for idx, posp in tqdm(enumerate((all_frms_poses_pths))):
        frm_no = osp.basename(osp.dirname(posp))[-4:]
        imgp = osp.join(imgs_dir, f'{int(frm_no):010d}.png')  # try to have frame id between 0 to 999
        # bb()
        # if idx < 424:
        #     continue
        if not Path(imgp).is_file():
            print(f"Warning: For pose file {posp} corresponding image path {imgp} does not exists!! No Projection!!")
            continue
        img = plt.imread(imgp)
        
        pose = np.linalg.inv(np.loadtxt(posp)) # inv because we need tgt to src for proj

        tgt_pcd_pts = np.array(tgt_pcd.points)
        tgt_pcd_pts_projtd = project(P=(intrinsics @ pose[:3, :]), X=tgt_pcd_pts)

        # plot
        plt.figure()
        plt.imshow(img)
        plt.scatter(x=tgt_pcd_pts_projtd[:, 0], y=tgt_pcd_pts_projtd[:, 1], c='r', alpha=0.01, s=0.05)
        os.makedirs(save_dir, exist_ok=True)
        fn_plt = osp.join(save_dir, osp.basename(imgp))
        fn_pos = osp.join(save_dir, osp.basename(imgp).replace('png', 'txt'))
        plt.savefig(fn_plt)
        plt.close()
        np.savetxt(fn_pos, tgt_pcd_pts_projtd)
        # bb()
    print(f'Saved here: {save_dir}')

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("project GT pcd to all frames of a given seq")
    parser.add_argument('--sqn', type=str, required=True,
                        help='seq name')

    args = parser.parse_args()
    print("args:", args)

    ICP_DIR = '/home/aswamy/my_drive/github_repos/Fast-Robust-ICP'

    tgt_pcd_pth = f'{ICP_DIR}/data/{args.sqn}/data_for_reg/tgt_pcd.ply'
    all_frms_poses_pths = sorted(glob.glob(f'{ICP_DIR}/res/{args.sqn}/*/f_trans.txt'))
    # bb()
    imgs_dir = f'/home/aswamy/server_mounts/gfs-ssd/user/aswamy/dataset/hand-obj/{args.sqn}/rgb'
    assert Path(imgs_dir).is_dir(), f"rgb dir does not exist {imgs_dir}" 

    # save_dir = f'/home/aswamy/server_mounts/gfs-ssd/user/aswamy/dataset/hand-obj/{args.sqn}/proj'
    save_dir = f'/home/aswamy/my_drive/data/dataset/hand-obj/{args.sqn}/proj'

    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
            )
    out = proj_gt_ont_frms(tgt_pcd_pth, all_frms_poses_pths, imgs_dir, save_dir, intrinsics)
    print('Done!')
