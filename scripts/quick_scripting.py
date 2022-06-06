from copy import deepcopy
from curses import raw
from email.mime import base
from xml.etree.ElementTree import TreeBuilder
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
import open3d as o3d


def imgs2vid_ffmpeg(imgs_dir, file_pth, ext='png',  frm_rate=10):
    print(f"ffmpeg creating video...")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate {frm_rate} -pattern_type glob -i '{imgs_dir}/*.{ext}' -c:v " f"libx264 -vf fps=30 -pix_fmt yuv420p {file_pth} -y "
    os.system(cmd)
   # os.system(f"rm {imgs_dir}/*.jpg")
    return print(f"video saved here: {file_pth}")

# Convert Midas depth into masks 
if False:
    base_dir = '/Users/aswamy/My_drive/github_repos/MiDaS/output/'
    imgs_dir = '/Users/aswamy/My_drive/github_repos/MiDaS/output/dpt_large'
    dimgs_dir = '/Users/aswamy/My_drive/github_repos/MiDaS/output/dpt_large'
    dimgs_pths = sorted(glob.glob(osp.join(dimgs_dir, '*.png')))
    imgs_pths = sorted(glob.glob(osp.join(dimgs_dir, '*.jpeg')))
    for ithresh in np.linspace(0.1, 0.6, 6):
        for dimp, imp in tqdm(zip(dimgs_pths, imgs_pths)):
            # get mask
            dimg = plt.imread(dimp)
            mask = (dimg > ithresh).astype(np.uint8)
            save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}')
            fname_mask = osp.join(save_dir, osp.basename(dimp))
            os.makedirs(osp.dirname(fname_mask), exist_ok=True)
            plt.imsave(fname_mask, mask)

            # get masked image
            img = plt.imread(imp)
            mskd_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}_mskd_img')
            fname_mskd_img = osp.join(save_dir, osp.basename(imp))
            os.makedirs(osp.dirname(fname_mskd_img), exist_ok=True)
            plt.imsave(fname_mskd_img, mskd_img)

# Convert RGBD data to colored pcds(.ply) 
if False:
    # get colored pointclouds from RGB and depth info
    base_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud'
    imgs_dir =  '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud'
    pcds_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud'

    imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.bgr.npz')))
    pcds_pths = sorted(glob.glob(osp.join(pcds_dir, '*.xyz.npz')))

    for (imp, pcdp) in tqdm(zip(imgs_pths, pcds_pths)):
        im = dict(np.load(imp))['arr_0']
        pcd = dict(np.load(pcdp))['arr_0']
        
        # replace nan values with max depth values
        points = np.nan_to_num(pcd)
        x_max = points[:, :, 0].max()
        y_max = points[:, :, 1].max()
        z_max = points[:, :, 2].max()

        points[:, :, 0] = np.nan_to_num(pcd[:, :, 0], nan=x_max)
        points[:, :, 1] = np.nan_to_num(pcd[:, :, 1], nan=y_max)
        points[:, :, 2] = np.nan_to_num(pcd[:, :, 2], nan=z_max)

        # create open3d point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.cpu.pybind.utility.Vector3dVector(points.reshape(-1, 3))
        pcd_o3d.colors = o3d.cpu.pybind.utility.Vector3dVector(im[:, :, ::-1].reshape(-1, 3) / 255.)
        pcd_o3d.scale(1000, center=pcd_o3d.get_center())
        fname = osp.join(base_dir, osp.basename(imp).replace('bgr.npz', 'ply'))
        os.makedirs(osp.dirname(fname), exist_ok=True)
        o3d.io.write_point_cloud(fname, pcd_o3d)
    print('Done!!')

# Segmenting hand pcds from background pcds
if True:
    # get colored pointclouds from RGB and depth info
    base_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud_seg'
    imgs_dir =  '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud'
    pcds_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220524/20220524_155142/pointcloud'

    imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.bgr.npz')))
    pcds_pths = sorted(glob.glob(osp.join(pcds_dir, '*.xyz.npz')))

    for (imp, pcdp) in tqdm(zip(imgs_pths, pcds_pths)):
        im = dict(np.load(imp))['arr_0']
        raw_pts = dict(np.load(pcdp))['arr_0']

        pts = raw_pts.reshape(-1, 3)
        pts_colors = im[:, :, ::-1].reshape(-1, 3) / 255.
        pts_depth = np.linalg.norm(raw_pts.reshape(-1, 3), axis=1)
        tmp_depth = np.nan_to_num(pts_depth)
        depth_max = tmp_depth.max()

        # replace all  nan values with max dist
        pts_depth_wo_nan = np.nan_to_num(pts_depth, nan=depth_max)

        DEPTH_THRESH = 1.0 # 1m 
        sel_inds = (pts_depth_wo_nan < DEPTH_THRESH)

        sel_pts = pts[sel_inds]
        sel_pts_colors = pts_colors[sel_inds]

        # skip if selected points are less than 1000 points
        if sum(sel_inds) < 50000:
            print(f"Warning! Frame {osp.basename(pcdp)} has less than 50000 point points. Frame dropped!!")
            continue

        # create open3d point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.cpu.pybind.utility.Vector3dVector(sel_pts)
        pcd_o3d.colors = o3d.cpu.pybind.utility.Vector3dVector(sel_pts_colors)

        # pcd_o3d.scale(1000, center=pcd_o3d.get_center())
        fname = osp.join(base_dir, osp.basename(imp).replace('bgr.npz', 'ply'))
        os.makedirs(osp.dirname(fname), exist_ok=True)
        o3d.io.write_point_cloud(fname, pcd_o3d, write_ascii=True)
    print('Done!!')
