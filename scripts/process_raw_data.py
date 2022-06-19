# python process_raw_data.py --save_rgb --save_depth --save_pcd --depth_thresh 1.0 --depth_type dist_xyz

from http.client import PRECONDITION_REQUIRED
from re import M
from signal import default_int_handler
from ssl import VERIFY_X509_TRUSTED_FIRST
import numpy as np
import matplotlib.pyplot as plt
import glob, copy
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
import open3d as o3d
import time
from io3d import *

def _get_seq_name(seq_dir):
    """
    seq_dir: "../../../seq_name" 
    """
    seq_name = seq_dir.rstrip('/').split('/')[-1]

    return seq_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Processing captured raw data")

    parser.add_argument('--inp_seq_dir', type=str, default='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547/pointcloud',
                        help='path to the input data dir(dir with *.bgr.npz and *.xyz.npz')
    parser.add_argument('--save_base_dir', type=str, default='/tmp-network/dataset/hand-obj',
                        help='base path to save processed data')
    # parser.add_argument('--save_imgs_dir', type=str, default='/tmp-network/dataset/hand-obj/',
    #                     help='dir path to the save extracted/processed frames')
    # parser.add_argument('--save_pcd_dir', type=str, default=,
    #                     help='dir path to the save extracted/processed pointclouds')
    parser.add_argument('--save_rgb', action='store_true',
                        help='flag to save rgb image')    
    parser.add_argument('--save_depth', action='store_true',
                        help='flag to save depth')    
    parser.add_argument('--save_pcd', action='store_true',
                        help='flag to save only xyz')
    parser.add_argument('--save_mask', action='store_true',
                        help='flag to save foreground mask')                        
    parser.add_argument('--seg_pcd', action='store_true',
                        help='flag to segment foreground poincloud')  
    parser.add_argument('--depth_thresh', type=float, default=1.0,
                        help='depth threshold in (meters) to segment the foreground(hand+obj)')
    parser.add_argument('--depth_type', type=str, default='dist_xyz',
                        help='depth value choice; "dist_xyz":distance of pcd or "zaxis_val":z-axis val of pcd') 

    args = parser.parse_args()

    print("args:", args)                        

    # get paths
    imgs_pths = sorted(glob.glob(osp.join(args.inp_seq_dir, 'pointcloud', '*.bgr.npz')))
    assert(len(imgs_pths) > 0, f"No .bgr.npz files in given seq dir:{args.inp_seq_dir}")
    # pcds_pths = sorted(glob.glob(osp.join(args.inp_seq_dir, '*.xyz.npz')))

    start = time.time()
    for (imp) in tqdm(imgs_pths):
        
        if args.save_rgb and args.save_pcd:
            pcdp = imp.replace('bgr', 'xyz')
            assert(osp.isfile(pcdp), f"file {pcdp} does not exist")
        
        bgr = dict(np.load(imp))['arr_0']
        xyz_raw = dict(np.load(pcdp))['arr_0']
        
        # replace nan to 0
        if np.isnan(xyz_raw).any():
            xyz = np.nan_to_num(xyz_raw) 
        else:
            xyz = xyz_raw

        # save rgb
        if args.save_rgb: 
            rgb_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'rgb')
            os.makedirs(rgb_sdir, exist_ok=True)
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)

        # save depth maps(not the depth values) for the depth values compute from xyz
        if args.save_depth:
            if args.depth_type == 'zaxis_val':
                depth = xyz[:, :, 2]
            elif args.depth_type == 'dist_xyz':
                depth = np.linalg.norm(xyz, axis=2)
            else:
                raise ValueError('Wrong "depth_type" argument')

            depth_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'depth')
            os.makedirs(depth_sdir, exist_ok=True)
            fn_depth = osp.join(depth_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            depth_norm = (depth * 255 / (depth.max() - depth.min())).astype(np.uint8)
            cv2.imwrite(fn_depth, depth)
        
        # save pcds
        if args.save_pcd:
            xyz_resh = xyz.reshape(-1, 3)
            xyz_color_resh = bgr.reshape(-1, 3)
            depth_resh = depth.flatten()

            if args.save_mask:
                args.seg_pcd = True

            if args.seg_pcd:
                DEPTH_THRESH = args.depth_thresh
            else:
                DEPTH_THRESH = depth.max()
            
            sel_inds = depth_resh < DEPTH_THRESH
            xyz_sel = xyz_resh[sel_inds]
            xyz_color_sel = xyz_color_resh[sel_inds]

            if args.save_mask:
                mask_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'mask')
                os.makedirs(mask_sdir, exist_ok=True)
                fn_mask = osp.join(mask_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
                mask = sel_inds.reshape(bgr.shape[:2]).astype(np.uint8)*255
                cv2.imwrite(fn_mask, mask)
            
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.pybind.utility.Vector3dVector(xyz_sel)
            pcd_o3d.colors = o3d.pybind.utility.Vector3dVector(xyz_color_sel)
            pcd_o3d.estimate_normals() # verify this operation
            
            if args.seg_pcd:
                pcd_wn_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'seg_xyz_wn')
                pcd_wn_wc_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'seg_xyz_wn_wc')
            else:
                pcd_wn_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'xyz_wn')
                pcd_wn_wc_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_seq_dir), 'xyz_wn_wc')
            
            os.makedirs(pcd_wn_sdir, exist_ok=True)
            fn_pcd_wn = osp.join(pcd_wn_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
            write_ply(fn_pcd_wn, verts=xyz_sel, trias=None, color=None,
             normals=np.array(pcd_o3d.normals), binary=False) 
            
            os.makedirs(pcd_wn_wc_sdir, exist_ok=True)
            fn_pcd_wn_wc = osp.join(pcd_wn_wc_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
            write_ply(fn_pcd_wn_wc, verts=xyz_sel, trias=None, color=xyz_color_sel,
             normals=np.array(pcd_o3d.normals), binary=False)  
    print(f"Time: {(time.time() - start):.4f}s")


    
