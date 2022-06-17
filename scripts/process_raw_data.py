# python process_raw_data.py --save_rgb --save_depth --save_pcd --depth_thresh 1.0 --depth_type dist_xyz

from signal import default_int_handler
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

from .io3d import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Processing captured raw data")

    parser.add_argument('--inp_dir', type=str, default='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547/pointcloud',
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
    imgs_pths = sorted(glob.glob(osp.join(args.inp_dir, '*.bgr.npz')))
    # pcds_pths = sorted(glob.glob(osp.join(args.inp_dir, '*.xyz.npz')))
    bb()
    for (imp) in tqdm(zip(imgs_pths)):
        if args.save_rgb or args.save_pcd:
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
            rgb_sdir = osp.join(args.save_base_dir, 'rgb')
            os.makedirs(rgb_sdir, exist_ok=True)
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)
        
        # save depth
        if args.save_depth:
            if args.depth_type == 'zaxis_val':
                depth = xyz[:, :, 2]
            elif args.depth_type == 'xyz_dist':
                depth = np.linalg.norm(xyz, axis=2)
            else:
                raise ValueError('Wrong "depth_type" argument')

            depth_sdir = osp.join(args.save_base_dir, 'depth')
            os.makedirs(depth_sdir, exist_ok=True)
            fn_depth = osp.join(depth_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, depth)
        
        # save foreground masks
        if args.save_mask:
            pass

        # save pcds
        if args.save_pcd:
            xyz_resh = xyz.reshape(-1, 3)
            xyz_color_resh = bgr.reshape(-1, 3)
            depth_z_resh = depth.reshape(-1, 3)

            if args.seg_pcd:
                args.depth_thresh = depth.max()

            sel_inds = depth < args.depth_thresh
            xyz_sel = xyz_resh[sel_inds]
            xyz_color_sel = xyz_color_resh[sel_inds]

            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.cpu.pybind.utility.Vector3dVector(xyz_sel)
            pcd_o3d.colors = o3d.cpu.pybind.utility.Vector3dVector(xyz_color_sel)
            pcd_o3d.estimate_normals() # verify this operation
            
            if args.seg_pcd:
                pcd_wn_sdir = osp.join(args.save_base_dir, 'seg_xyz_wn')
            else:
                pcd_wn_sdir = osp.join(args.save_base_dir, 'xyz_wn')

            os.makedirs(pcd_wn_sdir, exist_ok=True)
            fn_pcd_wn = osp.join(pcd_wn_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
            
            write_ply(path=fn_pcd_wn, verts=xyz_sel, trias=None, color=xyz_color_sel,
             normals=np.array(pcd_o3d.normals), binary=False)  
            pcd_wn_wc_sdir = osp.join(args.save_base_dir, 'xyz_wn_wc')
            os.makedirs(pcd_wn_wc_sdir, exist_ok=True)


class ProcessRawCamData:
    def __init__(self, inp_dir, depth_thresh, )




            
                      







        



        



    
