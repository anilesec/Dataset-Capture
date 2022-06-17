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
    seq_dir: "../../../seq_name/pointcloud" 
    """
    seq_name = seq_dir.rstrip('/').split('/')[-2]

    return seq_name


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
            rgb_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'rgb')
            os.makedirs(rgb_sdir, exist_ok=True)
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)

        # save depth
        if args.save_depth:
            if args.depth_type == 'zaxis_val':
                depth = xyz[:, :, 2]
            elif args.depth_type == 'dist_xyz':
                depth = np.linalg.norm(xyz, axis=2)
            else:
                raise ValueError('Wrong "depth_type" argument')

            depth_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'depth')
            os.makedirs(depth_sdir, exist_ok=True)
            fn_depth = osp.join(depth_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            depth_norm = (depth * 255 / (depth.max() - depth.min())).astype(np.uint8)
            cv2.imwrite(fn_depth, depth)

        # save foreground masks
        if args.save_mask:
            pass
        
        # save pcds
        if args.save_pcd:
            xyz_resh = xyz.reshape(-1, 3)
            xyz_color_resh = bgr.reshape(-1, 3)
            depth_resh = depth.flatten()

            if args.seg_pcd:
                args.depth_thresh = depth.max()
            
            sel_inds = depth_resh < args.depth_thresh
            xyz_sel = xyz_resh[sel_inds]
            xyz_color_sel = xyz_color_resh[sel_inds]
            
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.pybind.utility.Vector3dVector(xyz_sel)
            pcd_o3d.colors = o3d.pybind.utility.Vector3dVector(xyz_color_sel)
            pcd_o3d.estimate_normals() # verify this operation
            
            if args.seg_pcd:
                pcd_wn_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'seg_xyz_wn')
                pcd_wn_wc_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'seg_xyz_wn_wc')
            else:
                pcd_wn_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'xyz_wn')
                pcd_wn_wc_sdir = osp.join(args.save_base_dir, _get_seq_name(args.inp_dir), 'xyz_wn_wc')
            
            os.makedirs(pcd_wn_sdir, exist_ok=True)
            fn_pcd_wn = osp.join(pcd_wn_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
            write_ply(fn_pcd_wn, verts=xyz_sel, trias=None, color=None,
             normals=np.array(pcd_o3d.normals), binary=False) 
            
            os.makedirs(pcd_wn_wc_sdir, exist_ok=True)
            fn_pcd_wn_wc = osp.join(pcd_wn_wc_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
            write_ply(fn_pcd_wn_wc, verts=xyz_sel, trias=None, color=xyz_color_sel,
             normals=np.array(pcd_o3d.normals), binary=False)  
    print(f"PCD time: {(time.time() - start):.4f}s")


class ProcessRawCamData:
    def __init__(self, inp_dir, save_base_dir, save_rgb, save_depth, save_mask, save_pcd, seg_pcd, depth_type):
        self.inp_dir = inp_dir
        self.save_base_dir = save_base_dir
        self.save_rgb = save_rgb
        self.save_depth = save_depth,
        self.save_mask = save_mask,
        self.save_pcd  = save_pcd,
        self.seg_pcd = seg_pcd
        self.depth_type = depth_type


    def save_rgbs(self,):
        start = time.time()
        imgs_pths = sorted(glob.glob(osp.join(self.inp_dir, '*.bgr.npz')))
        for (imp) in tqdm(imgs_pths):
            bgr = dict(np.load(imp))['arr_0']
    
            rgb_sdir = osp.join(args.save_base_dir, self._get_seq_name(self.inp_dir), 'rgb')
            os.makedirs(rgb_sdir, exist_ok=True)
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)
        print(f"rgb save here: {rgb_sdir}")

    def save_depth_maps(self,):
        "save depth values of each pixels"
        pass
    
    def save_masks(self,):
        "save foreground masks"
        pass

    def save_pointclouds(self, with_color=True, with_normals=True):
        "save poinclouds"
        pass

    def _get_seq_name(self,):
        """
        seq_dir: "../../../seq_name" 
        """
        seq_name = self.inp_dir.rstrip('/').split('/')[-1]

        return seq_name



    

    



    
