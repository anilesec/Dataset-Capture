from contextlib import AsyncExitStack
from dis import dis
import numpy as  np
import glob, copy
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
import time
import os
osp = os.path
from io3d import *
# import open3d as o3d

class L515DataInterface:
    def __init__(self, inp_seq_dir, save_base_dir, save_rgb=False, save_depth=False,
     save_mask=False, save_pcd=False, pcd_bkgd_rm=False, pcd_with_color=False,
     pcd_with_normals=False, dist_thresh=0.8, depth_type='dist_xyz'):
        self.inp_seq_dir = inp_seq_dir
        self.save_base_dir = save_base_dir
        self.save_rgb = save_rgb
        self.save_depth = save_depth,
        self.save_mask = save_mask,
        self.save_pcd  = save_pcd,
        self.pcd_bkgd_rm = pcd_bkgd_rm,
        self.depth_type = depth_type,
        self.pcd_with_color = pcd_with_color,
        self.pcd_with_normals = pcd_with_normals,
        self.dist_thresh = dist_thresh,


    def save_rgbs(self,):
        start = time.time()
        imgs_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.bgr.npz')))
        assert len(imgs_pths) > 0, f"No .bgr.npz files in given seq dir:{self.inp_seq_dir}"

        rgb_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'rgb')
        os.makedirs(rgb_sdir, exist_ok=True)

        for (imp) in tqdm(imgs_pths):
            bgr = dict(np.load(imp))['arr_0']    
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)
        print(f'RGB Save Time: {(time.time() - start):0.4f}s')
        print(f"rgb save here: {rgb_sdir}")

        return None
    
    def save_pointclouds(self,):
        "save poinclouds"
        start = time.time()
        pcds_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.xyz.npz')))
        assert len(pcds_pths) > 0, f"No .xyz.npz files in given seq dir:{self.inp_seq_dir}"

        imgs_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.bgr.npz')))
        assert len(imgs_pths) > 0, f"No .bgr.npz files in given seq dir:{self.inp_seq_dir}"

        for (pcdp, imp) in tqdm(zip(pcds_pths, imgs_pths)):
            xyz_raw = dict(np.load(pcdp))['arr_0']
            bgr = dict(np.load(imp))['arr_0']
            
            # replace nan to 0
            if np.isnan(xyz_raw).any():
                xyz = np.nan_to_num(xyz_raw) 
            else:
                xyz = xyz_raw
            
            xyz_resh = xyz.reshape(-1, 3)
            xyz_color_resh = bgr[:, :, ::-1].reshape(-1, 3) 

            if self.pcd_bkgd_rm:
                sel_inds, _ = self._pcd_bkgd_rm(pts=xyz_resh)
                xyz_sel = xyz_resh[sel_inds]
                clr_sel = xyz_color_resh[sel_inds]
            else:
                xyz_sel = xyz_resh
                clr_sel = xyz_color_resh

            if self.pcd_with_color:
                pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wc')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=clr_sel, normals=None, binary=False)
            elif self.pcd_with_normals:
                pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wn')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                # compute normals
                xyz_nrmls = self._compute_normals(pts=xyz, rad=0.05, maxnn=10)
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=None, normals=xyz_nrmls, binary=False)
            elif self.pcd_with_color and self.pcd_with_normals:
                pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wc_wn')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                # compute normals
                xyz_nrmls = self._compute_normals(pts=xyz, rad=0.05, maxnn=10)
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=clr_sel, normals=xyz_nrmls, binary=False)
            else:
                pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=clr_sel, normals=None, binary=False)

        print(f'PCD Save Time: {(time.time() - start):0.4f}s')
        print(f"PCD saved here: {pcd_sdir}")
        
        return None

    def save_depth_maps(self,):
        "save depth values of each pixels"
        raise NotImplementedError("Depth maps should be computed!")

    def save_masks(self,):
        "save foreground masks"
        raise NotImplementedError("foregorund masks should be computed!")

    @property
    def _get_seq_name(self,):
        """
        seq_dir: "../../../seq_name" 
        """
        seq_name = self.inp_seq_dir.rstrip('/').split('/')[-1]

        return seq_name
    
    def _pcd_bkgd_rm(self, pts):
        "dist_thresh to be set manually for each seq"
        pts = pts.reshape(-1, 3)
        dist = np.linalg.norm(pts, axis=1)
        sel_inds = dist < self.dist_thresh
        pts_sel = pts[sel_inds]
        
        return sel_inds, pts_sel

    def _compute_normals(self, pts, rad=0.05, maxnn=10):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.pybind.utility.Vector3dVector(pts)
        pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=maxnn))
        return np.array(pcd_o3d.normals)

if __name__ == "__main__":
    interface = L515DataInterface(inp_seq_dir='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547',
    save_base_dir='/tmp-network/user/aswamy/temp/', save_rgb=True, save_depth=False,
     save_mask=False, save_pcd=True, pcd_bkgd_rm=True, pcd_with_color=False,
     pcd_with_normals=False, dist_thresh=0.8, depth_type='dist_xyz')

    # interface.save_rgbs()  # works
    bb()
    interface.save_pointclouds()
    

