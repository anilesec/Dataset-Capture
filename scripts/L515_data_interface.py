import numpy as  np
import glob, copy
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
import time
import os
osp = os.path
from scripts.io3d import *
# import open3d as o3d

class L515DataInterface:
    def __init__(self, inp_seq_dir, save_base_dir, save_rgb=False, save_depth=False,
     save_mask=False, save_pcd=False, pcd_bkgd_rm=False, pcd_with_color=False,
     pcd_with_normals=False, dist_thresh=0.8, crop_arm=False, depth_type='dist_xyz'):
        self.inp_seq_dir = inp_seq_dir
        self.save_base_dir = save_base_dir
        self.save_rgb = save_rgb
        self.save_depth = save_depth
        self.save_mask = save_mask
        self.save_pcd  = save_pcd
        self.pcd_bkgd_rm = pcd_bkgd_rm
        self.depth_type = depth_type
        self.pcd_with_color = pcd_with_color
        self.pcd_with_normals = pcd_with_normals
        self.crop_arm = crop_arm
        self.dist_thresh = dist_thresh

    def save_rgbs(self,):
        print("Saving rgbs...")
        start = time.time()
        imgs_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.bgr.npz')))
        assert len(imgs_pths) > 0, f"No .bgr.npz files in given seq dir:{self.inp_seq_dir}"

        rgb_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'rgb')
        os.makedirs(rgb_sdir, exist_ok=True)

        for (imp) in tqdm(imgs_pths):
            bgr = dict(np.load(imp))['arr_0']    
            fn_rgb = osp.join(rgb_sdir, osp.basename(imp).replace('bgr.npz', 'png'))
            cv2.imwrite(fn_rgb, bgr)
        print('Done!')
        print(f'RGB Save Time: {(time.time() - start):0.4f}s')
        print(f"rgb save here: {rgb_sdir}")
        
        return None
    
    def save_pointclouds(self,):
        "save poinclouds"
        print('Saving PCDs...')
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

            # if self.pcd_bkgd_rm:
            #     sel_inds, _ = self._pcd_bkgd_rm(pts=xyz_resh, dist_thresh=self.dist_thresh)
            #     xyz_sel = xyz_resh[sel_inds]
            #     clr_sel = xyz_color_resh[sel_inds]
            # else:
            #     xyz_sel = xyz_resh
            #     clr_sel = xyz_color_resh

            if self.pcd_bkgd_rm:
                sel_inds_frgnd, _ = self._pcd_bkgd_rm(pts=xyz_resh, dist_thresh=self.dist_thresh)
            else:
                sel_inds_frgnd = np.ones(xyz_resh.shape[0])
            
            if self.crop_arm:
                sel_inds_armless = self.get_armless_inds(img_pth=imp)
            else:
                sel_inds_armless = np.ones(xyz_resh.shape[0])

            xyz_sel = xyz_resh[sel_inds_frgnd & sel_inds_armless]
            clr_sel = xyz_color_resh[sel_inds_frgnd & sel_inds_armless]
            
            if self.pcd_with_color and self.pcd_with_normals:
                if self.pcd_bkgd_rm:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'fgnd_xyz_wc_wn')
                else:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wc_wn')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                # compute normals
                xyz_nrmls = self._compute_normals(pts=xyz, rad=0.05, maxnn=10)
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=clr_sel, normals=xyz_nrmls, binary=False)
            elif self.pcd_with_normals:
                if self.pcd_bkgd_rm:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'fgnd_xyz_wn')
                else:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wn')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                # compute normals
                xyz_nrmls = self._compute_normals(pts=xyz, rad=0.05, maxnn=10)
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=None, normals=xyz_nrmls, binary=False)
            elif self.pcd_with_color:
                if self.pcd_bkgd_rm:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'fgnd_xyz_wc')
                else:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz_wc')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=clr_sel, normals=None, binary=False)
            else:
                if self.pcd_bkgd_rm:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'fgnd_xyz')
                else:
                    pcd_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'xyz')
                os.makedirs(pcd_sdir, exist_ok=True)
                fn_pcd = osp.join(pcd_sdir, osp.basename(imp).replace('bgr.npz', 'ply'))
                write_ply(fn_pcd, verts=xyz_sel, trias=None, color=None, normals=None, binary=False)

        print('Done!')
        print(f'PCD Save Time: {(time.time() - start):0.4f}s')
        print(f"PCD saved here: {pcd_sdir}")
        
        return None

    def save_depth_maps(self,):
        "save depth values of each pixels"
        raise NotImplementedError("Depth maps should be computed!")

        return None

    def save_masks(self,):
        "save foreground masks"
        print('Saving masks...')
        start = time.time() 

        mask_sdir = osp.join(self.save_base_dir, self._get_seq_name, 'mask')
        os.makedirs(mask_sdir, exist_ok=True)

        pcds_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.xyz.npz')))
        assert len(pcds_pths) > 0, f"No .xyz.npz files in given seq dir:{self.inp_seq_dir}"

        for pcdp in tqdm(pcds_pths):
            xyz_raw = dict(np.load(pcdp))['arr_0']

            # replace nan to 0
            if np.isnan(xyz_raw).any():
                xyz = np.nan_to_num(xyz_raw) 
            else:
                xyz = xyz_raw   
            
            # replace 0 dist with max dist values
            xyz_dist = np.linalg.norm(xyz, axis=2)
            xyz_dist = np.where(xyz_dist==0., xyz_dist.max(), xyz_dist)

            sel_inds =  xyz_dist < self.dist_thresh
            mask = (sel_inds.astype(np.uint8) * 255).reshape(*sel_inds.shape, 1)
            mask_3ch = np.repeat(mask, repeats=3, axis=2)

            fn_mask = osp.join(mask_sdir, osp.basename(pcdp).replace('xyz.npz', 'png'))
            cv2.imwrite(fn_mask, mask_3ch)

        print('Done!')
        print(f'RGB Save Time: {(time.time() - start):0.4f}s')
        print(f"rgb save here: {mask_sdir}")

        return None

    @property
    def _get_seq_name(self,):
        """
        seq_dir: "../../../seq_name" 
        """
        seq_name = self.inp_seq_dir.rstrip('/').split('/')[-1]

        return seq_name
    
    def _pcd_bkgd_rm(self, pts, dist_thresh):
        "dist_thresh to be set manually for each seq"
        pts = pts.reshape(-1, 3)
        dist = np.linalg.norm(pts, axis=1)
        sel_inds = dist < dist_thresh
        pts_sel = pts[sel_inds]
        
        return sel_inds, pts_sel
    
    def get_armless_inds(self, img_pth):
        arm_mask_pth = img_pth.replace('pointcloud', 'armless_msk').replace('bgr.npz', 'png')
        arm_mask = cv2.imread(arm_mask_pth)

        return arm_mask[:, :, 0].flatten().astype(bool)

    def _compute_normal1s(self, pts, rad=0.05, maxnn=10):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.pybind.utility.Vector3dVector(pts)
        pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=maxnn))

        return np.array(pcd_o3d.normals)
    
    def __call__(self):
        if self.save_rgb:
            self.save_rgbs()
        if self.save_pcd:    
            self.save_pointclouds()
        if self.save_depth:
            self.save_depth_maps()
        if self.save_mask:
            self.save_masks()

        return None

if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser("Processing captured raw data")

    # parser.add_argument('--inp_seq_dir', type=str, default='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547/pointcloud',
    #                     help='path to the input data dir(dir with *.bgr.npz and *.xyz.npz')
    # parser.add_argument('--save_base_dir', type=str, default='/tmp-network/dataset/hand-obj',
    #                     help='base path to save processed data')
    # parser.add_argument('--save_rgb', action='store_true',
    #                     help='flag to save rgb image')    
    # parser.add_argument('--save_depth', action='store_true',
    #                     help='flag to save depth')    
    # parser.add_argument('--save_pcd', action='store_true',
    #                     help='flag to save only xyz')
    # parser.add_argument('--save_mask', action='store_true',
    #                     help='flag to save foreground mask, need save_pcd and seg_pcd flags to be set')                        
    # parser.add_argument('--pcd_normals', action='store_true',
    #                     help='flag to compute pcd normals')  
    # parser.add_argument('--pcd_color', action='store_true',
    #                     help='flag to save pcd color')                        
    # parser.add_argument('--pcd_bkgd_rm', action='store_true',
    #                     help='flag to segment foreground poincloud, needs save_pcd flag to be set')  
    # parser.add_argument('--depth_thresh', type=float, required=True,
    #                     help='depth/dist threshold in (meters) to segment the foreground(hand+obj)')
    # parser.add_argument('--depth_type', type=str, default='dist_xyz',
    #                     help='depth value choice; "dist_xyz":distance of pcd or "zaxis_val":z-axis val of pcd') 

    # args = parser.parse_args()

    # print("args:", args)     

    # create class instance
    # interface = L515DataInterface(
    #     inp_seq_dir=args.inp_seq_dir,
    #     save_base_dir=args.save_base_dir,
    #     save_rgb=args.save_rgb,
    #     save_depth=args.save_depth,
    #     save_mask=args.save_mask,
    #     save_pcd=args.save_pcd,
    #     pcd_bkgd_rm=args.pcd_bkgd_rm,
    #     depth_type=args.depth_type,
    #     pcd_with_color=args.pcd_color,
    #     pcd_with_normals=args.pcd_normals,
    #     dist_thresh=args.depth_thresh
    # )
    # # run class instance
    # interface()
    

    # or run below lines for test
    interface = L515DataInterface(inp_seq_dir='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547',
    save_base_dir='/tmp-network/user/aswamy/dataset/hand-obj', save_rgb=False, save_depth=False,
     save_mask=False, save_pcd=True, pcd_bkgd_rm=True, pcd_with_color=True,
     pcd_with_normals=False, dist_thresh=0.8, depth_type='dist_xyz')
    # interface.save_rgbs() # for rgb
    # interface.save_pointclouds() # for pcd
    # interface() # for both rgb and pcds
    interface.save_pointclouds()
    print('Done!!')