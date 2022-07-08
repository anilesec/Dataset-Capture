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
from sleeve_segm import getLargestCC, segm_sleeve

def imgs2vid_ffmpeg(imgs_dir, file_pth, ext='png',  frm_rate=10):
    import os
    print(f"ffmpeg creating video...")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate {frm_rate} -pattern_type glob -i '{imgs_dir}/*.{ext}' -c:v " f"libx264 -vf fps=30 -pix_fmt yuv420p {file_pth} -y "
    os.system(cmd)
   # os.system(f"rm {imgs_dir}/*.jpg")s
    return print(f"video saved here: {file_pth}")

class L515DataInterface:
    def __init__(self, inp_seq_dir, save_base_dir, save_rgb=False, save_depth=False,
     save_mask=False, save_pcd=False, pcd_bkgd_rm=False, pcd_with_color=False,
     pcd_with_normals=False, dist_thresh=0.8, crop_slv=False, depth_type='dist_xyz',
     start_ind=None, end_ind=None, slv_clr=None, norm_type=None, kms_max_iter=None,
     kms_eps=None, kms_num_clstrs=None):
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
        self.crop_slv = crop_slv
        self.dist_thresh = dist_thresh
        self.start_ind = start_ind
        self.end_ind = end_ind

        # sleeve segm parameters
        if slv_clr is not None:
            self.slv_clr = slv_clr
        else:
            raise ValueError("class arg slv_color is None!!")

        if norm_type is not None:
            self.norm_type = norm_type
        else:
            ValueError("class arg norm_type is None!!")
        
        if kms_max_iter is not None:
            self.kms_max_iter = kms_max_iter 
        else:
            raise ValueError("class arg kms_max_iter is None!!")         
        
        if kms_eps is not None:
            self.kms_eps = kms_eps
        else:
            raise ValueError("class arg kms_eps is None!!") 
         
        if kms_num_clstrs is not None:
            self.kms_num_clstrs = kms_num_clstrs 
        else:
            raise ValueError("class arg kms_num_clstrs is None!!")  

    def save_rgbs(self,):
        print("Saving rgbs...")
        start = time.time()
        imgs_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.bgr.npz')))[self.start_ind : self.end_ind]
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

        pcds_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.xyz.npz')))[self.start_ind : self.end_ind]
        assert len(pcds_pths) > 0, f"No .xyz.npz files in given seq dir:{self.inp_seq_dir}"

        imgs_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.bgr.npz')))[self.start_ind : self.end_ind]
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
                sel_inds_frgnd = np.ones(xyz_resh.shape[0]).astype(bool)
            
            if self.crop_slv:
                sel_inds_slvless = self.get_slvless_inds(img_pth=imp)
            else:
                sel_inds_slvless = np.ones(xyz_resh.shape[0]).astype(bool)

            xyz_sel = xyz_resh[sel_inds_frgnd & sel_inds_slvless]
            clr_sel = xyz_color_resh[sel_inds_frgnd & sel_inds_slvless]
            
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

        pcds_pths = sorted(glob.glob(osp.join(self.inp_seq_dir, 'pointcloud', '*.xyz.npz')))[self.start_ind : self.end_ind]
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
    
    def save_slvless_masks(self,):
        "segment sleeve/arm and save sleevless/armless masks"
        print('Segmenting and Saving sleevless/armless masks...')
        start = time.time()
        rgbs_dir = osp.join(self.save_base_dir, self._get_seq_name, 'rgb')
        save_seq_dir = osp.join(self.save_base_dir, self._get_seq_name)
        slvless_msk_dir = segm_sleeve(rgbs_dir,  save_seq_dir, self.slv_clr, self.start_ind, self.end_ind,
                    self.norm_type, self.kms_max_iter, self.kms_eps, self.kms_num_clstrs)
        print('Done!')
        print(f'RGB Save Time: {(time.time() - start):0.4f}s')
        print(f"rgb save here: {slvless_msk_dir}")

        return None
    
    def create_rgb_vid(self):
        print('creating rgb video...')
        rgb_imgs_dir = osp.join(self.save_base_dir, self._get_seq_name, 'rgb')
        rgb_vid_fn = osp.join(rgb_imgs_dir, 'vid_rgb.mp4')
        imgs2vid_ffmpeg(rgb_imgs_dir, rgb_vid_fn, ext='png', frm_rate=10)
        print(f"rbg vid saved here: {rgb_vid_fn}")

        return None

    def create_slvless_img_vid(self):
        print('creating slvless img video...')
        slvlss_imgs_dir = osp.join(self.save_base_dir, self._get_seq_name, 'slvless_img')
        slvlss_imgs_vid_fn = osp.join(slvlss_imgs_dir, 'vid_slvless.mp4')
        imgs2vid_ffmpeg(slvlss_imgs_dir, slvlss_imgs_vid_fn, ext='png', frm_rate=10)
        print(f"rbg vid saved here: {slvlss_imgs_vid_fn}")

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

    def get_slvless_inds(self, img_pth):
        slvless_msk_pth = osp.join(self.save_base_dir, self._get_seq_name, 'slvless_msk',
         osp.basename(img_pth).replace('bgr.npz', 'png'))
        # armless_msk_pth = img_pth.replace('pointcloud', 'armless_msk').replace('bgr.npz', 'png')
        slvless_msk = cv2.imread(slvless_msk_pth)

        return slvless_msk[:, :, 0].flatten().astype(bool)

    def _compute_normal1s(self, pts, rad=0.05, maxnn=10):
        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.pybind.utility.Vector3dVector(pts)
        pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=maxnn))

        return np.array(pcd_o3d.normals)
    
    def write_seq_info(self):
        "write the config file"
        pass

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

    parser = argparse.ArgumentParser("Processing captured raw data")

    parser.add_argument('--inp_seq_dir', type=str, default='/tmp-network/user/aswamy/L515_seqs/20220614/20220614171547',
                        help='path to the input data dir(dir with *.bgr.npz and *.xyz.npz')
    parser.add_argument('--save_base_dir', type=str, default='/gfs-ssd/dataset/hand-obj',
                        help='base path to save processed data')
    parser.add_argument('--start_ind', type=int, default=None,
                        help='start index of a seq')
    parser.add_argument('--end_ind', type=int, default=None,
                        help='End index of a seq')

    # segm slv args
    parser.add_argument('--norm_type', type=str, default='L2', choices=['L2', 'ratio'],
                        help='norm choice; "L2" or "ratio"')
    parser.add_argument('--kms_max_iter', type=int, default=100,
                        help='k-means max iteration criteria (increases computation time)') 
    parser.add_argument('--kms_eps', type=float, default=0.2,
                        help='k-means accuracy criteria')                        
    parser.add_argument('--kms_num_clstrs', type=int, default=10,
                        help='k-means number of clusters')
    args = parser.parse_args()

    print("args:", args)     

    # sleeve segm parameters
    SLV_COLOR = [128, 204, 77] # normalization color of arm sleeve(observe the segmented image after clustering and then set this)
    KMEANS_MAX_ITER = 100  # k-means max iteration criteria (increases computation time)
    KMEANS_EPSILON = 0.2 # k-means accuracy criteria
    KMEANS_NUM_CLUSTERS = 10 # k-means number of clusters
    
    #run below lines for test
    interface = L515DataInterface(inp_seq_dir=args.inp_seq_dir, save_base_dir=args.save_base_dir,
    save_rgb=True, save_depth=False, save_mask=True, save_pcd=True, pcd_bkgd_rm=True,
    pcd_with_color=True, pcd_with_normals=False, dist_thresh=0.8, crop_slv=True,
    depth_type='dist_xyz', start_ind=args.start_ind, end_ind=args.end_ind, slv_clr=SLV_COLOR,
    norm_type=args.norm_type, kms_max_iter=args.kms_max_iter, kms_eps=args.kms_eps,
    kms_num_clstrs=args.kms_num_clstrs)
    bb()
    interface.save_rgbs()
    interface.save_slvless_masks() 
    interface.save_pointclouds()
    bb()
    print('Done!!')