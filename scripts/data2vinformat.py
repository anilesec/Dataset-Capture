import os, sys, glob, cv2
from tkinter.messagebox import RETRY
import pickle
import numpy as np
from ipdb import set_trace as bb
from tqdm import tqdm
import roma
from PIL import Image
osp = os.path

def load_pkl(pkl_file):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


class VHull_DataInterface:
    def __init__(self, rgbs_dir, masks_dir, pose_pkl, intrinsics_npy, save_base_dir, frame_start, frame_end):
        self.rgbs_dir = rgbs_dir
        self.masks_dir = masks_dir
        self.pose_pkl = pose_pkl
        self.save_base_dir = save_base_dir
        self.intrinsics_pth = intrinsics_npy
        self.frame_start = frame_start
        self.frame_end = frame_end

        self.intrinsics = np.load(self.intrinsics_npy)

    def write2vinformat(self, intrinsics, images, masks, poses, save_rt_dir):
        for i, (im, msk, pos) in tqdm(enumerate(zip(images, masks, poses))):
            # save image
            img_save_pth = os.path.join(save_rt_dir, 'images', 'cam_' + '{:03d}'.format(i), 'view_0.png')
            if not os.path.exists(os.path.dirname(img_save_pth)):
                os.makedirs(os.path.dirname(img_save_pth), exist_ok=True)
            Image.fromarray(im).save(img_save_pth)

            # save mask
            mask_save_pth = os.path.join(save_rt_dir, 'masks', 'cam_' + '{:03d}'.format(i), 'view_0.png')
            if not os.path.exists(os.path.dirname(mask_save_pth)):
                os.makedirs(os.path.dirname(mask_save_pth), exist_ok=True)
            cv2.imwrite(mask_save_pth, msk)
            # Image.fromarray((msk * 255).astype(np.uint8)).save(mask_save_pth)

            # save proj trnsfm mat (intrinsics @ fiktiv_cam_poses)
            proj_trnsfm = intrinsics @ pos[:3, :]
            proj_trnsfm_pth = os.path.join(save_rt_dir, 'cameras', 'cam_' + '{:03d}'.format(i) + '.txt')
            if not os.path.exists(os.path.dirname(proj_trnsfm_pth)):
                os.makedirs(os.path.dirname(proj_trnsfm_pth), exist_ok=True)
            np.savetxt(proj_trnsfm_pth, proj_trnsfm)
        print(f"Vin MVS format data stored here:{save_rt_dir}")

    def load_rgbs(self):
        print(f"Loading rgb images...")
        # rgbs
        rgbs_pths = sorted(glob.glob(osp.join(self.rgbs_dir, '*.png')))
        all_imgs = []
        for imp in tqdm(rgbs_pths):
            im = cv2.imread(imp)
            all_imgs.append(im[:, :, ::-1])
        self.all_imgs = np.array(all_imgs)

    def load_masks(self):
        print(f"Loading  masks...")
        masks_pths = sorted(glob.glob(osp.join(self.masks_dir, '*.png')))
        all_masks = []
        for mp in tqdm(masks_pths):
            mask = cv2.imread(mp)
            all_masks.append(mask[:, :, ::-1])
        self.all_masks = np.array(all_masks)
    
    def load_poses(self):
        print(f"Loading poses...")
        data = load_pkl(self.pose_pkl)
        trans = np.array(data['transl'])
        rots = np.array(roma.rotvec_to_rotmat(data['global_orient']))

        tmp = np.eye(4).reshape(1, 4, 4)
        self.all_poses = np.repeat(tmp, repeats=len(trans), axis=0)
        self.all_poses[:, :3, :3] = rots
        self.all_poses[:, :3, 3] = trans
    
    def __call__(self):
        print(f"Loading rgb images...")
        # rgbs
        rgbs_pths = sorted(glob.glob(osp.join(self.rgbs_dir, '*.png')))
        all_imgs = []
        for imp in tqdm(rgbs_pths):
            im = cv2.imread(imp)
            all_imgs.append(im[:, :, ::-1])
        self.all_imgs = np.array(all_imgs)

        # masks
        print(f"Loading  masks...")
        masks_pths = sorted(glob.glob(osp.join(self.masks_dir, '*.png')))
        all_masks = []
        for mp in tqdm(masks_pths):
            mask = cv2.imread(mp)
            all_masks.append(mask[:, :, ::-1])
        self.all_masks = np.array(all_masks)

        # cam params
        print(f"Loading poses...")
        data = load_pkl(self.pose_pkl)
        trans = np.array(data['transl'])
        rots = np.array(roma.rotvec_to_rotmat(data['global_orient']))

        tmp = np.eye(4).reshape(1, 4, 4)
        self.all_poses = np.repeat(tmp, repeats=len(trans), axis=0)
        self.all_poses[:, :3, :3] = rots
        self.all_poses[:, :3, 3] = trans

        #intrinsics
        print(f"Loading intrinsics...")
        self.intrinsics = np.load(self.intrinsics_pth)

        # write data to vin format
        print(f"Writing to vin format...")
        self.write2vinformat(self.intrinsics, self.all_imgs[self.start_ind : self.end_ind],
        self.all_masks[self.start_ind : self.end_ind], self.all_poses[self.start_ind : self.end_ind],
        self.save_base_dir)
    
    @property
    def get_numof_rgbs(self,):
        return len(self.rgbs_pths)
    
    @property
    def get_numof_masks(self,):
        return len(self.masks_pths)



def create_vin_mvs_dformat_seq(intrinsics, images, masks, poses, save_rt_dir):
    for i, (im, msk, pos) in tqdm(enumerate(zip(images, masks, poses))):
        # save image
        img_save_pth = os.path.join(save_rt_dir, 'images', 'cam_' + '{:03d}'.format(i), 'view_0.png')
        if not os.path.exists(os.path.dirname(img_save_pth)):
            os.makedirs(os.path.dirname(img_save_pth), exist_ok=True)
        Image.fromarray(im).save(img_save_pth)

        # save mask
        mask_save_pth = os.path.join(save_rt_dir, 'masks', 'cam_' + '{:03d}'.format(i), 'view_0.png')
        if not os.path.exists(os.path.dirname(mask_save_pth)):
            os.makedirs(os.path.dirname(mask_save_pth), exist_ok=True)
        cv2.imwrite(mask_save_pth, msk)
        # Image.fromarray((msk * 255).astype(np.uint8)).save(mask_save_pth)

        # save proj trnsfm mat (intrinsics @ fiktiv_cam_poses)
        proj_trnsfm = intrinsics @ pos[:3, :]
        proj_trnsfm_pth = os.path.join(save_rt_dir, 'cameras', 'cam_' + '{:03d}'.format(i) + '.txt')
        if not os.path.exists(os.path.dirname(proj_trnsfm_pth)):
            os.makedirs(os.path.dirname(proj_trnsfm_pth), exist_ok=True)
        np.savetxt(proj_trnsfm_pth, proj_trnsfm)

    print(f"Vin MVS format data stored here:{save_rt_dir}")

    return None


if __name__ == "__main__":
    if False:
        import argparse
        parser = argparse.ArgumentParser("Processing captured raw data")

        parser.add_argument('-rd', '--rgbs_dir', type=str, required=True,
                            help='path to rgbs dir')
        parser.add_argument('-md' ,'--masks_dir', type=str, required=True,
                            help='path to masks dir')
        parser.add_argument('-cd', '--cams_dir', type=str, required=True,
                            help='path to projection matrices dir')
        args = parser.parse_args()

    # rgbs
    if True:
        rgbs_dir = '/gfs/team/cv/Users/aswamy/Fabien_PoseBert/L515_seqs/20220614171338_green_duster/rgb'
        rgbs_pths = sorted(glob.glob(osp.join(rgbs_dir, '*.png')))
        all_imgs = []
        for imp in tqdm(rgbs_pths):
            im = cv2.imread(imp)
            all_imgs.append(im[:, :, ::-1])
        all_imgs = np.array(all_imgs)

    # masks
    if True:
        masks_dir = '/gfs/team/cv/Users/aswamy/Fabien_PoseBert/L515_seqs/20220614171338_green_duster/mask'
        masks_pths = sorted(glob.glob(osp.join(masks_dir, '*.png')))
        all_masks = []
        for mp in tqdm(masks_pths):
            mask = cv2.imread(mp)
            all_masks.append(mask[:, :, ::-1])
        all_masks = np.array(all_masks)

    # cam params
    if True:
        data = load_pkl('/gfs/team/cv/Users/aswamy/Fabien_PoseBert/L515_seqs/20220614171338_green_duster/res/posebert_outputs.pkl')
        trans = np.array(data['transl'])
        rots = np.array(roma.rotvec_to_rotmat(data['global_orient']))

        tmp = np.eye(4).reshape(1, 4, 4)
        all_poses = np.repeat(tmp, repeats=len(trans), axis=0)
        all_poses[:, :3, :3] = rots
        all_poses[:, :3, 3] = trans

        #intrinsics
        intrinsics = np.load('/gfs/team/cv/Users/aswamy/Fabien_PoseBert/L515_seqs/20220614171547_green_duster/intrinsics.npy')

        # projection matrix
        # all_cam_poses = []
        # for pos in tqdm(all_poses):
        #     # cam pose is inv of global pose
        #     proj_mat = intrinsics @ np.linalg.inv(pos)[:3, :]
        #     all_cam_poses.append(proj_mat)
        # all_cam_poses = np.array(all_cam_poses)
    

    # write data to vin format
    start_ind = 100
    end_ind = 400
    create_vin_mvs_dformat_seq(intrinsics, all_imgs[start_ind:end_ind], all_masks[start_ind:end_ind], all_poses[start_ind:end_ind],
    '/tmp-network/user/aswamy/dataset/vin_format_ho/20220614171338_green_duster_v1/')
    print('Done!')

    




    









