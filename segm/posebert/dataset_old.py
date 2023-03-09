import json
import numpy as np
import sys
import ipdb
from posebert.skeleton import inverse_projection_to_3d, preprocess_skeleton_torch, visu_pose3d, convert_jts, perspective_projection, visu_pose2d
from PIL import Image, ImageOps
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import roma
import smplx
try:
    import _pickle as pickle
except:
    import pickle

os.umask(0x0002)  # give write right to the team for all created files

CONTACT_POSE = '/tmp-network/dataset/contactpose'

def stats_cam_intrinsics():
    """
    kinect2_right
    focal_length: x={533.8558726083219} - y={533.9060001703177}
    principal point: x={473.40603980311704} - y={270.14347071052225}

    kinect2_left
    focal_length: x={520.6131697223657} - y={521.3136402402431}
    principal point: x={477.19634220979475} - y={275.5041232079494}

    kinect2_middle
    focal_length: x={527.131752150462} - y={527.6703920066655}
    principal point: x={491.36222791837804} - y={272.6102738305844}
    """
    fn = '/tmp-network/user/aswamy/cp_for_posebert/all_intrinsics.npz'
    x = np.load(fn, allow_pickle=True)
    for cam in ['kinect2_right', 'kinect2_left', 'kinect2_middle']:
        y = np.asarray(x['arr_0'].tolist()[cam]) # [N,3,3]
        print()
        print(cam)
        print(y)
        print(f"focal_length: x={set(y[:,0,0].tolist())} - y={set(y[:,1,1].tolist())}")
        print(f"principal point: x={set(y[:,0,-1].tolist())} - y={set(y[:,1,-1].tolist())}")

def show_one_seq(
    # fn='/tmp-network/dataset/contactpose/full1_use/apple/images_full/kinect2_left/all_gt_3d_jts_cam.npy',
    # fn='/tmp-network/dataset/contactpose/full1_use/apple/images_full/kinect2_right/all_gt_3d_jts_cam.npy',
    # fn='/tmp-network/dataset/contactpose/full1_use/apple/images_full/kinect2_middle/all_gt_3d_jts_cam.npy',
    fn='/tmp-network/dataset/contactpose/full1_use/ps_controller/images_full/kinect2_middle/all_gt_3d_jts_cam.npy',
    t_start=0, t_end=10000,
):
    x = convert_jts(np.load(fn), 'contactpose', 'mano')

    camera_name = fn.split('/')[-2]
    A, K = get_A(camera_name), get_K(camera_name)
    print("Initial camera instrinsics: ", K)
    print("F*** affine transform: ", A)

    imgs_dir = fn.replace('all_gt_3d_jts_cam.npy','color')

    tmp_dir = "/tmp/fab"
    os.makedirs(tmp_dir, exist_ok=True)
    for t in tqdm(range(t_start, t_end)):
        img_rgb = np.asarray(Image.open(os.path.join(imgs_dir, f"frame{t:03d}.png")))
        j3d = x[t]

        # perspective projection
        points = x[t]
        projected_points = points / points[:, [-1]]
        print(projected_points.shape)
        projected_points =  A @ K @ projected_points.T
        projected_points = projected_points.T[:, :2]
        img_2d = visu_pose2d(img_rgb.copy(), projected_points)

        # 3d centered around the wrist
        j3d_centered = j3d - j3d[[0]]
        img_3d = visu_pose3d(j3d_centered.reshape(1, -1, 3), res=img_rgb.shape[0])

        img = np.concatenate([img_rgb, img_2d, img_3d], 1)
        Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")
    fn = "video.mp4"
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {fn} -y"
    os.system(cmd)
    os.system(f"rm {tmp_dir}/*.jpg")

    return 1

def load_dope_pkl(all_dope_dets):
    """
    taken from https://oss.navercorp.com/anilkumar-swamy/hand-obj-recon/blob/fullbody-dev/ContactPose/scripts/load_dope_dets.py#L35
    @param dets_pkls_dir: path to pkls dir of a sequence
    @return poses_2d, poses_3d: 2d poses and 3d poses
    """
    # all_dope_dets = load_pkl(dope_pkl_pth)
    poses_3d = []
    poses_2d = []
    mask = []
    # for idx, dope_dict in tqdm(enumerate(all_dope_dets)):
    for idx, dope_dict in enumerate(all_dope_dets):
        # consider only one-hand(RIGHT hand dets); this may needs to be changed in future
        # for whichever hand has better detections. For now, its just RIGHT hand
        dets = dope_dict['detections']
        if len(dets) > 0:
            if len(dets) > 1:
                if (dets[0]['hand_isright'] and dets[1]['hand_isright']) or \
                        (not dets[0]['hand_isright'] and not dets[1]['hand_isright']):
                    if dets[0]['score'] >= dets[1]['score']:
                        right_hand_id = 0
                    elif dets[0]['score'] < dets[1]['score']:
                        right_hand_id = 1
                    else:
                        raise ValueError("Error!! Agrrrr!! Check your inefficient conditional statements >_<")
                elif dets[0]['hand_isright']:
                    right_hand_id = 0
                elif dets[1]['hand_isright']:
                    right_hand_id = 1
                else:
                    raise ValueError("Error!! Agrrrr!! Check your stupid conditional statements >_<")
            else:
                right_hand_id = 0
            
            poses_2d.append(dets[right_hand_id]['pose2d'])
            poses_3d.append(dets[right_hand_id]['pose3d'])
            mask.append(1)
        else:
            poses_2d.append(np.zeros((21, 2)))
            poses_3d.append(np.zeros((21, 3)))
            mask.append(0)
    poses_2d = np.stack(poses_2d, axis=0)
    poses_3d = np.stack(poses_3d, axis=0)
    mask = np.stack(mask)

    return poses_2d, poses_3d, mask

def update_seq_with_missing_frames(pose, start_inds, num_missin_frms, const=0.):
    if len(num_missin_frms) == 0:
        return pose, np.ones(pose.shape[0])
    start_inds.append(pose.shape[0]-1)
    num_missin_frms.append(0)

    prev = 0
    pose_, valid_ = [], []
    for t in range(len(start_inds)):
                                                # observed
                                                pose_obs = pose[prev:start_inds[t]]
                                                pose_.append(pose_obs)
                                                valid_.extend([1 for _ in range(pose_obs.shape[0])])

                                                # missing
                                                pose_missing = const * pose[[start_inds[t]]].repeat(num_missin_frms[t], axis=0)
                                                pose_.append(pose_missing)
                                                valid_.extend([0 for _ in range(pose_missing.shape[0])])

                                                # update
                                                prev = start_inds[t]

    pose = np.concatenate(pose_)
    valid = np.stack(valid_)
    return pose, valid


def get_K(camera_name):
  if camera_name == 'kinect2_left':
      return np.asarray([[520.61316972,  0.,         477.19634221],
                        [  0.,         521.31364024, 275.50412321],
                        [  0.,           0.,           1.        ]])
  elif camera_name == 'kinect2_right':
      return np.asarray([[533.85587261,  0.,         473.4060398],
                        [  0.,          533.90600017, 270.14347071],
                        [  0.,           0.,           1.        ]])
  elif camera_name == 'kinect2_middle':
      return np.asarray([[527.13175215,  0.,         491.36222792],
                        [  0.,         527.67039201, 272.61027383],
                        [  0.,           0.,           1.        ]])
  else:
      raise NotImplementedError  

def get_A(camera_name, W=960, H=540):
  """
  Get the affine transformation matrix applied after 3D->2D projection
  """
  def flipud(H):
      return np.asarray([[1, 0, 0], [0, -1, H], [0, 0, 1]])
  def fliplr(W):
      return np.asarray([[-1, 0, W], [0, 1, 0], [0, 0, 1]])
  def transpose():
      return np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

  if camera_name == 'kinect2_left':
      return np.dot(fliplr(H), transpose())
  elif camera_name == 'kinect2_right':
      return np.dot(flipud(W), transpose())
  elif camera_name == 'kinect2_middle':
      return np.dot(fliplr(W), flipud(H))
  else:
      raise NotImplementedError


def prepare_annots(split='test', out_dir='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose',
debug=False, cam_debug='kinect2_left'):
    bm = smplx.create('/tmp-network/SlowHA/user/fbaradel/data/SMPLX', 'mano', use_pca=True, num_pca_comps=18, is_rhand=True)

    VAL_PERS = [5, 15, 25, 35, 45]
    TRAIN_PERS = [x for x in range(51) if x not in VAL_PERS]
    persons = TRAIN_PERS if split == 'train' else VAL_PERS
    
    # Looping
    seq2pose = {}
    n_frames = 0
    for pers in tqdm(persons):
        if os.path.isdir(os.path.join(CONTACT_POSE, f"full{pers}_use")):
            xs = os.listdir(os.path.join(CONTACT_POSE, f"full{pers}_use"))
            for x in xs:
                if os.path.isdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x)):
                    ys = os.listdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x))
                    for y in ys:
                        if os.path.isdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x, y)):
                            zs = os.listdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x, y))
                            for z in zs:
                                if os.path.isdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x, y, z)):
                                    cs = os.listdir(os.path.join(CONTACT_POSE, f"full{pers}_use", x, y, z))
                                    if 'all_gt_3d_jts_cam.npy' in cs:
                                        fn = os.path.join(CONTACT_POSE, f"full{pers}_use", x, y, z, 'all_gt_3d_jts_cam.npy')

                                        # missing frames
                                        fn_ = fn.replace('all_gt_3d_jts_cam.npy', 'missing_frames.pkl')
                                        with open(fn_, 'rb') as f:
                                            miss = pickle.load(f)
                                        num_missin_frms = miss['num_missin_frms']
                                        start_inds = miss['start_inds']

                                        # 3d pose
                                        pose = np.load(fn)
                                        pose = convert_jts(pose, 'contactpose', 'mano')

                                        # 2d pose
                                        K = get_A(z) @ get_K(z)
                                        image_size = [960, 540] if z == 'kinect2_middle' else [540, 960]
                                        j3d = torch.from_numpy(pose)
                                        j3d = j3d / j3d[..., [-1]]
                                        K_ = torch.from_numpy(K).reshape(1, 3, 3).repeat(j3d.shape[0], 1, 1)
                                        j2d = torch.einsum('bij,bkj->bki', K_, j3d)[..., :2].float().numpy()

                                        # dope
                                        fn_dope = fn.replace('all_gt_3d_jts_cam.npy', 'dope_raw/dope_dets.pkl')
                                        with open(fn_dope, 'rb') as f:
                                            dope = pickle.load(f)
                                        dope_2d, dope_3d, mask = load_dope_pkl(dope)
                                        dope_2d = convert_jts(dope_2d, 'dope_hand', 'mano')
                                        dope_3d = convert_jts(dope_3d, 'dope_hand', 'mano')

                                        # mano
                                        fn_ = fn.replace('all_gt_3d_jts_cam.npy', 'mano_annotations.pkl')
                                        fn_ = fn.split('images_full')[0] + 'mano_fits_15.json'
                                        with open(fn_, 'rb') as f:
                                            mano = json.load(f)
                                        mano_pca = torch.Tensor(mano[0]['pose']).float()
                                        betas = torch.Tensor(mano[0]['betas']).reshape(10)
                                        with torch.no_grad():
                                                hand_pose = bm(hand_pose=mano_pca.reshape(1, -1), betas=betas.reshape(1, -1)).hand_pose.reshape(15, 3) # axis-angle representation

                                        # visu
                                        if z == cam_debug and debug:
                                            t_ = 0
                                            seq = os.path.dirname(fn)

                                            # 2d/3d
                                            img1 = np.asarray(Image.open(os.path.join(seq, 'color', f"frame{t_:03d}.png"))) 
                                            img2 = visu_pose2d(img1.copy(), dope_2d[t_])
                                            img3 = visu_pose3d(dope_3d[[t_]], res=img1.shape[0])
                                            img4 = visu_pose2d(img1.copy(), j2d[t_])
                                            l_img = [img4, img2, img3]

                                            # mano
                                            from pytorch3d.renderer import look_at_view_transform
                                            from renderer import PyTorch3DRenderer
                                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                            global_orient = torch.Tensor([[np.pi/2., 0., 0.]])
                                            for i_v in range(2):
                                                if i_v == 0:
                                                    bm = smplx.create('/tmp-network/SlowHA/user/fbaradel/data/SMPLX', 'mano', use_pca=True, num_pca_comps=18, is_rhand=True)
                                                    out = bm(global_orient=global_orient, hand_pose=mano_pca.reshape(1, -1)) # [778,3]
                                                    verts = out.vertices[0]
                                                else:
                                                    bm = smplx.create('/tmp-network/SlowHA/user/fbaradel/data/SMPLX', 'mano', use_pca=False, is_rhand=True)
                                                    verts = bm(global_orient=global_orient, hand_pose=hand_pose.reshape(1, -1)).vertices[0] # [778,3]

                                                image_size=img1.shape[0]
                                                faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32)) # [1538,3]
                                                renderer = PyTorch3DRenderer(image_size=image_size).to(device)
                                                dist, elev, azim = 0.8, 0., 180
                                                rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
                                                focal_length = torch.Tensor([[4., 4.]])
                                                principal_point = torch.Tensor([[0.5 * image_size, 0.5 * image_size]]) # in pixel space
                                                principal_point = ((principal_point / image_size) - 0.5) * 2.
                                                img5 = renderer.renderPerspective(vertices=[verts.to(device)], 
                                                                                faces=[faces.to(device)],
                                                                                rotation=rotation.to(device),
                                                                                camera_translation=cam.to(device),
                                                                                principal_point=principal_point.to(device),
                                                                                focal_length=focal_length,
                                                                                ).cpu().numpy()[0]
                                                l_img.append(img5)

                                            img = np.concatenate(l_img, 1)
                                            Image.fromarray(img).save('img.jpg')
                                            ipdb.set_trace()

                                        # update
                                        t_frames, _ =  update_seq_with_missing_frames(np.arange(0, pose.shape[0]), start_inds, num_missin_frms, const=-1.)
                                        pose, valid =  update_seq_with_missing_frames(pose, start_inds, num_missin_frms)
                                        j2d, _ =  update_seq_with_missing_frames(j2d, start_inds, num_missin_frms)
                                        dope_2d, _ = update_seq_with_missing_frames(dope_2d, start_inds, num_missin_frms)
                                        dope_3d, _ = update_seq_with_missing_frames(dope_3d, start_inds, num_missin_frms)
                                        mask, _ = update_seq_with_missing_frames(mask, start_inds, num_missin_frms)

                                        n_frames += pose.shape[0]
                                        try:
                                            assert j2d.shape[0] == \
                                                valid.shape[0] == \
                                                    dope_2d.shape[0] == \
                                                        dope_3d.shape[0] == \
                                                            mask.shape[0] == \
                                                            t_frames.shape[0]
                                        except:
                                            ipdb.set_trace()
                                        seq2pose[os.path.join(CONTACT_POSE, f"full{pers}_use", x, y, z)] = {
                                            # GT
                                            'j2d': j2d, # [t,21,2]
                                            'valid': valid, # [t]
                                            't_frames': t_frames, # [t]
                                            # DOPE
                                            'dope2d': dope_2d, # [t,21,2]
                                            'dope3d': dope_3d, # [t,21,3]
                                            'mask': mask, # [t]
                                            # J3D for training
                                            'j3d': pose, # [t,21,3]
                                            # MANO fits provided by the authors
                                            'manoHandPose': hand_pose.numpy(), # [15,3] - axis angle
                                            'manoBetas': betas.numpy() # [10]

                                        }
    
    print(f"Stats: N_seq={len(seq2pose)}, N_frames={n_frames}")

    # Saving
    fn = os.path.join(out_dir, split, f"j3d.pkl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    print(f"Dumping sequences in {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(seq2pose, f)

    return 1

class J3dDataset(Dataset):
    def __init__(self,
                 data_dir='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose',
                 split='test',
                 cam='all',
                 seq_len=64, training=False, n_iter=1000, n=-1, data_augment=False,
                 max_tilt_angle=180, max_roll_angle=180, max_vertical_angle=180,
                 sanity_check_inverse_projection=True,
                 focal_length=530,
                 image_size=960
                 ):
        super().__init__()
        assert split in ['train', 'test']
        assert cam in ['kinect2_middle', 'kinect2_right', 'kinect2_left', 'all']

        self.data_dir = data_dir
        self.cam = cam
        self.split = split
        self.seq_len = seq_len
        self.training = training
        self.n_iter = n_iter
        self.data_augment = data_augment
        self.sanity_check_inverse_projection = sanity_check_inverse_projection
        self.focal_length = focal_length
        self.image_size = image_size

        # J3D
        with open(os.path.join(self.data_dir, split, "j3d.pkl"), 'rb') as f:
            self.seq2jannots = pickle.load(f)
        seq = list(self.seq2jannots.keys())

        # Use a specific camera
        if self.cam == 'all':
            # take all cams into account
            self.seq = seq
        else:
            self.seq = [x for x in seq if x.split('/')[-1] == self.cam]

        assert len(self.seq) > 0
        if n > 0:
            self.seq = self.seq[:n]

        # Camera intrinsics
        self.image_size = torch.Tensor([self.image_size, self.image_size]).float()
        self.focal_length = torch.Tensor([self.focal_length, self.focal_length]).float()
        self.K = torch.zeros(3, 3).float()
        self.K[0,0], self.K[1,1] = focal_length, focal_length
        self.K[-1,-1] = 1.
        self.K[0,-1], self.K[1,-1] = image_size / 2., image_size / 2.
        self.K_inverse = torch.inverse(self.K)
        
        # Data augmentation by applying a random rotation
        self.tilt = torch.arange(-max_tilt_angle, max_tilt_angle + 0.01) * np.pi / 180
        self.roll = torch.arange(-max_roll_angle, max_roll_angle + 0.01) * np.pi / 180
        self.ry = torch.arange(-max_vertical_angle, max_vertical_angle + 0.01) * np.pi / 180
        self.trans_x = torch.arange(-0.2, 0.2 + 0.01, step=0.01)
        self.trans_y = torch.arange(-0.2, 0.2 + 0.01, step=0.01)
        self.trans_z = torch.arange(0.2, 1. + 0.01, step=0.01)

    def __len__(self):
        if self.training:
            return self.n_iter
        else:
            return len(self.seq)

    def __repr__(self):
        return "J3D: Dirname: {} - Size: {}".format(self.data_dir, self.__len__())

    def __getitem__(self, idx):
        # Retrieve info
        i = np.random.choice(range(len(self.seq))) if self.training else idx
        seqname = self.seq[i]
        j3d = torch.from_numpy(self.seq2jannots[seqname]['j3d']).float()
        valid = torch.from_numpy(self.seq2jannots[seqname]['valid']).float()
        j3d = torch.from_numpy(self.seq2jannots[seqname]['j3d']).float()
        
        # Sample a subseq
        if self.training:
            start = np.random.choice(range(0, max([1, j3d.shape[0] - self.seq_len])))
        else:
            start = max([0, j3d.shape[0] // 2 - self.seq_len // 2])
        j3d, valid = j3d[start:start + self.seq_len], valid[start:start + self.seq_len]
        timesteps = torch.arange(start, start+j3d.shape[0])
        t_missing = self.seq_len - j3d.shape[0]
        if t_missing > 0:
            if not self.training or np.random.choice([True, False]):
                # starting with real seq
                j3d = torch.cat([j3d, j3d[-1:].repeat(t_missing, 1, 1)])
                valid = torch.cat([valid, valid[-1:].repeat(t_missing)])
                timesteps = torch.cat([timesteps, timesteps[-1:].repeat(t_missing)])
            else:
                # starting with an invalid mask - for real time demo
                j3d = torch.cat([j3d[:1].repeat(t_missing, 1, 1), j3d])
                valid = torch.cat([valid[:1].repeat(t_missing), valid])
                timesteps = torch.cat([timesteps[:1].repeat(t_missing), timesteps])          

        # Data augmentation
        if self.training and self.data_augment:
            # Random rotation
            tilt, ry, roll = np.random.choice(self.tilt), np.random.choice(self.ry), np.random.choice(self.roll)
            rot = roma.rotvec_composition([torch.Tensor([tilt, 0., 0.]), torch.Tensor([0., ry, 0.]), torch.Tensor([0., 0., roll])])

            # Apply
            trans0 = j3d[:, 0]
            j3d = torch.matmul(roma.rotvec_to_rotmat(rot).unsqueeze(0).unsqueeze(0), j3d.unsqueeze(-1)).squeeze(-1)
            j3d = j3d - j3d[[0], [0]] + trans0.unsqueeze(1)

        # Center j3d
        if self.training:
            j3d = j3d - j3d[:, [0]].mean(0, keepdims=True)
            j3d[..., 0] = j3d[..., 0] + np.random.choice(self.trans_x)
            j3d[..., 1] = j3d[..., 1] + np.random.choice(self.trans_y)
            j3d[..., 2] = j3d[..., 2] + np.random.choice(self.trans_z)  

        # project 3d in 2d
        j3d_ = j3d / (j3d[..., [-1]] + 1e-8)
        K_ = self.K.unsqueeze(0).repeat(j3d_.shape[0], 1, 1)
        j2d = torch.einsum('bij,bkj->bki', K_, j3d_)[..., :2]

        # update valid
        valid2d_x = (j2d[...,0] > 0).float() * (j2d[...,0] < self.image_size[0]).float()
        valid2d_y = (j2d[...,1] > 0).float() * (j2d[...,1] < self.image_size[1]).float()
        valid2d = ((valid2d_x * valid2d_y).mean(1) > 0.5).long()
        valid = (valid * valid2d).long()

        # sanity check about inverse reprojection
        if self.sanity_check_inverse_projection:
            K_inverse_ = self.K_inverse.unsqueeze(0).repeat(self.seq_len, 1, 1)
            distance = j3d[..., [-1]]
            points = torch.cat([j2d, torch.ones_like(j2d[..., :1])], -1)
            # I_inverse_: [b,3,3] - points: [b,21,3]
            points_ = torch.einsum('bij,bkj->bki', K_inverse_, points)
            j3d_back = (points_ * distance) + 1e-8
            err = (j3d_back - j3d).abs().sum(-1).sum(-1).mean().item()
            print(err)
            try:
                assert err < 1e-3
            except:
                ipdb.set_trace()
        
        manoValid = 1.
        if self.seq2jannots[seqname]['manoHandPose'].sum() == 0.:
            manoValid = 0.

        return {
            'j3d': j3d,
            'j2d': j2d,
            'valid': valid,
            'timesteps': timesteps,
            'image_size': self.image_size,
            'K': self.K,
            'K_inverse': self.K_inverse,
            'manoHandPose': torch.from_numpy(self.seq2jannots[seqname]['manoHandPose']),
            'manoValid': torch.Tensor([manoValid]).float(),
            'manoBetas': torch.from_numpy(self.seq2jannots[seqname]['manoBetas']),
        }

def test_j3d(data_dir='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose',
         split='train', data_augment=True, training=True, i_max=1000, seq_len=64, batch_size=1):
    from torch.utils.data import DataLoader

    dataset = J3dDataset(data_dir=data_dir, training=training, data_augment=data_augment, split=split, seq_len=seq_len, sanity_check_inverse_projection=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                        shuffle=False, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)

    for i, x in enumerate(tqdm(loader, total=len(loader))):
        print(x['valid'].float().mean())
        pass
        
        if i == 100:
            break

    tmp_dir = '/tmp/fbaradel'
    os.makedirs(tmp_dir, exist_ok=True)
    print('valid', x['valid'].float().mean())
    for t in tqdm(range(x['j2d'].shape[1])):
        img1 = np.zeros((int(dataset.image_size[0]), int(dataset.image_size[1]), 3)).astype(np.uint8)
        img2 = visu_pose2d(img1.copy(), x['j2d'][0,t])
        print('j2d', x['j2d'][0,[t], [0]])
        print('j3d', x['j3d'][0,[t], [0]])
        img3 = visu_pose3d(x['j3d'][0,[t]] - x['j3d'][0,[t], [0]], res=960)
        img = np.concatenate([img2, img3])
        Image.fromarray(img).save(f"{tmp_dir}/{t:5d}.jpg")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y"
    os.system(cmd)
    os.system(f"rm {tmp_dir}/*.jpg")

class ContactPoseDataset(Dataset):
    def __init__(self,
                 data_dir='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose',
                 cam='kinect2_right',
                 split='test',
                 n=-1,
                 ):
        super().__init__()
        assert split in ['train', 'test']
        assert cam in ['kinect2_middle', 'kinect2_right', 'kinect2_left', 'all']

        self.data_dir = data_dir
        self.split = split
        self.cam = cam

        # Data
        with open(os.path.join(self.data_dir, split, "j3d.pkl"), 'rb') as f:
            self.seq2annots = pickle.load(f)
        seq = list(self.seq2annots.keys())

        # Use a specific camera
        if self.cam == 'all':
            # take all cams into account
            self.seq = seq
        else:
            self.seq = [x for x in seq if x.split('/')[-1] == self.cam]

        if n > 0:
            self.seq = self.seq[:n]

        self.focal_length = 530
        self.cam2K = {}
        for cam in ['kinect2_right', 'kinect2_left', 'kinect2_middle']:
            K = torch.zeros(3, 3).float()
            K[0,0], K[1,1] = self.focal_length, self.focal_length
            K[-1,-1] = 1.
            if cam == 'kinect2_middle':
                width, height = 960, 540
            else:
                width, height = 540, 960
            # get_K(cam)[0,-1]
            # K[0, -1] = height /2.
            K[0, -1] = width /2.
            # get_K(cam)[1,-1]
            # K[1, -1] = width /2.
            K[1, -1] = height /2.
            K_inverse = torch.inverse(K)
            self.cam2K[cam] = {'K': K, 'K_inverse': K_inverse}

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return "ContactPose: Dirname: {} - Size: {}".format(self.data_dir, self.__len__())
    
    def __getitem__(self, i):
        # Retrieve info
        seq = self.seq[i]
        annot = self.seq2annots[seq]
        annot_ = {k: torch.from_numpy(v).float() for k, v in annot.items()}

        annot_['seq'] = seq

        cam = seq.split('/')[-1]
        image_size = [960, 540] if cam == 'kinect2_middle' else [540, 960] # middle if of shape W=960, H=460
        annot_['image_size'] = torch.Tensor(image_size)

        annot_['K'] = self.cam2K[cam]['K']
        annot_['K_inverse'] = self.cam2K[cam]['K_inverse']
        return annot_

def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1)
    print("Worker id: {} - Seed: {}".format(worker_id, seed))
    np.random.seed(seed)

def compute_dope_error(data_dir='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose', split='Test', perc_thresh_pck2d=0.1, cam='kinect2_right'):
    from torch.utils.data import DataLoader
    from skeleton import compute_similarity_transform_batch

    dataset = ContactPoseDataset(data_dir=data_dir, split=split, cam=cam)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        shuffle=False, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)

    pa_mpjpe, pck2d = [], []
    masks, valids = [], []
    for i, x in enumerate(tqdm(loader, total=len(loader))):
        valid = x['valid'][0]
        mask = x['mask'][0]

        # PA-MPJPE only on frames detected by DOPE.. so far
        j3d_hat = x['dope3d'][0]
        j3d = x['j3d'][0]
        j3d, j3d_hat = j3d - j3d[:, [0]], j3d_hat - j3d_hat[:, [0]] # not sure it is useful to center because we run procruste alignement after...
        j3d_hat_ = compute_similarity_transform_batch(j3d_hat.numpy(), j3d.numpy())
        err = 1000. * np.sqrt( ((j3d - j3d_hat_)** 2).sum(axis=-1)).mean(axis=-1)

        # PCK-2D
        j2d_hat = x['dope2d'][0].numpy()
        j2d = x['j2d'][0].numpy()
        dist = np.sqrt(((j2d - j2d_hat) ** 2).sum(-1))

        # Bbox
        bbox_size = np.max(j2d, 1) - np.min(j2d, 1)
        bbox_size = np.max(bbox_size, 1).reshape(-1, 1)
        thresh = perc_thresh_pck2d * bbox_size # a percent of the bbox
        pck2d_ = 100. * (dist < thresh).mean(1)  # [T]

        # Fill in - take only DOPE preds into account - do not take missing detections
        for j, mask_j in enumerate(mask):
            # if mask_j.item() != valid[j].item():
                # ipdb.set_trace()
            if mask_j.item() == 1:
                pa_mpjpe.append(err[j])
                pck2d.append(pck2d_[j])

        masks.append(mask)
        valids.append(valid)

    # Log
    pa_mpjpe, pck2d = np.stack(pa_mpjpe), np.stack(pck2d)
    perc_missing_detections = 100. * (1. - np.concatenate(masks).sum() / np.concatenate(valids).sum())
    print(f"MISSING DETECTIONS={perc_missing_detections:.2f}% - PA-MPJPE={pa_mpjpe.mean():.2f}mm - PCK2D={pck2d.mean():.1f}%")

    # Visu
    t = j2d.shape[0] // 2
    img1 = np.asarray(Image.open(os.path.join(x['seq'][0], 'color', f"frame{int(x['t_frames'][0,t].item()):03d}.png"))) 
    img2 = visu_pose2d(img1.copy(), j2d_hat[t])
    img3 = visu_pose2d(img1.copy(), j2d[t])
    img = np.concatenate([img1, img2, img3], 1)
    Image.fromarray(img).save('img.jpg')
    

    #     if x['valid'].float().mean() < 0.5:
    #         i =  torch.where(x['valid'].float().mean(1) < 1)[0][0].item()
    #         print("Final seq: ", x['valid'][i].float().mean())
    #         break
    #     if i == i_max:
    #         i = 0
    #         break

    # # print('j3d: ', x['j3d'][0])
    # # print('j2d: ', x['j2d'][0])
    # tmp_dir = "/tmp/fab"
    # os.makedirs(tmp_dir, exist_ok=True)
    # i = 0 if i > batch_size else i
    # print(x['seq'][i])
    # width, height = x['image_size'][i].long().numpy()
    # print(f"WxH={width}x{height}")
    # for t in tqdm(range(x['j2d'][i].shape[0])):
        
    #     if x['valid'][i][t] == 1:
    #         # print(t, x['valid2d'][i][t].item())
    #         # RGB
    #         img1 = np.asarray(Image.open(os.path.join(x['seq'][i], 'color', f"frame{x['timesteps'][i][t].item():03d}.png"))) # [H,W,3]
    #         # 2D projection
    #         img2 = np.zeros((height,width,3)).astype(np.uint8)
    #         img2 = visu_pose2d(img1, x['j2d'][i][t])
    #         # Cat
    #         # img = np.concatenate([img1, img2], 1)
    #         img = img2
    #     else:
    #         img = np.zeros((height,width,3)).astype(np.uint8)
    #     Image.fromarray(img).save(f"{tmp_dir}/{t:05d}.jpg")
    # fn = "video.mp4"
    # cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {fn} -y"
    # os.system(cmd)
    # os.system(f"rm {tmp_dir}/*.jpg")
    # return 1



if __name__ == "__main__":
    exec(sys.argv[1])