from posebert.constants import SMPLX_DIR, INTERHAND_DIR, INTERHAND_PREPROCESS_DIR
import ipdb
import sys
from torch.utils.data import Dataset
import os
import json
import torch
import smplx
from posebert.skeleton import update_mano_joints
import numpy as np
from tqdm import tqdm
import roma
import pickle
from posebert.skeleton import left_to_right_mano_params

def split_k_hot(x=(np.random.rand(30) > 0.5).astype(np.int32).tolist(), max_interval=2, min_seq_len=1):
    y = []
    start = 0
    while start < len(x):
        end = start + 1
        interval_size = 0
        while x[start] == 1 and end < len(x) and interval_size < max_interval:
            if x[end] == 0:
                interval_size += 1
            elif x[end] == 1:
                interval_size = 0
            end += 1
        if end - interval_size - start > min_seq_len:
            y.append([start, end - interval_size])
        start = end
    return y

def get_pose(seq, j, hand_type):
    pose = torch.from_numpy(np.asarray(seq[str(j)][hand_type]['pose'])).reshape(16, 3).float()
    trans = torch.from_numpy(np.asarray(seq[str(j)][hand_type]['trans'])).reshape(1, 3).float()
    if hand_type == 'left':
        pose = left_to_right_mano_params(pose)
        # TODO do soemthing for trans
    pose = torch.cat([pose, trans], 0)
    return pose

@torch.no_grad()
def preprocess_mano(split='train', fps=30, debug=True, max_interval=3, min_seq_len=32):
    """
    Adapted from https://github.com/facebookresearch/InterHand2.6M/blob/main/tool/MANO_render/render.py
    """
    assert split in ['train', 'val', 'test']

    with open(os.path.join(INTERHAND_DIR, f"fps{fps}", f"InterHand2.6M_{split}_MANO_NeuralAnnot.json")) as f:
        data = json.load(f)

    list_pose = []
    for id, seq in tqdm(data.items()):
        sys.stdout.flush()
        print(f"Seq: {id} - N_seq={len(list_pose)}")
        for hand_type in ['left', 'right']:
            # Check timesteps with available annot
            timesteps = []
            for t in list(seq.keys()):
                if seq[str(t)][hand_type] is not None:
                    timesteps.append(int(t))
            timesteps.sort()
            valid = [1 if t in timesteps else 0 for t in range(min(timesteps), max(timesteps) + 1)]
            print(f"Perc. of annot: {100. * np.asarray(valid).mean():.2f} %")

            # Split into subseq
            list_start_end = split_k_hot(valid, max_interval=max_interval, min_seq_len=min_seq_len)

            # Retrieve info
            for k, (start, end) in enumerate(list_start_end):
                pose = []
                j = min(timesteps) + start
                while j < min(timesteps) + end:
                    if j in timesteps:
                        pose_j = get_pose(seq, j, hand_type)
                        pose.append(pose_j.unsqueeze(0))
                        j += 1
                    else:
                        # assumption that the the first and the last timestep have a pose
                        found_prev, found_next = False, False
                        id_prev, id_next = j, j
                        while not (found_prev and found_next):
                            if not found_prev:
                                id_prev -= 1
                                found_prev = id_prev in timesteps
                            if not found_next:
                                id_next += 1
                                found_next = id_next in timesteps

                        # slerp
                        steps = torch.linspace(0, 1.0, id_next - id_prev - 1)
                        pose_prev, pose_next = get_pose(seq, id_prev, hand_type), get_pose(seq, id_next, hand_type)
                        pose_interpolated = roma.rotvec_slerp(pose_prev, pose_next, steps)

                        pose.append(pose_interpolated)
                        j = id_next

                pose = torch.cat(pose).flatten(1)

                # Visu debug
                if debug and len(list_pose) > 10:
                    from posebert.renderer import PyTorch3DRenderer
                    from pytorch3d.renderer import look_at_view_transform
                    from PIL import Image
                    mano_layer = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    renderer = PyTorch3DRenderer(800).to(device)
                    faces = torch.from_numpy(mano_layer.faces.astype(np.int32)).reshape(1, -1, 3).to(device)
                    dist, elev, azim = 0.0001, 0., -180
                    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
                    tmp_dir = '/tmp/fbaradel'
                    os.makedirs(tmp_dir, exist_ok=True)
                    os.system(f"rm {tmp_dir}/*.jpg")
                    with torch.no_grad():
                        t_max = min([128, pose.size(0)])
                        for t in tqdm(range(t_max)):
                            mesh = mano_layer(global_orient=pose[[t], :3], hand_pose=pose[[t], 3:-3], transl=pose[[t], -3:]).vertices
                            image = renderer.renderPerspective(vertices=mesh.to(device),
                                                               camera_translation=cam.to(device),
                                                               faces=faces.to(device),
                                                               focal_length=4.5,
                                                               rotation=rotation.to(device),
                                                               ).cpu().numpy()[0]
                            Image.fromarray(image).save(f"{tmp_dir}/{t:03d}.jpg")
                    os.system(
                        f"ffmpeg -framerate 30 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p video.mp4 -y")
                    os.system(f"rm {tmp_dir}/*.jpg")
                    ipdb.set_trace()

                list_pose.append(pose)

        # break

    # Save
    print(f"N_seq: {len(list_pose)} - Average sequence length: {np.mean(np.asarray([x.shape[0] for x in list_pose])):.1f} - Total number of timesteps: {np.sum(np.asarray([x.shape[0] for x in list_pose]))}")
    out_fn = os.path.join(INTERHAND_PREPROCESS_DIR, "mano", f"{split}.pkl")
    print(out_fn)
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    with open(out_fn, 'wb') as f:
        pickle.dump(list_pose, f)

    return 1

@torch.no_grad()
def extract_j3d(pose, bs=32, zero_global_orient=True):
    """
    List of poses
    """
    print("Extracting 3d joints from poses using the MANO model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True).to(device)

    poses = torch.stack(pose)
    if zero_global_orient:
        poses[:,0] = 0.
    list_poses = torch.split(poses, bs, 0)
    list_j3d = []
    for p_ in tqdm(list_poses):
        bs_ = p_.shape[0]
        out = bm(global_orient=p_[:,0].to(device), hand_pose=p_[:, 1:].flatten(1).to(device), transl=bm.transl.repeat(bs_, 1).to(device), betas=bm.betas.repeat(bs_, 1).to(device))
        verts = out.vertices
        jts = out.joints
        j3d = update_mano_joints(jts, verts)
        list_j3d.append(j3d.cpu())
    list_j3d = [x[0] for x in torch.split(torch.cat(list_j3d), 1, 0)]
    return list_j3d

def worker_init_fn(worker_id):
    seed = int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1)
    print("Worker id: {} - Seed: {}".format(worker_id, seed))
    np.random.seed(seed)

class ManoDataset(Dataset):
    def __init__(self,
                 split='test',
                 seq_len=16, 
                 training=False, 
                 n_iter=1000, n=-1,
                 range_x_init=[-0.1,0.1], range_y_init=[-0.1,0.1], range_z_init=[0.3,1.],
                 synthetic_trajectory=True,
                 ):
        super().__init__()

        self.training = training
        self.n_iter = n_iter
        self.split = split
        self.seq_len = seq_len
        self.synthetic_trajectory = synthetic_trajectory

        # Load interhand poses
        print("Loading poses from InterHands...")
        with open(os.path.join(INTERHAND_DIR, f"fps30", f"InterHand2.6M_{split}_MANO_NeuralAnnot.json")) as f:
            data = json.load(f)
        self.list_pose, self.list_j3d = [], []
        for k1, v1 in tqdm(data.items()):
            for k2, v2 in v1.items():
                if 'right' in v2.keys():
                    try:
                        pose = torch.Tensor(v2['right']['pose']).reshape(16, 3).float()
                        self.list_pose.append(pose)
                    except:
                        pass

                if n > 0 and len(self.list_pose) > n:
                    break
        
        # Extracting j3d
        self.list_j3d = extract_j3d(self.list_pose, zero_global_orient=True)

        # Args for translation
        self.range_x_init = np.arange(range_x_init[0], range_x_init[1], 0.01)
        self.range_y_init = np.arange(range_y_init[0], range_y_init[1], 0.01)
        self.range_z_init = np.arange(range_z_init[0], range_z_init[1], 0.01)
        self.range_delta = np.arange(-0.2, 0.2, 0.01)

        # MANO annots from InterHands
        with open(os.path.join(INTERHAND_PREPROCESS_DIR, "mano", f"{split}.pkl"), 'rb') as f:
            self.data_interhands = pickle.load(f)

    def __len__(self):
        if self.training:
            return self.n_iter
        else:
            return len(self.list_pose)

    def __repr__(self):
        return "ManoDataset - HandPose from Interhand: Split: {} - Size: {}".format(self.split, self.__len__())

    def __getitem__(self, idx):
        """
        Return:
            - j3d: (seq_len,21,3)
            - global_orient: (seq_len,3)
            - transl: (seq_len,3)
            - hand_pose: (45)
        """
        i = np.random.choice(range(len(self.list_pose))) if self.training else idx
        pose, j3d = self.list_pose[i], self.list_j3d[i]
        hand_pose = pose[1:].flatten(0)

        synthetic_trajectory = np.random.choice([True, False]) if self.training else False

        if synthetic_trajectory and self.synthetic_trajectory:
            # Generating a rotation and translation
            x, y, z = 0., 0., 0.
            list_global_orient = []
            list_transl = []
            rotvec0, rotvec1 = torch.randn(3), torch.randn(3)

            t_rot, t_transl = 0., 0.
            min_dur, max_dur = 10, 80
            while t_rot < self.seq_len or t_transl < self.seq_len:

                # rot
                t_ = np.random.choice(np.arange(min_dur, max_dur))
                steps = torch.linspace(0, 1.0, t_)
                rotvec_interpolated = roma.rotvec_slerp(rotvec0, rotvec1, steps)
                list_global_orient.append(rotvec_interpolated)
                rotvec0 = rotvec1.clone()
                rotvec1 = torch.randn(3)
                t_rot += t_

                # transl
                t_ = np.random.choice(np.arange(min_dur, max_dur))
                x_next = x+np.random.choice(self.range_delta)
                y_next = y+np.random.choice(self.range_delta)
                z_next = z+np.random.choice(self.range_delta)
                transl_x = np.linspace(x, x_next, t_)
                transl_y = np.linspace(y, y_next, t_)
                transl_z = np.linspace(z, z_next, t_)
                transl = torch.from_numpy(np.stack([transl_x, transl_y, transl_z], 1)).float()
                x, y, z = x_next, y_next, z_next
                list_transl.append(transl)

                t_transl += t_
            global_orient = torch.cat(list_global_orient)[:self.seq_len]
            transl = torch.cat(list_transl)[:self.seq_len]
        else:
            i = np.random.choice(range(len(self.data_interhands))) if self.training else idx % len(self.data_interhands)
            pose = self.data_interhands[i]

            if self.training:
                start = np.random.choice(range(0, max([1, pose.shape[0] - self.seq_len])))
            else:
                start = max([0, pose.shape[0] // 2 - self.seq_len // 2])
            pose = pose[start:start + self.seq_len]
            global_orient = pose[:, :3]
            transl = pose[:, -3:]
            transl = transl - transl[[0]] # start from (0,0,0)

        # Initial starting point in the scene
        x, y, z = np.random.choice(self.range_x_init), np.random.choice(self.range_y_init), np.random.choice(self.range_z_init)
        transl[:,0] = transl[:,0] + x
        transl[:,1] = transl[:,1] + y
        transl[:,2] = transl[:,2] + z
            
        # apply the rotation to the j3d and add transl
        wrist = j3d[[0]].clone()
        j3d_wrt_wrist = j3d - wrist
        j3d_wrt_wrist_up = torch.matmul(roma.rotvec_to_rotmat(global_orient).unsqueeze(1), j3d_wrt_wrist.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        j3d = j3d_wrt_wrist_up + wrist + transl.unsqueeze(1)

        # print(j3d.shape, global_orient.shape, transl.shape, hand_pose.shape)
        return j3d, global_orient, transl, hand_pose

@torch.no_grad()
def test(i_max=2, split='val', seq_len=8):
    from pytorch3d.renderer import look_at_view_transform
    from PIL import Image
    from posebert.renderer import PyTorch3DRenderer
    from posebert.skeleton import perspective_projection, visu_pose2d, get_mano_skeleton
    
    x, y, z = 0.01, 0., 0.4
    dataset = ManoDataset(n=10, split=split, seq_len=seq_len, synthetic_trajectory=False,
                        range_x_init=[x, x+0.01], range_y_init=[y, y+0.01], range_z_init=[z, z+0.01],
                        training=True,
                        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # mano
    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
    faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32)) # [1538,3]

    # renderer
    width, height = 1280, 720
    image_size = max([width, height])
    ratio = torch.Tensor([[image_size/width, image_size/height]]).float()
    f_x, f_y = 901.5, 901.7
    c_x, c_y = 664.1, 380.3
    renderer = PyTorch3DRenderer(image_size=image_size).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    focal_length = torch.Tensor([[2*f_x/image_size, 2*f_y/image_size]]) # 2 * focal_length / image_size 4
    principal_point = torch.Tensor([[c_x/width, c_y/height]])
    principal_point = (principal_point - 0.5) * 2 # values should be between -1 and 1. (0,0) is the image center
    principal_point /= ratio # solving principal point issue for rendering with non-square image

    # Create a visu
    tmp_dir = '/tmp/fbaradel/vid'
    import os
    os.makedirs(tmp_dir, exist_ok=True)
    for i, x in enumerate(tqdm(dataset)):
        j3d, global_orient, transl, hand_pose = x

        if i == i_max:
            for t in tqdm(range(j3d.shape[0])):
                # rendering
                verts = bm(global_orient=global_orient[[t]], hand_pose=hand_pose.unsqueeze(0), transl=transl[[t]]).vertices[0]
                img = renderer.renderPerspective(vertices=[verts.to(device)], 
                                            faces=[faces.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            ).cpu().numpy()[0]
                delta = np.abs(width - height)//2
                if delta > 0:
                    if width > height:
                        img = img[delta:height+delta]
                    else:
                        img = img[:,delta:width+delta]

                # projection
                j2d = perspective_projection(j3d[[t]], c_x, c_y, f_x, f_y)
                img_ = visu_pose2d(img.copy(), j2d[0], bones=get_mano_skeleton())
                img = np.concatenate([img, img_], 1)

                Image.fromarray(img).save(os.path.join(tmp_dir, f"{t:05d}.jpg"))
            fn = "video.mp4"
            cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {fn} -y"
            os.system(cmd)
            os.system(f"rm {tmp_dir}/*.jpg")
            os._exit(0)

if __name__ == "__main__":
    exec(sys.argv[1])