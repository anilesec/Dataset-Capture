import os, glob
import numpy as np
import pickle
from tqdm import tqdm
from ipdb import set_trace as bb
import pathlib

osp = os.path
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')
CAM_INTR = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]])

def ours2hocnet(**kwargs):
    return np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])

def ours2dope(**kwargs):
    return np.array([0, 1, 2, 3, 4, 5, 6, 11, 16, 7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20])


def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data

def tform_points(T, X):
    """
    X: Nx3
    T: 4x4 homogeneous
    """
    X = np.vstack((X.T, np.ones(len(X))))
    X = T @ X
    X = X[:3].T
    return X

def load_poses3d_ann(seq_pose3d_ann_dir):
    all_poses_hom = []
    frms_poses_pths = sorted(glob.glob(osp.join(seq_pose3d_ann_dir, '*', 'f_trans.txt')))
    print('Loading annotations...')
    for frm_pose_pth in tqdm(frms_poses_pths):
        pose = np.linalg.inv(np.loadtxt(frm_pose_pth))
        all_poses_hom.append(pose)

    return np.array(all_poses_hom)

def load_dope_poses(dets_pkls_dir):
    """
    @param dets_pkls_dir: path to pkls dir of a sequence
    @return poses_2d, poses_3d: 2d poses and 3d poses
    """
    # load the pkl files and save the 2D and 3D detections
    pkls_pths = sorted(glob.glob(os.path.join(dets_pkls_dir, '*')))
    # print(f"Loading dope detections .. at {dets_pkls_dir}")
    poses_3d = []
    poses_2d = []
    # bb()
    for idx, pkl_pth in tqdm(enumerate(pkls_pths)):
        pkl_data = load_pkl(pkl_pth)
        # consider only one-hand(RIGHT hand dets); this may needs to be changed in future
        # for whichever hand has better detections. For now, its just RIGHT hand
        dets = pkl_data['detections']
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
    poses_2d = np.stack(poses_2d, axis=0)
    poses_3d = np.stack(poses_3d, axis=0)
    # bb()
    return poses_2d, poses_3d

def rotran2homat(R, T):
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = R
    hom_mat[:3, 3] = T

    return hom_mat


def trnsfm_points(trnsfm, pts):
    """
    pts: Nx3
    trnsfm: 4x4 homogeneous
    """
    if trnsfm.shape == (3, 4):
        trnsfm_hom = np.vstack([trnsfm, np.array([0, 0, 0, 1])])

    pts = np.vstack((pts.T, np.ones(len(pts))))
    pts = trnsfm_hom @ pts
    pts = pts[:3].T

    return pts

trnsfm2homat = lambda t: np.vstack((t, [0., 0., 0., 1.]))

def load_ann_jts(sqn_dir, disp=None):
    if disp == True:
        all_j2d_pths = sorted(glob.glob(osp.join(sqn_dir, 'proj_jts_disp/*.txt')))
        all_j3d_pths = sorted(glob.glob(osp.join(sqn_dir, 'jts3d_disp/*.txt')))
    else:
        all_j2d_pths = sorted(glob.glob(osp.join(sqn_dir, 'proj_jts/*.txt')))
        all_j3d_pths = sorted(glob.glob(osp.join(sqn_dir, 'jts3d/*.txt')))

    all_j2d = []
    all_j3d = []
    # bb()
    for j2dp, j3dp in zip(all_j2d_pths, all_j3d_pths):
        j2d = np.loadtxt(j2dp)
        j3d = np.loadtxt(j3dp)
        all_j2d.append(j2d)
        all_j3d.append(j3d)

    return np.array(all_j2d), np.array(all_j3d)

 
def load_poses3d_ann(seq_pose3d_ann_dir):
    all_poses_hom = []
    frms_poses_pths = sorted(glob.glob(osp.join(seq_pose3d_ann_dir, '*', 'f_trans.txt')))
    print('Loading annotations...')
    for frm_pose_pth in tqdm(frms_poses_pths):
        pose = np.linalg.inv(np.loadtxt(frm_pose_pth))
        all_poses_hom.append(pose)

    return np.array(all_poses_hom)


def compute_similarity_transform(S1, S2, return_transfom=False, mu_idx=None, scale=None):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    if mu_idx is not None:
        mu1 = S1[:, [mu_idx]]
        mu2 = S2[:, [mu_idx]]
    else:
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    if scale is None:
        scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    tform = {
        'rot': R,
        'tran': t,
        'scale': scale
    }

    if return_transfom:
        return S1_hat, tform

    return S1_hat


def load_dope_det_frm_pkl(pkl_pth):
    pkl_data = load_pkl(pkl_pth)
    # consider only one-hand(RIGHT hand dets); this may needs to be changed in future
    # for whichever hand has better detections. For now, its just RIGHT hand
    dets = pkl_data['detections']
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
    jts2d = (dets[right_hand_id]['pose2d'])
    jts3d = (dets[right_hand_id]['pose3d'])

    return jts2d, jts3d