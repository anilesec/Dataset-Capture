import os
import re
from ipdb import set_trace as bb
import numpy as np
import pprint
osp =  os.path
import pathlib
from tqdm import tqdm
import glob
import cv2
import sys
from evaluation.eval_utils import *
from evaluation.viz_utils import *

PCK_THRESH = 50.0
AUC_MIN = 0.0
AUC_MAX = 200.0
NUM_SEQS = 87
osp = os.path

intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
)

def ours2hocnet(**kwargs):
    return np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Ensure that the first argument is the prediction
    Source: https://en.wikipedia.org/wiki/Kabsch_algorithm
    :param S1 predicted joint positions array 24 x 3
    :param S2 ground truth joint positions array 24 x 3
    :return S1_hat: the predicted joint positions after apply similarity transform
            R : the rotation matrix computed in procrustes analysis
    '''
    # If all the values in pred3d are zero then procrustes analysis produces nan values
    # Instead we assume the mean of the GT joint positions is the transformed joint value

    if not (np.sum(np.abs(S1)) == 0):
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        mu1 = S1[:, 0][:, None]
        mu2 = S2[:, 0][:, None]
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
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat, R
    else:
        S1_hat = np.tile(np.mean(S2, axis=0), (SMPL_NR_JOINTS, 1))
        R = np.identity(3)

        return S1_hat, R

def align_by_root(joints):
    """
    Assumes joints is 24 x 3 in SMPL order.
    Subtracts the location of the root joint from all the other joints
    """
    root = joints[0, :]

    return joints - root 

def compute_errors(preds3d, gt3ds):
    """
    Gets MPJPE after root alignment + MPJPE after Procrustes.
    Evaluates on all the 24 joints joints.
    Inputs:
    :param gt3ds: N x 24 x 3
    :param preds: N x 24 x 3
    :returns
        MPJPE : scalar - mean of all MPJPE errors
        MPJPE_PA : scalar- mean of all MPJPE_PA errors
        errors_pck : N x 24 - stores the error b/w GT and prediction for each joint separate
        proc_mats : N x 3 x 3 - for each frame, stores the 3 x 3 rotation matrix that best aligns the prediction and GT
    """
    errors, errors_pa, errors_pck = [], [], []

    proc_rot = []

    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds3d)):
        # gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_root(gt3d)
        pred3d = align_by_root(pred3d)
        # bb()
        # Compute MPJPE
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Joint errors for PCK Calculation
        # joint_error_maj = joint_error[SMPL_MAJOR_JOINT]
        # errors_pck.append(joint_error_maj)

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d)
        # bb()
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)
    
    # return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
    #        np.stack(errors_pck, 0), np.stack(proc_rot, 0)
    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), None, None


if __name__ == "__main__":
    RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    # get all 3djts files
    print('Listing all jts3d txt files...')
    all_hocnet_jts3d_txts = sorted(glob.glob(osp.join(RES_DIR, '*/jts3d_hocnet/*.txt')))

    all_jts3d_ann = []
    all_jts3d_hocnet = []
    all_MPJPE = []
    all_MPJPE_PA = []
    missing_poses_frms = []
    for txtp in tqdm(all_hocnet_jts3d_txts):
        # check if GT 3D jts exist for corresponding hocnet det
        jts3d_ann_pth = pathlib.Path(txtp.replace('jts3d_hocnet', 'jts3d_disp'))
        if not jts3d_ann_pth.exists():
            print(f'Missing: {jts3d_ann_pth}')
            missing_poses_frms.append(jts3d_ann_pth)
            continue
        jts3d_hocnet = np.loadtxt(txtp)
        all_jts3d_hocnet.append(jts3d_hocnet)

        jts3d_ann = np.loadtxt(jts3d_ann_pth)
        all_jts3d_ann.append(all_jts3d_ann)
        # bb()
        MPJPE, MPJPE_PA, _, _ = compute_errors(jts3d_hocnet.reshape(1, 21, 3), jts3d_ann[ours2hocnet()].reshape(1, 21, 3))
        all_MPJPE.append(MPJPE)
        all_MPJPE_PA.append(MPJPE_PA)

        # bb()
        # img_hocnet = viz_hand_jts3d(jts3d_hocnet, 'CP', grid_axis='ON',
        #             line_sz=2, dot_sz=4, elev=-90, azim=-90,
        #             xlim=[jts3d_hocnet[:, 0].min() - 0.1, jts3d_hocnet[:, 0].max() + 0.1],
        #             ylim=[jts3d_hocnet[:, 1].min() - 0.1, jts3d_hocnet[:, 1].max() + 0.1],
        #             zlim=[jts3d_hocnet[:, 2].min() - 0.1, jts3d_hocnet[:, 2].max() + 0.1],
        #             title='3D Joints in wrist frame')
        # img_ann = viz_hand_jts3d(jts3d_ann, 'OURS', grid_axis='ON',
        #             line_sz=2, dot_sz=4, elev=-90, azim=-90,
        #             xlim=[jts3d_ann[:, 0].min() - 0.1, jts3d_ann[:, 0].max() + 0.1],
        #             ylim=[jts3d_ann[:, 1].min() - 0.1, jts3d_ann[:, 1].max() + 0.1],
        #             zlim=[jts3d_ann[:, 2].min() - 0.1, jts3d_ann[:, 2].max() + 0.1],
        #             title='3D Joints in wrist frame')
        # plt.imsave('./hocnet.png', img_hocnet)
        # plt.imsave('./ann.png', img_ann)
        # bb()

    print(f"MPJPE eval on {len(all_MPJPE)} frames: {np.mean(all_MPJPE)} m | {np.median(all_MPJPE)} m")    
    print(f"MPJPE_PA eval on {len(all_MPJPE_PA)}: {np.mean(all_MPJPE_PA)} m | {np.median(all_MPJPE_PA)} m")    

    print('Done!!')




