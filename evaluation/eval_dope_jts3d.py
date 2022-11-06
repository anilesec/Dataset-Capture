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

PCK_THRESH = 2.0
AUC_MIN = 0.0
AUC_MAX = 200.0
NUM_SEQS = 87

def ours2dope(**kwargs):
    return np.array([0, 1, 2, 3, 4, 5, 6, 11, 16, 7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20])


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 9g, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))

def compute_auc(xpts, ypts):
    """
    Calculates the AUC.
    :param xpts: Points on the X axis - the threshold values
    :param ypts: Points on the Y axis - the pck value for that threshold
    :return: The AUC value computed by integrating over pck values for all thresholds
    """
    a = np.min(xpts)
    b = np.max(xpts)
    from scipy import integrate
    myfun = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(myfun, a, b)[0]
    return auc

def compute_pck(errors, THRESHOLD):
    """
    Computes Percentage-Correct Keypoints
    :param errors: N x 12 x 1
    :param THRESHOLD: Threshold value used for PCK
    :return: the final PCK value
    """
    errors_pck = errors <= THRESHOLD
    errors_pck = np.mean(errors_pck, axis=1)
    return np.mean(errors_pck)

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
        
        # Compute MPJPE
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        # Joint errors for PCK Calculation
        # joint_error_maj = joint_error[SMPL_MAJOR_JOINT]
        # errors_pck.append(joint_error_maj)

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)
    
    # return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
    #        np.stack(errors_pck, 0), np.stack(proc_rot, 0)
    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), None, None

def load_dope_poses_from_pkls(pkls_pths):
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

if __name__ == "__main__":
    RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')
    # MPJPE and MPJPE_PA
    # get all dope dets(pkls)
    print('Listing all dope pkl files...')
    all_dope_pkls = sorted(glob.glob(osp.join(RES_DIR, '*/dope_dets/*.pkl')))

    all_jts3d_ann = []
    all_jts3d_dope = []
    all_MPJPE = []
    all_MPJPE_PA = []
    missing_poses_frms = []
    for pklpth in tqdm(all_dope_pkls):
        # bb()
        # check if pose/3D jts exist for corresponding dope det
        jts3d_ann_pth = pathlib.Path(pklpth.replace('dope_dets', 'jts3d_disp').replace('pkl', 'txt'))
        if not jts3d_ann_pth.exists():
            print(f'Missing: {jts3d_ann_pth}')
            missing_poses_frms.append(jts3d_ann_pth)
            continue
        _, jts3d_dope = load_dope_det_frm_pkl(pklpth)
        # all_jts3d_dope.append(jts3d_dope)

        jts3d_ann = np.loadtxt(jts3d_ann_pth)
        # all_jts3d_ann.append(all_jts3d_ann)
        # bb()
        MPJPE, MPJPE_PA, _, _ = compute_errors(jts3d_dope.reshape(1, 21, 3), jts3d_ann[ours2dope()].reshape(1, 21, 3))
        all_MPJPE.append(MPJPE)
        all_MPJPE_PA.append(MPJPE_PA)

        # img_dope = viz_hand_jts3d(jts3d_dope, 'DOPE', grid_axis='ON',
        #             line_sz=2, dot_sz=4, elev=-90, azim=-90,
        #             xlim=[jts3d_dope[:, 0].min() - 0.1, jts3d_dope[:, 0].max() + 0.1],
        #             ylim=[jts3d_dope[:, 1].min() - 0.1, jts3d_dope[:, 1].max() + 0.1],
        #             zlim=[jts3d_dope[:, 2].min() - 0.1, jts3d_dope[:, 2].max() + 0.1],
        #             title='3D Joints in wrist frame')
        # img_ann = viz_hand_jts3d(jts3d_ann, 'OURS', grid_axis='ON',
        #             line_sz=2, dot_sz=4, elev=-90, azim=-90,
        #             xlim=[jts3d_ann[:, 0].min() - 0.1, jts3d_ann[:, 0].max() + 0.1],
        #             ylim=[jts3d_ann[:, 1].min() - 0.1, jts3d_ann[:, 1].max() + 0.1],
        #             zlim=[jts3d_ann[:, 2].min() - 0.1, jts3d_ann[:, 2].max() + 0.1],
        #             title='3D Joints in wrist frame')
        # plt.imsave('./dope.png', img_dope)
        # plt.imsave('./ann.png', img_ann)
        # bb()
    # all_jts3d_dope = np.array(all_jts3d_dope)
    # all_jts3d_ann = np.array(all_jts3d_ann)
    # all_MPJPE = np.array(all_MPJPE)
    # all_MPJPE_PA = np.array(all_MPJPE_PA)

    # compute errors
    # MPJPE, MPJPE_PA = compute_errors(preds3d=all_jts3d_dope, gt3ds=all_jts3d_ann)
    print(f"MPJPE eval on {len(all_MPJPE)} frames: {np.mean(all_MPJPE)} m")    
    print(f"MPJPE_PA eval on {len(all_MPJPE_PA)}: {np.mean(all_MPJPE_PA)} m")    


    

    print('Done!!')



