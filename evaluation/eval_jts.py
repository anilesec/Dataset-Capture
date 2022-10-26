import os
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

PCK_THRESH = 50.0
AUC_MIN = 0.0
AUC_MAX = 200.0
NUM_SEQS = 87

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
        pred3d_sym, _ = compute_similarity_transform_v2(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)
    
    # return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
    #        np.stack(errors_pck, 0), np.stack(proc_rot, 0)
    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('compute jts err')
    # parser.add_argument('--seq_rgbs_dir', type=str, default=None,
    #                     help='sequnce rgbs dir')
    # parser.add_argument('--seq_pose3d_ann_dir', type=str, required=True,
    #                     help='sequence 3d pose annotation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/icp_res')
    # parser.add_argument('--seq_jts2d_est_dir', type=str, required=False,
    #                     help='sequence 2D key points  estimation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/dope_dets')
    # parser.add_argument('--seq_jts3d_est_dir', type=str, required=True,
    #                     help='sequence 3D key points estimation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/dope_dets')
    parser.add_argument('--bsln_method', type=str, required=True,
                        help='baseline method like DOPE, FRANKMOCAP, etc. ')

    # parser.add_argument('--rel_ptype', type=str, default='CONSEQ', help='CONSEQ or REF')   

    parser.add_argument('--sqn', type=str, required=True, help='seq id')   
    

    # Flags                     
    parser.add_argument('--save', type=int, default=0, choices=[0, 1],
                        help='save(1), not_save(0)')
    parser.add_argument('--viz', type=int, default=0, choices=[0, 1],
                        help='viz(1), not_viz(0)')
    parser.add_argument('--filter', type=int, default=0, choices=[0, 1],
                        help='filter(1), not_filter(0)')
    
    args = parser.parse_args()

    # print the args
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(f"args: {args}")

    RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')
    
    all_seqs_tar_pths = glob.glob(f'{RES_DIR}/*.tar')
    all_seqs_ids = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_seqs_ids.append(osp.basename(spth.split('.')[0]))

    # ALL_SEQS_IDS = ['20220805164755', '20220829155218', '20220902153221', '20220824154342', '20220830162330', '20220809161015', '20220905151029', '20220905155551', '20220905153946', '20220824104203', '20220913145135', '20220812180133', '20220811170225_dummy', '20220907153810', '20220905154829', '20220905111237', '20220902154737', '20220913151554', '20220909114359', '20220912144751', '20220909145039', '20220909140016', '20220902163904', '20220811170459', '20220705173214', '20220912155637', '20220824142508', '20220905140306', '20220824160141', '20220805165947', '20220902114024', '20220830161218', '20220902111535', '20220902104048', '20220909151546', '20220824152850', '20220912161700', '20220909111450', '20220824150228', '20220913153520', '20220824105341', '20220811172657', '20220912160620', '20220909113237', '20220823113402', '20220902111409', '20220809163444', '20220819155412', '20220824181949', '20220909142411', '20220912151849', '20220902151726', '20220811165540', '20220811163525', '20220907155615', '20220909134639', '20220909120614', '20220912143756', '20220905105332', '20220902170443', '20220905112733', '20220913144436', '20220811161557', '20220823115809', '20220902110304', '20220902163950', '20220912164407', '20220819162529', '20220823114538', '20220905142354', '20220812170512', '20220809171854', '20220829154032', '20220912165455', '20220913154643', '20220811171507', '20220909115705', '20220824155144', '20220830163143', '20220909152911', '20220824144438', '20220902164854', '20220905112623', '20220907152036', '20220905141444', '20220812174356', '20220912161552', '20220909141430', '20220824180652', '20220909121541', '20220819164041', '20220912142017', '20220912152000', '20220809170847', '20220824102636', '20220902115034', '20220812172414', '20220811154947']
    
    print('Loading Annotated poses...')
    # load anno poses
    seq_pose3d_ann_dir = osp.join(RES_DIR, args.sqn, 'icp_res')
    # bb()
    all_poses3d_ann_hom = load_poses3d_ann(seq_pose3d_ann_dir)    

    # bb()
    print('Loading/Computing 3D joints...')
    # load GT 2d and compute 3d joints of each frame
    jts3d_gt = np.loadtxt(f'{RES_DIR}/{args.sqn}/jts3d.txt')
    all_jts3d_gt = np.array(
        [tform_points(np.linalg.inv(pose), jts3d_gt) for pose in all_poses3d_ann_hom]
    )

    all_jts2d_pths = sorted(glob.glob(f'{RES_DIR}/{args.sqn}/proj_jts/*.txt'))
    all_jts2d = np.array(
        [np.loadtxt(jts2d_pth) for jts2d_pth in all_jts2d_pths]
    )
    
    assert all_jts2d.shape[:2] == all_jts3d_gt.shape[:2], "Miss-match in total number(frames) of 2d and 3d joints"

    if args.bsln_method == 'DOPE':
        print('Loading dope detections...')
        seq_jts3d_est_dir = osp.join(RES_DIR, args.sqn, 'dope_dets')
        all_jts2d, all_jts3d = load_dope_poses(dets_pkls_dir=seq_jts3d_est_dir)

        assert all_jts3d.shape[:2] == all_jts3d_gt.shape[:2], "mis-match in dope and gt number of frames!!"
    
    MPJPE_final, MPJPE_PA_final, errors_pck, mat_procs = compute_errors(all_jts3d, all_jts3d_gt)

    print(f"MPJPE_final: {MPJPE_final:04f} m ")
    print(f"MPJPE_PA_final: {MPJPE_PA_final:04f} m")
    print('Done!!')



