# python posebert/evaluate.py -h
import numpy as np
import pprint
import argparse
import os
import json
from collections import OrderedDict
np.set_printoptions(suppress=True)
# from contactpose.utilities.dataset import ContactPose
# import contactpose.utilities.misc as mutils
# reference: https://github.com/aymenmir1/3dpw-eval (add to the final code)

PCJ_THRESH = 5.0
AUC_MIN = 0.0
AUC_MAX = 200.0


def load_json(fpth):
    with open(fpth, 'rb') as fid:
        anno = json.load(fid)

    return anno

def compute_auc(xpts, ypts):
    """
    Calculates the AUC.
    :param xpts: Points on the X axis - the threshold values
    :param ypts: Points on the Y axis - the pcj value for that threshold
    :return: The AUC value computed by integrating over pcj values for all thresholds
    """
    a = np.min(xpts)
    b = np.max(xpts)
    from scipy import integrate
    myfun = lambda x: np.interp(x, xpts, ypts)
    auc = integrate.quad(myfun, a, b)[0]
    return auc


def compute_pcj(errors, THRESHOLD):
    """
    Computes Percentage-Correct joints
    :param errors: N x 21 x 1
    :param THRESHOLD: Threshold value used for PCj
    :return: the final PCj value
    """
    errors_pcj = errors <= THRESHOLD
    errors_pcj = np.mean(errors_pcj, axis=1)
    return np.mean(errors_pcj)


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
    Subtracts the location of the root joint from all the other joints
    """
    root = joints[0, :]

    return joints - root


def compute_3djts_errors(preds3d, gt3ds):
    """
    Gets MPJPE after root alignment + MPJPE after Procrustes.
    Evaluates on all the 24 joints joints.
    Inputs:
    :param gt3ds: N x 21 x 3
    :param preds: N x 21 x 3
    :returns
        MPJPE : scalar - mean of all MPJPE errors
        MPJPE_PA : scalar- mean of all MPJPE_PA errors
        errors_pcj : N x 21 - stores the error b/w GT and prediction for each joint separate
        proc_mats : N x 3 x 3 - for each frame, stores the 3 x 3 rotation matrix that best aligns the prediction and GT
    """
    errors, errors_pa, errors_pcj = [], [], []

    proc_rot = []

    for i, (gt3d, pred3d) in enumerate(zip(gt3ds, preds3d)):
        # gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_root(gt3d)
        pred3d = align_by_root(pred3d)

        # Compute MPJPE
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))

        errors_pcj.append(joint_error)

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)

    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
           np.stack(errors_pcj, 0), np.stack(proc_rot, 0)

def compute_cam_pose_error(cam_poses_gt, cam_poses_pred):
    """
    cam pose errors(rel transltaion error and rel rotation error)
    cam_poses_gt, cam_poses_pred: Nx4x4
    """
    ################# Rotation Error##################
    # GT relative pose(1-2, 2-3, 3-4, ...)
    gt_rel_poses = np.vstack(
        [cam_poses_gt[i] @ inv_trnsfm(cam_poses_gt[i-1].reshape(1, -1, 4))
         for i in range(1, len(cam_poses_gt))]
    )
    
    # Pred relative pose(1-2, 2-3, 3-4, ...)
    pred_rel_poses = np.vstack(
        [cam_poses_pred[i] @ inv_trnsfm(cam_poses_pred[i-1].reshape(1, -1, 4))
         for i in range(1, len(cam_poses_pred))]
    )

    # geodesic error b/w rel rots
    rel_rot_err = np.array(
        [(geodesic_distance_for_rotations(gt_rel_poses[i][:3, :3], pred_rel_poses[i][:3, :3]) * (180 / np.pi))
             for i in range(len(gt_rel_poses))]
    )

    ################# Translation Error##################
    # rel trabsltion error
    rel_tran_err = np.linalg.norm((pred_rel_poses[:, :3, 3] - gt_rel_poses[:, :3, 3]), axis=1)

    return rel_rot_err, rel_tran_err

def inv_trnsfm(T):
    """
    compute inverse transform of batch of 3x4 transforms
    :param T: Nx3x4
    :return: Nx4x4
    """
    assert T.ndim == 3, "T should be of size Nx3x4 or Nx4x4"

    if T.shape[-2] != 4:
        hom_axis = np.repeat(np.array([0, 0, 0, 1]).reshape(1, 1, -1), T.shape[0], 0)
        T = np.concatenate((T, hom_axis), 1)

    return np.linalg.inv(T)

trnsfm2homat = lambda t: np.vstack((t, [0., 0., 0., 1.]))

def safe_arccos(x):
    """
        Returns the arcosine of x, clipped between -1 and 1.
        Use this when you know x is a cosine, but it might be
        slightly over 1 or below -1 due to numerical errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))


def geodesic_distance_for_rotations(R1, R2):
    """
        Returns the geodesic distance between two rotation matrices.
        It is computed as the angle of the rotation :math:`R_1^{*} R_2^{-1}``.
    """
    R = np.dot(R1, R2.T)
    axis1, angle1 = axis_angle_from_rotation(R)  # @UnusedVariable
    # print(axis1)
    return angle1


def axis_angle_from_rotation(R):
    """
        Returns the *(axis,angle)* representation of a given rotation.
        There are a couple of symmetries:
        * By convention, the angle returned is nonnegative.
        * If the angle is 0, any axis will do.
          In that case, :py:func:`default_axis` will be returned.
        cite: PyGeometry
    """
    angle = safe_arccos((R.trace() - 1) / 2)

    if angle == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    else:
        v = np.array(
            [R[2, 1] - R[1, 2],
             R[0, 2] - R[2, 0],
             R[1, 0] - R[0, 1]]
        )

        computer_with_infinite_precision = False
        if computer_with_infinite_precision:
            axis = (1 / (2 * np.sin(angle))) * v
        else:
            # OK, the formula above gives (theoretically) the correct answer
            # but it is imprecise if angle is small (dividing by a very small
            # quantity). This is way better...
            axis = (v * np.sign(angle)) / np.linalg.norm(v)

        return axis, angle


def evaluate_jts(jts3d_gt, jts3d_pred):
    # Joint errors and procrustes matrices
    mpjpe, mpjpe_pa, errors_pcj, mat_procs = compute_3djts_errors(jts3d_pred * 100., jts3d_gt * 100.)

    # PCJ value
    pcj = compute_pcj(errors_pcj, PCJ_THRESH) * 100.

    # AUC value
    if False: #(not required for now)
        auc_range = np.arange(AUC_MIN, AUC_MAX)
        pcj_aucs = []
        for pcj_thresh_ in auc_range:
            err_pcj_tem = compute_pcj(errors_pcj, pcj_thresh_)
            pcj_aucs.append(err_pcj_tem)
        auc = compute_auc(auc_range / auc_range.max(), pcj_aucs)

    # RP 

    metrics_dict = {
        'MPJPE': mpjpe,
        'MPJPE_PA': mpjpe_pa,
        'PCJ': pcj,
        # 'AUC': auc
    }

    return metrics_dict

def evaluate_poses(cam_poses_gt, cam_poses_pred):
    # poses errorr
    rot_err, tran_err = compute_cam_pose_error(cam_poses_gt, cam_poses_pred)
    metrics_dict = {
        'RRE': rot_err,
        'RTE': tran_err
    }

    return metrics_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Hand joints evalutation script')
    parser.add_argument('--jts3d_pred_pth', type=str, required=True,
                        help='path to 3d hand joints predictions')
    parser.add_argument('--jts3d_gt_pth', type=str, required=True,
                        help='path to 3d GT hand joints')
    parser.add_argument('--jts3d_dope_pth', type=str, required=True,
                        help='path to 3d DOPE hand joints')
    parser.add_argument('--poses_gt_pth', type=str, required=True,
                        help='path to GT camera poses')
    parser.add_argument('--poses_pred_pth', type=str, required=True,
                        help='path to estimated camera poses')                        
    args = parser.parse_args()

    # print the args
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(f"args: {args}")

    # load GT jts
    jts3d_gt = np.load(args.jts3d_gt_pth)

    # load pred jts
    jts3d_pred = np.load(args.jts3d_pred_pth)

    # laod DOPE jts
    jts3d_dope = np.load(args.jts3d_dope_pth)

    breakpoint()
    # evaluate jts: posebert v/s GTs
    eval_dict_jts_pb = OrderedDict(evaluate_jts(jts3d_gt, jts3d_pred))

    print(f" \n------------PoseBert vs GT------------")
    print(f"MPJPE: {eval_dict_jts_pb['MPJPE']:.4f} cm")
    print(f"MPJPE_PA: {eval_dict_jts_pb['MPJPE_PA']:.4f} cm")
    print(f"PCJ: {eval_dict_jts_pb['PCJ']:.3f}% with threshold {PCJ_THRESH} cm")

    # evaluate jts:  DOPE v/s GTs
    eval_dict_jts_dp = OrderedDict(evaluate_jts(jts3d_gt, jts3d_dope))
    print(f" \n------------DOPE vs GT------------")
    print(f"MPJPE: {eval_dict_jts_dp['MPJPE']:.4f} cm")
    print(f"MPJPE_PA: {eval_dict_jts_dp['MPJPE_PA']:.4f} cm")
    print(f"PCJ: {eval_dict_jts_dp['PCJ']:.3f}% with threshold {PCJ_THRESH} cm")

    breakpoint()
    # load GT cam poses
    cam_poses_gt = np.load(args.poses_gt_pth)

    # load pred cam poses
    cam_poses_pred = np.load(args.poses_pred_pth)

    # compute cam pose error
    eval_dict_poses = OrderedDict(evaluate_poses(cam_poses_gt, cam_poses_pred))
    print(f" \n------------Rotation Error------------")
    print(f"\nRot mat diff b/w rel-poses of GT and virtual poses:")
    print(f"Mean: {np.mean(eval_dict_poses['RRE']):04f}° Median: {np.median(eval_dict_poses['RRE']):04f}°"
          f" Std: {np.std(eval_dict_poses['RRE']):04f}°")
    print(f"Min: {np.min(eval_dict_poses['RRE']):04f}° Max: {np.max(eval_dict_poses['RRE']):04f}°")
    rel_rot_err_prcntls = np.percentile(eval_dict_poses['RRE'], [25, 50, 75, 90])
    print(f"Percentiles - 25th: {rel_rot_err_prcntls[0]:04f}° 50th: {rel_rot_err_prcntls[1]:04f}° "
          f"75th: {rel_rot_err_prcntls[2]:04f}° 90th:{rel_rot_err_prcntls[3]:04f}°")

    print(f" \n------------Translation Error------------")
    print(f"Mean: {np.mean(eval_dict_poses['RTE']):04f}m Median: {np.median(eval_dict_poses['RTE']):04f}m "
          f"Std: {np.std(eval_dict_poses['RTE']):04f}m")
    print(f"Min: {np.min(eval_dict_poses['RTE']):04f}m Max: {np.max(eval_dict_poses['RTE']):04f}m")
    rel_tran_err_prcntls = np.percentile(eval_dict_poses['RTE'], [25, 50, 75, 90])
    print(f"Percentiles - 25th: {rel_tran_err_prcntls[0]:04f}m 50th: {rel_tran_err_prcntls[1]:04f}m "
          f"75th: {rel_tran_err_prcntls[2]:04f}m 90th:{rel_tran_err_prcntls[3]:04f}m")

    # breakpoint()




