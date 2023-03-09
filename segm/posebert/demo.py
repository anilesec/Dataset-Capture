from asyncio import constants
import os
import pickle

from pandas import factorize
import ipdb
from PIL import Image
from posebert.skeleton import estimate_translation, visu_pose2d, visu_pose3d, get_mano_skeleton, convert_jts, perspective_projection, normalize_skeleton_by_bone_length, get_mano_traversal, update_mano_joints, visu_pose3d, normalize_skeleton_by_bone_length_updated
from posebert.constants import ANIL_DIR, SMPLX_DIR
import numpy as np
from tqdm import tqdm
import torch
import smplx
from posebert.model import PoseBERT
import roma
from posebert.renderer import PyTorch3DRenderer
from pytorch3d.renderer import look_at_view_transform
from torch import nn
import sys
import argparse
import glob
import cv2

# ANIL_DIR = '/gfs/team/cv/Users/aswamy/Fabien_PoseBert/L515_seqs'
ANIL_DIR = '/scratch/1/user/aswamy/data/hand-obj'

def compute_similarity_transform(S1, S2, mu_idx=0):
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
        if mu_idx is None:
            mu1 = S1.mean(axis=1, keepdims=True)
            mu2 = S2.mean(axis=1, keepdims=True)
        else:
            mu1 = S1[:, mu_idx][:, None]
            mu2 = S2[:, mu_idx][:, None]
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
        SMPL_NR_JOINTS = S2.shape[0]
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
        errors_pck.append(joint_error)        

        # Compute MPJPE_PA and also store similiarity matrices to apply them later to rotation matrices for MPJAE_PA
        pred3d_sym, R = compute_similarity_transform(pred3d, gt3d, mu_idx=None)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_pa.append(np.mean(pa_error))

        proc_rot.append(R)
    
    # return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), \
    #        np.stack(errors_pck, 0), np.stack(proc_rot, 0)
    return np.mean(np.array(errors)), np.mean(np.array(errors_pa)), np.stack(errors_pck, 0), None

def compute_similarity_transform_batch(S1, S2, return_transfom=False, mu_idx=None, scale=None, mask=[]):
    """Batched version of compute_similarity_transform."""
    if mask !=[]:
        assert(mask.shape[0] == S1.shape[0]),f"mask.shape[0]={mask.shape[0]} whereas S1.shape[0]={S1.shape[0]}, they should be equal"
    
    S1_hat = np.zeros_like(S1)
    s_list, R_list, t_list = [], [], []
    for i in range(S1.shape[0]):
        if mask != [] and mask[i]: #To avoid exception check mask before    
            if return_transfom:
                S1_hat[i], (s, R, t)  = compute_similarity_transform_updated(S1[i], S2[i], return_transfom, mu_idx, scale)
                s_list.append(s)
                R_list.append(R)
                t_list.append(t)
            else:
                S1_hat[i] = compute_similarity_transform_updated(S1[i], S2[i], return_transfom, mu_idx, scale)
        else:
            S1_hat[i] = S1[i]
            if return_transfom:
                s_list.append(0)
                R_list.append(0)
                t_list.append(0)
    return S1_hat, s_list, R_list, t_list

def load_dope(seqname='ho_greenduster', t_start=0, t_max=-1, tmp_img_dir='/scratch/1/user/fbaradel/tmp/roar_demo', take_left_hands_into_account=False,
            anil_loading=False,
            salma_loading=True,
            fn='postprocessed_fix_righthands_postpro_dope.pkl'
):
    # os.makedirs(tmp_img_dir, exist_ok=True)
    # os.system(f"rm {tmp_img_dir}/*")

    if salma_loading:
        SALMA_DIR = '/scratch/1/user/aswamy/data/salma-hand-obj'
        with open(os.path.join(SALMA_DIR, seqname, fn), 'rb') as f:
        # with open(os.path.join(SALMA_DIR, seqname, 'postprocessed_padded_righthands.pkl'), 'rb') as f:
            x = pickle.load(f)
        print(x.keys())
        print(1000.* x['mpjpe'], 1000.*x['pa_mpjpe'], x['pck@0.0300'])
        print(1000.* x['mpjpe_postpro'], 1000.*x['pa_mpjpe_postpro'], x['pck@0.0300'])
        j3ds_gt = convert_jts(torch.from_numpy(x['gts']), 'dope_hand', 'mano').numpy()
        if False:
            j3ds = convert_jts(torch.from_numpy(x['j3d_centered']), 'dope_hand', 'mano')
            j2ds = convert_jts(torch.from_numpy(x['j2ds']),'dope_hand', 'mano')
        else:
            j3ds = convert_jts(torch.from_numpy(x['j3d_postprocessed']), 'dope_hand', 'mano')
            j2ds = convert_jts(torch.from_numpy(x['j2ds']),'dope_hand', 'mano')
        masks = torch.Tensor(x['keep_mask'])
        width, height = 1280, 720
        hand_isrights = 1
        img_paths = []

        if False:
            gts_centered = j3ds_gt - j3ds_gt[:,[0]]
            rescaled_j3ds = j3ds.numpy()
            keep = masks.numpy()

            pck_thresh = 0.03
            errors = torch.sqrt(torch.sum((torch.from_numpy(rescaled_j3ds - gts_centered))**2, axis=-1)).numpy()
            pck = compute_pck(errors, THRESHOLD=pck_thresh) * 100.

            print('masks', masks.mean())
            mpjpe = np.mean(errors[np.where(keep)])
            j3ds_hat = []
            for t in range(rescaled_j3ds.shape[0]):
                # j3ds_hat_t = compute_similarity_transform_updated(rescaled_j3ds[t], gts_centered[t])
                j3ds_hat_t,_ = compute_similarity_transform(rescaled_j3ds[t], gts_centered[t], mu_idx=None)
                j3ds_hat.append(j3ds_hat_t)
            j3ds_hat = np.stack(j3ds_hat)

            # j3ds_hat, shapes, Rots, Trans = compute_similarity_transform_batch(rescaled_j3ds, gts_centered, return_transfom=True, mask=np.asarray(keep))

            pa_mpjpe = np.mean(torch.sqrt(torch.sum((torch.from_numpy(j3ds_hat - gts_centered))**2, axis=-1)).numpy()[np.where(keep)])
            print(1000. * mpjpe, 1000. * pa_mpjpe, pck.mean())
            
            mpjpe, pa_mpjpe, errors, _ = compute_errors(j3ds.numpy(), j3ds_gt)
            pck = compute_pck(errors, THRESHOLD=pck_thresh) * 100.
            print(1000. * mpjpe, 1000. * pa_mpjpe, pck.mean())
            # ipdb.set_trace()

        return j2ds, j3ds, masks, width, height, img_paths, hand_isrights, j3ds_gt
        ipdb.set_trace()

    dirname = os.path.join(ANIL_DIR, seqname, 'dope_dets')
    fns = os.listdir(dirname)
    fns.sort()

    fns = fns[t_start:]

    if t_max > 0:
        fns = fns[:t_max]

    j2ds, j3ds, masks, img_paths, hand_isrights = [], [], [], [], []
    # print("loading DOPE...")
    # for t, fn in enumerate(tqdm(fns)):
    print('nb dope files: ', len(fns))
    for t, fn in enumerate(fns):
        with open(os.path.join(dirname, fn), 'rb') as f:
            dets = pickle.load(f)

        try:
            if anil_loading:
                pkl_data = dets
                dets_ = pkl_data['detections']
                if len(dets_) > 1:
                    if (dets_[0]['hand_isright'] and dets_[1]['hand_isright']) or \
                            (not dets_[0]['hand_isright'] and not dets_[1]['hand_isright']):
                        if dets_[0]['score'] >= dets_[1]['score']:
                            right_hand_id = 0
                        elif dets_[0]['score'] < dets_[1]['score']:
                            right_hand_id = 1
                        else:
                            raise ValueError("Error!! Agrrrr!! Check your inefficient conditional statements >_<")
                    elif dets_[0]['hand_isright']:
                        right_hand_id = 0
                    elif dets_[1]['hand_isright']:
                        right_hand_id = 1
                    else:
                        raise ValueError("Error!! Agrrrr!! Check your stupid conditional statements >_<")
                    
                    masks.append(1.)
                    hand_isrights.append(1.)
                else:
                    right_hand_id = 1
                    masks.append(1.)
                    hand_isrights.append(1.)
                jts2d = (dets[right_hand_id]['pose2d'])
                jts3d = (dets[right_hand_id]['pose3d'])
                j2d = jts2d
                j3d = jts3d
            else:
                hand_isright = [p['hand_isright'] for p in dets['detections']]
                if True in hand_isright:
                    idx = np.where(hand_isright)[0][0]
                    hand = dets['detections'][idx]
                    j2d = hand['pose2d']
                    j3d = hand['pose3d']
                    masks.append(1.)
                    hand_isrights.append(1.)
                else:
                    if take_left_hands_into_account:
                        print('left hand only...')
                        idx = 0
                        hand = dets['detections'][idx]
                        j2d = hand['pose2d']
                        j3d = hand['pose3d']
                        masks.append(1.)
                        hand_isrights.append(0.)
                    else:
                        j2d = np.zeros((21, 2)).astype(np.float32)
                        j3d = np.zeros((21, 3)).astype(np.float32)
                        masks.append(0.)
                        hand_isrights.append(1.)
        except:
            j2d = np.zeros((21, 2)).astype(np.float32)
            j3d = np.zeros((21, 3)).astype(np.float32)
            masks.append(0.)
            hand_isrights.append(1.)

        # dope to mano world
        j2d = convert_jts(j2d.reshape(1, 21, 2), 'dope_hand', 'mano')[0]
        j3d = convert_jts(j3d.reshape(1, 21, 3), 'dope_hand', 'mano')[0]

        j2ds.append(j2d)
        j3ds.append(j3d)

        # if dets['img_pth'].split('/')[1] == 'gfs-ssd':
        #     # copy img if in gfs
        #     img_name = dets['img_pth'].split('/')[-1]
        #     os.system(f"cp {dets['img_pth']} {os.path.join(tmp_img_dir, img_name)}")
        #     dets['img_pth'] = os.path.join(tmp_img_dir, img_name)
        #     # print(t, dets['img_pth'])
        img_paths.append(dets['img_pth'])
    
    # img = Image.open(dets['img_pth'])
    # width, height = img.size
    width, height = 1280, 720

    # print(f"Pourcentage of missing frames: {(1. - np.asarray(masks).mean())*100.:.1f}")

    # ipdb.set_trace()

    return torch.from_numpy(np.stack(j2ds)), torch.from_numpy(np.stack(j3ds)), torch.Tensor(masks), width, height, img_paths, hand_isrights

"""
2D boxes are N*4 numpy arrays (float32) with each row at format <xmin> <ymin> <xmax> <ymax>
"""

def get_bbox(x):
    x_min = torch.min(x[...,0], 1).values
    y_min = torch.min(x[...,1], 1).values
    x_max = torch.max(x[...,0], 1).values
    y_max = torch.max(x[...,1], 1).values
    bbox = torch.stack([x_min, y_min, x_max, y_max], 1)
    return bbox

def area2d(b):
    """ compute the areas for a set of 2D boxes"""
    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

def overlap2d(b1, b2):
    """ compute the overlaps between a set of boxes b1 and 1 box b2 """
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    width = np.maximum(0, xmax - xmin)
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)
    height = np.maximum(0, ymax - ymin)
    return width * height

def iou2d(b1, b2):
    """ compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]
    assert b2.shape[0] == 1
    o = overlap2d(b1, b2)
    return o / (area2d(b1) + area2d(b2) - o)

def ours2dope(**kwargs):
    return np.array([0, 1, 2, 3, 4, 5, 6, 11, 16, 7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20])

@torch.no_grad()
def visu(model, seqname='ho_greenduster', t_start=0, t_max=128, debug=True, use_precomputed_posebert=False, load_intrinsics=False,
         f_x= 899.783, f_y=900.019,
        c_x= 653.768, c_y=362.143,
        take_left_hands_into_account=False,
        outdirname='res/',
        render=False,
        bm=None, faces=None,
):
    # Load GT
    gt_dir = os.path.join(ANIL_DIR, seqname, 'jts3d_disp')
    jts3d_fns = os.listdir(gt_dir)
    jts3d_fns.sort()
    jts3d = []
    for x in jts3d_fns:
        y = np.loadtxt(os.path.join(gt_dir, x))
        y = y[ours2dope()]
        jts3d.append(y)
    jts3d = np.stack(jts3d)
    # jts3d = jts3d - jts3d[:,[0]]
    jts3d = convert_jts(jts3d, 'dope_hand', 'mano')

    name = os.path.join(outdirname, f"video.mp4")
    posebert_fn = os.path.join(outdirname, 'posebert_outputs.pkl')

    sys.stdout.flush()

    # DOPE outputs
    j2ds, j3ds, masks, width, height, img_paths, hand_isrights = load_dope(seqname=seqname, t_start=t_start, t_max=t_max, take_left_hands_into_account=take_left_hands_into_account)
    t_max = min([j2ds.shape[0], jts3d.shape[0]])
    j2ds, j3ds, masks, img_paths, hand_isrights = j2ds[:t_max], j3ds[:t_max], masks[:t_max], img_paths[:t_max], hand_isrights[:t_max]
    jts3d = jts3d[:t_max]
    j2ds_dope, j3ds_dope = j2ds.clone(), j3ds.clone()

    # GT relative pose
    T, _ = compute_relp_cam_from_jts3d(jts3d)

    # metrics
    all_MPJPE, all_MPJPE_PA, all_pck  = [], [], []
    j3ds_ = j3ds.clone() # convert_jts(j3ds.clone(), 'mano', 'dope_hand')
    j3ds_ = j3ds_ - j3ds_[:,[0]]
    for t in range(j3ds.shape[0]):
        if masks[t] == 1:
            MPJPE, MPJPE_PA, errors_pck, _ = compute_errors(j3ds_[t].numpy().reshape(-1, 21, 3), jts3d[t].reshape(-1, 21, 3))
            all_MPJPE.append(MPJPE)
            all_MPJPE_PA.append(MPJPE_PA)
            all_pck.append(errors_pck.reshape(21, 1))
        else:
            all_pck.append(np.ones((21, 1)))
    if True:
    # if False:
        print(f"*** DOPE ***")
        print(f"MPJPE eval on {len(all_MPJPE)} frames: {1000. * np.mean(all_MPJPE):2f} m")    
        print(f"MPJPE_PA eval on {len(all_MPJPE_PA)}: {1000. * np.mean(all_MPJPE_PA):2f} m")
        dope_pck_dict = dict()
        for pck_th in np.arange(0.01, 0.10001, 0.01):
            pck_final = compute_pck(all_pck, pck_th) * 100.
            dope_pck_dict[f'{pck_th:.4f}'] = pck_final
            print(f"PCK with thresh {pck_th:.4f}: {pck_final:.4f} %")

    # camera
    if load_intrinsics:
        # Load intrinsics
        with open(os.path.join(ANIL_DIR, 'intrinsics.txt'), 'r') as f:
            intrinsics = f.readlines()
        c_x, c_y = [float(h) for h in intrinsics[0].split('[')[2].split(']')[0].split(' ')]
        f_x, f_y = [float(h) for h in intrinsics[0].split('[')[3].split(']')[0].split(' ')]
    else:
        pass
        # f_x, f_y = 901.5, 901.7
        # c_x, c_y = 664.1, 380.3
        # f_x, f_y = focal_length, focal_length
        # c_x, c_y = width/2., height/2.
    # print(f"f_x:{f_x} - f_y:{f_y}")
    # print(f"c_x:{c_x} - c_y:{c_y}")
    image_size=max([width, height])
    ratio = torch.Tensor([[image_size/width, image_size/height]]).float()
    focal_length = torch.Tensor([[2*f_x/image_size, 2*f_y/image_size]])
    principal_point = torch.Tensor([[c_x/width, c_y/height]])
    principal_point = (principal_point - 0.5) * 2 # values should be between -1 and 1. (0,0) is the image center
    principal_point /= ratio # solving principal point issue for rendering with non-square image

    transls = estimate_translation(j3ds, j2ds, f_x, f_y, c_x, c_y)
    j3ds_transl = j3ds + transls.unsqueeze(1)
    # print("first transl: ", transls[0])

    # normalize j3d
    traversal, parents = get_mano_traversal()
    mean = update_mano_joints(bm().joints.detach(), bm().vertices.detach())[0].cpu().numpy()
    j3ds_norm = []
    # for t in tqdm(range(j3ds.shape[0])):
    for t in range(j3ds.shape[0]):
        j3ds_norm.append(normalize_skeleton_by_bone_length(j3ds[t].numpy(), mean, traversal, parents))
    j3ds_norm = torch.from_numpy(np.stack(j3ds_norm))
    transls = estimate_translation(j3ds_norm, j2ds, f_x, f_y, c_x, c_y)

    # inverse teh 3d of left hand
    l_j3ds_norm = []
    for t, is_right in enumerate(hand_isrights):
        if is_right:
            l_j3ds_norm.append(j3ds_norm[t])
        else:
            x = j3ds_norm[t].clone()
            x_ = torch.stack([-x[:,0], x[:,1], x[:,2]],1)
            l_j3ds_norm.append(x_)
            # img = visu_pose3d(j3d.reshape(1, 21, 3).copy(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
            # Image.fromarray(img).save('img.jpg')
            # ipdb.set_trace()
    j3ds_norm = torch.stack(l_j3ds_norm)

    # add transl to 3d to place the hand in the scene
    j3ds_norm_transl = j3ds_norm + transls.unsqueeze(1)
    # print("first transl from norm: ", transls[0])

    # Rendering
    img_res = max([width, height])
    renderer = PyTorch3DRenderer(img_res, max_faces_per_bin=10000).to(device)
    dist, elev, azim = 1e-5, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    rotation = rotation.to(device)
    cam = cam.to(device)
    
    with torch.no_grad():
        tmp_dir = '/scratch/1/user/fbaradel/tmp/roar_demoo'
        os.makedirs(tmp_dir, exist_ok=True)
        
        if not use_precomputed_posebert:
            # PoseBERT
            # print("running posebert on the entire sequence...")
            sys.stdout.flush()
            global_orient_hat, transl_hat, hand_pose_hat = model.forward_full_sequence(j3d=j3ds_norm_transl.unsqueeze(0).to(device), 
                                                                                        j2d=j2ds.unsqueeze(0).to(device), 
                                                                                        mask=masks.unsqueeze(0).to(device).bool())
            global_orient = roma.rotmat_to_rotvec(global_orient_hat).cpu()[0]
            hand_pose = roma.rotmat_to_rotvec(hand_pose_hat).flatten(1).cpu()[0]
            transl = transl_hat.cpu()[0]

            # Rendering
            # print("rendering...")
            img_paths = img_paths
            l_jts_hat = []
            # for t, img_path in  enumerate(tqdm(img_paths)):
            for t, img_path in  enumerate(img_paths):
                sys.stdout.flush()
                # print(img_paths[t])
                
                out_hat = bm(
                    global_orient=global_orient[[t]].to(device), 
                    hand_pose=hand_pose.unsqueeze(0).to(device),
                    transl=transl[[t]].to(device)
                    )
                verts_hat = out_hat.vertices[0]
                jts_hat = update_mano_joints(out_hat.joints, out_hat.vertices)
                l_jts_hat.append(jts_hat.detach().cpu().numpy())

                if render:
                    print(t, j3ds.shape[0])
                    img = Image.open(img_paths[t])
                    with torch.no_grad():
                        img3 = renderer.renderPerspective(vertices=[verts_hat.to(device)], 
                                                        faces=[faces.to(device)],
                                                        rotation=rotation.to(device),
                                                        camera_translation=cam.to(device),
                                                        principal_point=principal_point.to(device),
                                                        focal_length=focal_length.to(device),
                                                        color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                        ).cpu().numpy()[0]
                    delta = np.abs(width - height)//2
                    if width > height:
                        img3 = img3[delta:-delta]
                    else:
                        img3 = img3[:,delta:-delta]

                    # compose with alpha blendings
                    fg_img = torch.from_numpy(img3)[None]
                    fg_masks = torch.from_numpy(img3.sum(-1) > 0)[None].float()
                    bg_img = torch.from_numpy(np.asarray(img))[None]
                    img3 = renderer.compose_foreground_on_background(fg_img.to(device), fg_masks.to(device), bg_img.to(device), alpha=1.)
                    img3 = img3.cpu().numpy()[0].astype(np.uint8)

                    # projection of j3d+transl
                    j2d_ = perspective_projection(j3ds_transl[t][None], c_x, c_y, f_x, f_y)[0]
                    j2d_norm_ = perspective_projection(j3ds_norm_transl[t][None], c_x, c_y, f_x, f_y)[0]

                    # visu
                    img1 = visu_pose2d(np.asarray(img), j2ds[t], get_mano_skeleton(), lw_line= 3, lw_dot = 3, color_dot='red')
                    img2 = visu_pose2d(np.asarray(img), j2d_, get_mano_skeleton(), lw_line= 2, lw_dot = 2, color_dot='green')
                    img4 = visu_pose2d(np.asarray(img), j2d_norm_, get_mano_skeleton(), lw_line= 2, lw_dot = 2, color_dot='green')
                    # img_ = np.concatenate([img1, img2, img4, img3])
                    # img_ = np.concatenate([img1, img2, img3], 1)
                    img_ = np.concatenate([img1, img4, img3], 1)
                    Image.fromarray(img_).save(f"{tmp_dir}/{t:06d}.jpg")

                    if debug:
                        Image.fromarray(img_).save(f"img.jpg")
                        ipdb.set_trace()

            if render:
                print("video creation...")
                os.makedirs(os.path.dirname(name), exist_ok=True)
                cmd = f"ffmpeg -hide_banner -loglevel error -framerate 30 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name} -y"
                os.system(cmd)
                os.system(f"rm {tmp_dir}/*.jpg")

            # Saving
            os.makedirs(os.path.dirname(posebert_fn), exist_ok=True)
            _dict = {
                'hand_pose': hand_pose, # [45]
                'global_orient': global_orient,
                'transl': transl,
            }
            with open(posebert_fn, 'wb') as f:
                pickle.dump(_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # metrics
    all_MPJPE_posebert, all_MPJPE_PA_posebert, all_pck_posebert = [], [], []
    j3ds = torch.from_numpy(np.concatenate(l_jts_hat))
    j3ds_ = j3ds.clone() # convert_jts(j3ds.clone(), 'mano', 'dope_hand')
    j3ds_ = j3ds_ - j3ds_[:,[0]]
    for t in range(j3ds.shape[0]):
        if masks[t] == 1:
            MPJPE, MPJPE_PA, errors_pck, _ = compute_errors(j3ds_[t].numpy().reshape(-1, 21, 3), jts3d[t].reshape(-1, 21, 3))
            all_MPJPE_posebert.append(MPJPE)
            all_MPJPE_PA_posebert.append(MPJPE_PA)
            all_pck_posebert.append(errors_pck.reshape(21, 1))
        else:
            all_pck_posebert.append(np.ones((21, 1)))
    if True:
        print(f"*** POSEBERT ***")
        print(f"MPJPE eval on {len(all_MPJPE_posebert)} frames: {1000. * np.mean(all_MPJPE_posebert):2f} m")    
        print(f"MPJPE_PA eval on {len(all_MPJPE_posebert)}: {1000. * np.mean(all_MPJPE_PA_posebert):2f} m")
        dope_pck_dict = dict()
        for pck_th in np.arange(0.01, 0.10001, 0.01):
            pck_final = compute_pck(all_pck_posebert, pck_th) * 100.
            dope_pck_dict[f'{pck_th:.4f}'] = pck_final
            print(f"PCK with thresh {pck_th:.4f}: {pck_final:.4f} %")

    # relative pose of dope
    transls = estimate_translation(j3ds_dope, j2ds_dope, f_x, f_y, c_x, c_y)
    j3ds_transl_dope = j3ds_dope + transls.unsqueeze(1)
    T_dope, _ = compute_relp_cam_from_jts3d(j3ds_transl_dope.numpy())
    rot_err_dope = np.asarray([(geodesic_distance_for_rotations(T_dope[t, :3, :3], T[t, :3, :3]) * (180 / np.pi)) for t in range(1, len(T))])
    transl_err_dope = np.linalg.norm((T_dope[:, :3, 3] - T[:, :3, 3]), axis=1)
    print(f"**DOPE**     Rel-Pose {seqname} - Rot: mean={rot_err_dope.mean():.4f} , median={np.median(rot_err_dope):.4f} - Transl: mean={transl_err_dope.mean():.4f} , median={np.median(transl_err_dope):.4f}")

    # relative pose of posebert
    j3ds_hat = torch.from_numpy(np.concatenate(l_jts_hat))
    T_hat, _ = compute_relp_cam_from_jts3d(j3ds_hat.numpy())
    rot_err = np.asarray([(geodesic_distance_for_rotations(T_hat[t, :3, :3], T[t, :3, :3]) * (180 / np.pi)) for t in range(1, len(T))])
    transl_err = np.linalg.norm((T_hat[:, :3, 3] - T[:, :3, 3]), axis=1)
    print(f"**PoseBERT** Rel-Pose {seqname} - Rot: mean={rot_err.mean():.4f} , median={np.median(rot_err):.4f} - Transl: mean={transl_err.mean():.4f} , median={np.median(transl_err):.4f}")
    # ipdb.set_trace()

    # Load PoseBERT outputs
    with open(posebert_fn, 'rb') as f:
        _dict = pickle.load(f)
    hand_pose = _dict['hand_pose']
    global_orient = _dict['global_orient']
    transl = _dict['transl']

    # Not useful for the moment...
    do_final_optim = False
    if do_final_optim:
        # Final optimization
        tdim = global_orient.shape[0]
        betas = bm.betas.repeat(tdim, 1)
        hand_pose_final = hand_pose.reshape(-1, 45).repeat(tdim, 1)
        global_orient_final = roma.rotvec_to_rotmat(global_orient.clone())
        global_orient_final = nn.Parameter(global_orient_final[:,:,:2])
        transl_final = nn.Parameter(transl.clone())
        j2d = j2ds[:transl_final.shape[0]].to(device) # target
        valid = masks.to(device) # [tdim]

        # Update valid such that if the j2d of DOPE are too far from the j2d from PoseBERT we do not take this j2d as target
        with torch.no_grad():
            out = bm(global_orient=roma.rotmat_to_rotvec(roma.special_gramschmidt(global_orient_final)).to(device), transl=transl_final.to(device),hand_pose=hand_pose_final.to(device), betas=betas.to(device))
            j3d_hat = update_mano_joints(out.joints, out.vertices)
            j2d_hat = perspective_projection(j3d_hat, c_x, c_y, f_x, f_y) # [tdim,21,2]

        # valid based on iou j2d-based
        bbox_hat = get_bbox(j2d_hat)
        bbox = get_bbox(j2d)
        ious = torch.Tensor([iou2d(bbox.cpu().numpy()[[k]], bbox_hat.cpu().numpy()[[k]]) for k in range(bbox.shape[0])]).float()
        thresh = 0.1
        valid_ = (ious > thresh).float()
        valid = (valid * valid_.to(device))

        lr = 0.001
        optimizer = torch.optim.Adam([global_orient_final, transl_final], lr=lr)
        iter = 1000
        for i in tqdm(range(iter), desc='Final optimization'):
            # print(global_orient_final.shape)
            out = bm(global_orient=roma.rotmat_to_rotvec(roma.special_gramschmidt(global_orient_final)).to(device), transl=transl_final.to(device),hand_pose=hand_pose_final.to(device), betas=betas.to(device))
            j3d_hat = update_mano_joints(out.joints, out.vertices)
            j2d_hat = perspective_projection(j3d_hat, c_x, c_y, f_x, f_y)

            # loss on projection
            loss_proj = ((j2d_hat - j2d)**2).sum(-1) +  ((j2d_hat - j2d).abs()).sum(-1)
            loss_proj = loss_proj.sum(-1)
            loss_proj = (loss_proj * valid).sum() / (valid.sum() + 1e-4)
            loss_proj = 0.0000001 * loss_proj

            # loss velocity of 3d joints
            loss_vel = ((j3d_hat[1:] - j3d_hat[:-1])**2).sum(-1) + ((j3d_hat[1:] - j3d_hat[:-1]).abs()).sum(-1)
            loss_vel = loss_vel.sum(-1).mean()
            loss_vel = 1. * loss_vel

            # loss velocity of translation
            loss_transl = ((transl_final[1:] - transl_final[:-1])**2).sum(-1) + ((transl_final[1:] - transl_final[:-1]).abs()).sum(-1)
            loss_transl = loss_transl.sum(-1).mean()
            loss_transl = 1. * loss_transl

            # loss on velocity of global orientation
            loss_global_orientation = (roma.special_gramschmidt(global_orient_final[1:]) - roma.special_gramschmidt(global_orient_final[:-1])).sum([1,2])
            loss_global_orientation = loss_global_orientation.mean()
            loss_global_orientation = 1. * loss_global_orientation

            # total loss
            # loss = loss_proj + loss_vel + loss_transl + loss_global_orientation
            loss = loss_proj + loss_vel

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(f"i={i} - loss={loss.item()} - loss_pro={loss_proj.item()} - loss_vel={loss_vel.item()}")

                # visu
                t_ = tdim//2
                img = Image.open(img_paths[t_])
                img = visu_pose2d(np.asarray(img), j2d_hat[t_], get_mano_skeleton(), lw_line= 3, lw_dot = 3, color_dot='red')
                Image.fromarray(img).save('img.jpg')

        # Final visu
        with torch.no_grad():
            for t in tqdm(range(tdim)):
                # image
                img = Image.open(img_paths[t])

                # rendering
                verts_hat = bm(global_orient=roma.rotmat_to_rotvec(roma.special_gramschmidt(global_orient_final[[t]])).to(device), hand_pose=hand_pose_final[[t]].to(device), transl=transl_final[[t]].to(device)).vertices[0]
                img3 = renderer.renderPerspective(vertices=[verts_hat], 
                                                faces=[faces],
                                                rotation=rotation,
                                                camera_translation=cam,
                                                focal_length=2*f_x/img_res,
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                ).cpu().numpy()[0]
                delta = np.abs(width - height)//2
                if width > height:
                    img3 = img3[delta:-delta]
                else:
                    img3 = img3[:,delta:-delta]

                # compose with alpha blendings
                fg_img = torch.from_numpy(img3)[None]
                fg_masks = torch.from_numpy(img3.sum(-1) > 0)[None].float()
                bg_img = torch.from_numpy(np.asarray(img))[None]
                img3 = renderer.compose_foreground_on_background(fg_img.to(device), fg_masks.to(device), bg_img.to(device), alpha=1.)
                img3 = img3.cpu().numpy()[0].astype(np.uint8)

                Image.fromarray(img3).save(f"{tmp_dir}/{t:06d}.jpg")

            name = '.'.join(name.split('.')[:-1]) + '_finalOptim.mp4'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            cmd = f"ffmpeg -hide_banner -loglevel error -framerate 30 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name} -y"
            os.system(cmd)
            os.system(f"rm {tmp_dir}/*.jpg")

    del renderer
    del bm

    return all_MPJPE, all_MPJPE_PA, all_pck, all_MPJPE_posebert, all_MPJPE_PA_posebert, all_pck_posebert

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

def rotran2homat(R, T):
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = R
    hom_mat[:3, 3] = T

    return hom_mat

def compute_similarity_transform_updated(S1, S2, return_transfom=False, mu_idx=None, scale=None):
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

def compute_relp_cam_from_jts3d(all_jts3d_cam, rel_type='REF'):
    if rel_type == 'CONSEQ':
        raise NotImplementedError
        print("CONSEQ-REL pose computation...")
        all_jts3d_prcrst_algnd = all_jts3d_cam[0].reshape(1, 21, 3)
        all_jts3d_prcrst_pose_rel = np.eye(4).reshape(1, 4, 4)
        for idx in tqdm(range(1, len(all_jts3d_cam))):
            pts_trnsfmd, trnsfm = compute_similarity_transform(
                all_jts3d_cam[idx-1], all_jts3d_cam[idx], return_transfom=True, mu_idx=None, scale=1
                )
            all_jts3d_prcrst_algnd = np.append(all_jts3d_prcrst_algnd, pts_trnsfmd.reshape(1, 21, 3), axis=0)
            all_jts3d_prcrst_pose_rel =  np.append(all_jts3d_prcrst_pose_rel, rotran2homat(trnsfm['rot'],
             trnsfm['tran'].flatten()).reshape(1, 4, 4), axis=0)
    elif rel_type == 'REF':
        print("REF-REL pose computation...")
        all_jts3d_prcrst_algnd = all_jts3d_cam[0].reshape(1, 21, 3)
        all_jts3d_prcrst_pose_rel = np.eye(4).reshape(1, 4, 4)
        for idx in tqdm(range(1, len(all_jts3d_cam))):
            pts_trnsfmd, trnsfm = compute_similarity_transform_updated(
                all_jts3d_cam[0], all_jts3d_cam[idx], return_transfom=True, mu_idx=None, scale=1
                )
            all_jts3d_prcrst_algnd = np.append(all_jts3d_prcrst_algnd, pts_trnsfmd.reshape(1, 21, 3), axis=0)
            all_jts3d_prcrst_pose_rel =  np.append(all_jts3d_prcrst_pose_rel, rotran2homat(trnsfm['rot'],
             trnsfm['tran'].flatten()).reshape(1, 4, 4), axis=0)
    else:
        raise ValueError(f"wrong {rel_type} value")

    return all_jts3d_prcrst_pose_rel, all_jts3d_prcrst_algnd

def safe_arccos(x):
    """
        Returns the arcosine of x, clipped between -1 and 1.
        Use this when you know x is a cosine, but it might be
        slightly over 1 or below -1 due to numerical errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))

def axis_angle_from_rotation(R):
    """
        Returns the *(axis,angle)* representation of a given rotation.
        There are a couple of symmetries:
        * By convention, the angle returned is nonnegative.
        * If the angle is 0, any axis will do.
          In that case, :py:func:`default_axis` will be returned.
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


def geodesic_distance_for_rotations(R1, R2):
    """
        Returns the geodesic distance between two rotation matrices.
        It is computed as the angle of the rotation :math:`R_1^{*} R_2^{-1}``.
    """
    R = np.dot(R1, R2.T)
    axis1, angle1 = axis_angle_from_rotation(R)  # @UnusedVariable
    # print(axis1)
    return angle1


def update_jts_with_shape(j3ds, masks, jts3d, traversal, parents, mean, shape):
    # gt shape
    if shape == 'gt':
        print('gt shape')
        _, bone_lengths = normalize_skeleton_by_bone_length_updated(torch.from_numpy(jts3d), torch.from_numpy(jts3d), traversal, parents, return_bone_lengths=True, required_bone_lengths=None)
    elif shape == 'dope_median':
        print('dope_median shape')
        # shape from DOPE - median on the temporal window
        idx = torch.where(masks)[0]
        _, bone_lengths = normalize_skeleton_by_bone_length_updated(j3ds[idx], j3ds[idx], traversal, parents, return_bone_lengths=True, required_bone_lengths=None)
    elif shape == 'mano':
        # shape from mano
        _, bone_lengths = normalize_skeleton_by_bone_length_updated(torch.from_numpy(mean).unsqueeze(0), torch.from_numpy(mean).unsqueeze(0), traversal, parents, return_bone_lengths=True, required_bone_lengths=None)
    else:
        return j3ds

    bone_lengths = torch.median(bone_lengths, 0).values
    bone_lengths = bone_lengths.unsqueeze(0).repeat(j3ds.shape[0], 1)
    j3ds = normalize_skeleton_by_bone_length_updated(j3ds, j3ds, traversal, parents, return_bone_lengths=False, required_bone_lengths=bone_lengths)
    return j3ds

@torch.no_grad()
def get_dope_metrics(shape='dope', debug=0, model=None, bm=None, method='dope', refine_posebert=False, pnp='pnp'):
    
    assert shape in ['dope', 'gt', 'dope_median', 'mano']

    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
    faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32))
    traversal, parents = get_mano_traversal()
    mean = update_mano_joints(bm().joints.detach(), bm().vertices.detach())[0].cpu().numpy()

    f_x= 899.783
    f_y=900.019
    c_x= 653.768
    c_y=362.143

    all_seqs_tar_pths = glob.glob(f"{ANIL_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(os.path.basename(spth.split('.')[0]))
    all_sqns.sort()
    l_MPJPE, l_MPJPE_PA = [], []
    all_pck = []
    rel_pose = {}
    # method = 'pnp'
    # method_ = 'kabsch'
    method_ = pnp
    T_dir = f"/scratch/1/user/fbaradel/ROAR/relative_camera_poses/{method}_{shape}_{method_}_refinePoseBERT{refine_posebert}_updated"
    os.makedirs(T_dir, exist_ok=True)
    for i, seqname in enumerate(tqdm(all_sqns)):
        # seqname = '20220909114359'
        # seqname = '20220909114359'
        # seqname = '20220705173214'
        # seqname = '20220905140306'
        # seqname = '20220805164755'
        # seqname = '20220805164755'
        print(seqname)

        # GT
        # print('loading GT')
        gt_dir = os.path.join(ANIL_DIR, seqname, 'jts3d_disp')
        jts3d_fns = os.listdir(gt_dir)
        jts3d_fns.sort()
        gt_proj_dir = os.path.join(ANIL_DIR, seqname, 'proj_jts_disp')
        jts2d_fns = [x for x in os.listdir(gt_proj_dir) if x[-4:] == '.txt']
        jts2d_fns.sort()
        # jts_fns = list(set(jts2d_fns) & set(jts3d_fns))
        # print(len(jts3d_fns), len(jts2d_fns))
        jts3d = []
        jts2d = []
        for t in range(len(jts3d_fns)):
            # j3d
            y = np.loadtxt(os.path.join(gt_dir, jts3d_fns[t]))[ours2dope()]
            jts3d.append(y.copy())
            try:
                # j2d
                y = np.loadtxt(os.path.join(gt_proj_dir, jts2d_fns[t]))[ours2dope()]
                jts2d.append(y.copy())
            except:
                # some files are missing...
                jts2d.append(jts2d[-1].copy())
        jts3d = np.stack(jts3d)
        jts3d = convert_jts(jts3d, 'dope_hand', 'mano')
        # jts3d = jts3d - jts3d[:,[0]]
        jts2d = np.stack(jts2d)
        jts2d = convert_jts(jts2d, 'dope_hand', 'mano')

        # DOPE
        # print('loading DOPE')
        j2ds, j3ds, masks, *_,  jts3d = load_dope(seqname=seqname, t_start=0, t_max=10000, take_left_hands_into_account=False,
                        anil_loading=False,
                        salma_loading=True, 
                        fn='postprocessed_fix_righthands_postpro_dope.pkl'
                        )
        j3ds = j3ds - j3ds[:, [0]]

        # same shape for the entire sequence
        if shape in ['gt', 'dope_median', 'mano']:
            j3ds = update_jts_with_shape(j3ds, masks, jts3d, traversal, parents, mean, shape)

        if 'dope_median_filtering' in method:
            l_j3ds = []
            l_j2ds = []
            l_masks = []
            window = 5
            for t in range(j3ds.shape[0]):
                if masks[t] == 0:
                    l_j3ds.append(j3ds[t])
                    l_masks.append(masks[t])
                    l_j2ds.append(j2ds[t])
                else:
                    # DO same for j2d???
                    start = np.clip(t-window, 0, j3ds.shape[0])
                    end = np.clip(t+window, 0, j3ds.shape[0])
                    start_end = [x for x in range(start,end) if masks[x] == 1]
                    j3ds_t = np.median(j3ds[start_end],0)
                    l_j3ds.append(j3ds_t)
                    mask_t = np.median(masks[start_end],0)
                    l_masks.append(mask_t)
                    j2ds_t = np.median(j2ds[start_end],0)
                    l_j2ds.append(j2ds_t)
            j3ds = torch.from_numpy(np.stack(l_j3ds))
            masks = (torch.from_numpy(np.stack(l_masks)) == 1).float()

            if '2d' in method:
                j2ds = torch.from_numpy(np.stack(l_j2ds))

        # # Not sure it is useful
        # j3ds = torch.from_numpy(np.nan_to_num(j3ds.numpy()))
        # j2ds = j2ds.numpy()
        # j2ds = np.nan_to_num(j2ds)
        # j2ds = torch.from_numpy(j2ds)

        # Relative-pose from the first good pose
        n_iter = 0
        j2ds_up = j2ds.clone()
        for k in range(1+n_iter):
            if k > 0:
                print('using projected j2ds')
                j2ds = j2ds_up.clone()

            if method_ == 'kabsch':
                transls = estimate_translation(j3ds, j2ds, f_x, f_y, c_x, c_y)
                j3ds_transl = j3ds + transls.unsqueeze(1)
            elif method_ == 'pnp':
                all_r2c_trnsfms, all_jts3d_cam = compute_root2cam_ops(j2ds.numpy(), j3ds.numpy())
                j3ds_transl = torch.from_numpy(all_jts3d_cam)

            j2ds_up = perspective_projection(j3ds_transl.float(), c_x, c_y, f_x, f_y)

            # median filtering on the j3d??

        # PoseBERT
        if 'posebert' in method:
        # if False:
            print('posebert')
            bm = bm.to(device)
            # rescale to mano bone length because it has been training like that
            j3ds_transl_norm = normalize_skeleton_by_bone_length_updated(j3ds_transl.clone(), torch.from_numpy(mean).reshape(1,21,3).repeat(j3ds_transl.shape[0], 1, 1), traversal, parents)
            j3ds_transl_norm = torch.from_numpy(np.nan_to_num(j3ds_transl_norm.numpy()))
            j2ds = j2ds.numpy()
            j2ds = np.nan_to_num(j2ds)
            j2ds = torch.from_numpy(j2ds)

            global_orient_hat, transl_hat, hand_pose_hat = model.forward_full_sequence(j3d=j3ds_transl_norm.unsqueeze(0).to(device).float(), 
                                                                                        j2d=j2ds.unsqueeze(0).to(device).float(), 
                                                                                        mask=masks.unsqueeze(0).to(device).bool())
            global_orient = roma.rotmat_to_rotvec(global_orient_hat).cpu()[0]
            hand_pose = roma.rotmat_to_rotvec(hand_pose_hat).flatten(1).cpu()[0]
            transl = transl_hat.cpu()[0]
            l_j3ds_transl = []
            for t in range(global_orient.shape[0]):
                out_hat = bm(
                    global_orient=global_orient[[t]].to(device), 
                    hand_pose=hand_pose.unsqueeze(0).to(device),
                    transl=transl[[t]].to(device),
                    betas=bm.betas.to(device),
                    )
                jts_hat = update_mano_joints(out_hat.joints, out_hat.vertices)
                l_j3ds_transl.append(jts_hat[0].cpu())
            j3ds_transl_ = torch.stack(l_j3ds_transl)
            print('dope-posebert diff', (j3ds_transl - j3ds_transl_).abs().sum(-1).mean().item())
            j3ds_transl = j3ds_transl_.clone()

            # if True:
            if 'shape' in method:
                print('updating posebert output according to a specific shape')
                # back to a good shape
                assert shape != 'dope'
                j3ds_transl = update_jts_with_shape(j3ds_transl, masks, jts3d, traversal, parents, mean, shape)

            if refine_posebert:
                j3ds = j3ds_transl.clone() - j3ds_transl.clone()[:, [0]]
                if method_ == 'kabsch':
                    transls = estimate_translation(j3ds, j2ds, f_x, f_y, c_x, c_y)
                    j3ds_transl = j3ds + transls.unsqueeze(1)
                elif method_ == 'pnp':
                    all_r2c_trnsfms, all_jts3d_cam = compute_root2cam_ops(j2ds.numpy(), j3ds.numpy())
                    j3ds_transl = torch.from_numpy(all_jts3d_cam)
        else:
            print('dope only')

        # TODO compute on all the timestep for posebert???
        if False:
        # if 'posebert' in method:
            print('keep all timesteps')
            idx = torch.Tensor(range(masks.shape[0])).int()
        else:
            idx = torch.where(masks)[0]
        # if masks.mean() < 1:
        #     ipdb.set_trace()
        #     masks[20]
        #     j3ds_transl[20]

        T_hat, _ = compute_relp_cam_from_jts3d(j3ds_transl.numpy()[idx])
        T, _ = compute_relp_cam_from_jts3d(jts3d[idx])
        rot_err = np.asarray([(geodesic_distance_for_rotations(T_hat[t, :3, :3], T[t, :3, :3]) * (180 / np.pi)) for t in range(0, len(T))])
        transl_err = np.linalg.norm((T_hat[:, :3, 3] - T[:, :3, 3]), axis=1)
        # print(f"Rel-Pose {seqname} - Rot: mean={rot_err.mean():.4f} , median={np.median(rot_err):.4f} - Transl: mean={transl_err.mean():.4f} , median={np.median(transl_err):.4f}")
        
        rel_pose[seqname] = {
            'rre_mean': rot_err.mean(),
            'rre_med': np.median(rot_err),
            'rte_mean': transl_err.mean(),
            'rte_med': np.median(transl_err),
        }
        for t in range(j3ds.shape[0]):
            if masks[t].item() == 1:
                MPJPE, MPJPE_PA, errors_pck, _ = compute_errors(j3ds_transl[t].numpy().reshape(-1, 21, 3), jts3d[t].reshape(-1, 21, 3)) # 
                l_MPJPE.append(MPJPE)
                l_MPJPE_PA.append(MPJPE_PA)
                all_pck.append(errors_pck.reshape(21, 1))

                # # visu
                # img1 = visu_pose3d(jts3d[t].reshape(1, 21, 3).copy(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
                # img2 = visu_pose3d(j3ds[t].reshape(1, 21, 3).clone(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
                # Image.fromarray(np.concatenate([img1, img2], 1)).save('img.jpg')
                # ipdb.set_trace()
            else:
                if 'posebert' in method:
                    _, _, errors_pck, _ = compute_errors(j3ds_transl[t].numpy().reshape(-1, 21, 3), jts3d[t].reshape(-1, 21, 3)) # 
                    all_pck.append(errors_pck.reshape(21, 1))
                else:
                    all_pck.append(np.ones((21, 1)))

        print(f"MPJPE eval on {len(l_MPJPE)} frames: {1000. * np.mean(l_MPJPE):.2f} m")    
        print(f"MPJPE_PA eval on {len(l_MPJPE_PA)}: {1000. * np.mean(l_MPJPE_PA):.2f} m")   
        dope_pck_dict = dict()
        # for pck_th in np.arange(0.001, 0.101, 0.001):
        # for pck_th in np.arange(0.01, 0.10001, 0.01):
        for pck_th in [0.03]:
            pck_final = compute_pck(np.stack(all_pck), pck_th) * 100.
            dope_pck_dict[f'{pck_th:.4f}'] = pck_final
            print(f"PCK with thresh {pck_th:.4f}: {pck_final:.2f} %")
        
        rre_mean, rre_med, rte_mean, rte_med = [], [], [], []
        for key, val in rel_pose.items():
            rre_mean.append(val['rre_mean'])
            rre_med.append(val['rre_med'])
            rte_mean.append(val['rte_mean'])
            rte_med.append(val['rte_med'])
        print(f"Rel-Pose - Rot: mean={np.asarray(rre_mean).mean():.4f} , median={np.asarray(rre_med).mean():.4f} - Transl: mean={np.asarray(rte_mean).mean():.4f} , median={np.asarray(rte_med).mean():.4f}")
        sys.stdout.flush()


        # Save camera params
        proj_mat_dir = os.path.join(T_dir, seqname, 'proj_mat')
        cam_pose_dir = os.path.join(T_dir, seqname, 'cam_pose')
        error_dir =  os.path.join(T_dir, seqname, 'cam_pose_error')
        j3d_mvs_dir =  os.path.join(T_dir, seqname, 'j3d_mvs')
        j3d_cam_dir =  os.path.join(T_dir, seqname, 'j3d_cam')
        j3d_mvs_gt_dir =  os.path.join(T_dir, seqname, 'j3d_mvs_gt')
        j3d_cam_gt_dir =  os.path.join(T_dir, seqname, 'j3d_cam_gt')
        os.makedirs(proj_mat_dir, exist_ok=True)
        os.makedirs(cam_pose_dir, exist_ok=True)
        os.makedirs(error_dir, exist_ok=True)
        os.makedirs(j3d_mvs_dir, exist_ok=True)
        os.makedirs(j3d_cam_dir, exist_ok=True)
        os.makedirs(j3d_mvs_gt_dir, exist_ok=True)
        os.makedirs(j3d_cam_gt_dir, exist_ok=True)
        img_fns = os.listdir(os.path.join(ANIL_DIR, seqname, 'rgb'))
        img_fns.sort()
        error = np.stack([rot_err, transl_err], 1)
        np.savetxt(os.path.join(T_dir, seqname, 'cam_pose_error_mean.txt'), error.mean(0))
        np.savetxt(os.path.join(T_dir, seqname, 'cam_pose_error_median.txt'), np.median(error,0))

        # 3d bbox
        # back to T=0
        T_hat_ = np.linalg.inv(T_hat)
        j3d_cam = (torch.from_numpy(T_hat_).reshape(-1,1,4,4).float() @ torch.cat([j3ds_transl[idx], torch.ones_like(j3ds_transl[idx][...,:1])],-1).reshape(-1,21,4,1).float())[:,:,:3,0]
        T_ =  np.linalg.inv(T)
        j3d_cam_gt = (torch.from_numpy(T_).reshape(-1,1,4,4).float() @ torch.cat([torch.from_numpy(jts3d)[idx], torch.ones_like(j3ds_transl[idx][...,:1])],-1).reshape(-1,21,4,1).float())[:,:,:3,0]

        l_img = []
        for t in range(T_hat.shape[0]):
            t_ = idx[t].item()
            l_img.append(img_fns[t_])

            # cam_pose
            np.savetxt(os.path.join(cam_pose_dir, img_fns[t_].replace('.png', '.txt')), T_hat[t])

            # proj_mat
            P = CAM_INTR @ T_hat[t][:3] # [3,4]  # proj_mat
            np.savetxt(os.path.join(proj_mat_dir, img_fns[t_].replace('.png', '.txt')), P)

            # error
            np.savetxt(os.path.join(error_dir, img_fns[t_].replace('.png', '.txt')), error[t])

            # j3d
            # ipdb.set_trace()
            np.savetxt(os.path.join(j3d_mvs_dir, img_fns[t_].replace('.png', '.txt')), j3d_cam[t].numpy())
            np.savetxt(os.path.join(j3d_cam_dir, img_fns[t_].replace('.png', '.txt')), j3ds_transl[idx][t].numpy())
            np.savetxt(os.path.join(j3d_mvs_gt_dir, img_fns[t_].replace('.png', '.txt')), j3d_cam_gt[t].numpy())
            np.savetxt(os.path.join(j3d_cam_gt_dir, img_fns[t_].replace('.png', '.txt')), jts3d[idx][t])

        # t = 0
        # T_ =  np.linalg.inv(T)
        # j3d_cam_gt = (torch.from_numpy(T_).reshape(-1,1,4,4).float() @ torch.cat([torch.from_numpy(jts3d)[idx], torch.ones_like(j3ds_transl[idx][...,:1])],-1).reshape(-1,21,4,1).float())[:,:,:3,0]
        # img1 = visu_pose3d((j3d_cam_gt[[t]] - j3d_cam_gt[[t], [0]]).clone(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
        # img1_ = visu_pose3d((j3d_cam[[t]] - j3d_cam[[t], [0]]).clone(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
        # t = masks.shape[0] // 2
        # img2 = visu_pose3d((j3d_cam_gt[[t]] - j3d_cam_gt[[t], [0]]).clone(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
        # img2_ = visu_pose3d((j3d_cam[[t]] - j3d_cam[[t], [0]]).clone(), res=400, bones=get_mano_skeleton(), factor=6., lw_line=1., lw_dot=0.)
        # Image.fromarray(np.concatenate([img1, img2, img1_, img2_], 1)).save('img.jpg')
        # ipdb.set_trace()

        def get_3d_bbox(j3d, factor=1.):
            """
            Args:
            - j3d: np.array - [tdim,k,3]
            """
            x_min, x_max = np.min(j3d[...,0]), np.max(j3d[...,0])
            y_min, y_max = np.min(j3d[...,1]), np.max(j3d[...,1])
            z_min, z_max = np.min(j3d[...,2]), np.max(j3d[...,2])
            xc, yc, zc = (x_max + x_min) / 2., (y_max + y_min) / 2., (z_max + z_min) / 2.
            xs, ys, zs = factor * np.abs(x_max - x_min), factor * np.abs(y_max - y_min), factor * np.abs(z_max - z_min)
            x1, x2 = xc - xs/2., xc + xs/2.
            y1, y2 = yc - ys/2., yc + ys/2.
            z1, z2 = zc - zs/2., zc + zs/2.
            return (x1,y1,z1,x2,y2,z2)
        bbox = get_3d_bbox(j3d_cam.numpy(), factor=1.)
        np.savetxt(os.path.join(T_dir, seqname, '3d_bbox_x1y1z1x2y2z2.txt'), np.asarray(bbox))
        # bbox_gt = get_3d_bbox(j3d_cam_gt.numpy(), factor=1.)
        # bbox_gt_ = get_3d_bbox(j3d_cam_gt.numpy()[:1], factor=1.)

        
        # sort the frame by the errors
        idx_sort = np.argsort(rot_err)
        l_img = np.asarray(l_img)[idx_sort].tolist()
        error = error[idx_sort]
        np.savetxt(os.path.join(T_dir, seqname, 'sorted_cam_pose_error.txt'), error)
        np.savetxt(os.path.join(T_dir, seqname, 'sorted_img_names.txt'), np.asarray(l_img), fmt='%s')
        np.savetxt(os.path.join(T_dir, seqname, 'sorted_idx.txt'), idx_sort)

        if debug and i == 3:
            os._exit(0)
        
CAM_INTR = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
   )

def cv2_PNPSolver(obj_pts_lst, img_pts_lst, intrinsics, dist_coeffs=np.zeros(4), method=None):
    """
    # Minimize the reprojection error, default method flag set to SOLVEPNP_ITERATIVE
    @param obj_pts_lst: 3d object points
    @param img_pts_lst: 2d objec points on image
    @param intrinsics: camera intrinsic matrix
    @param dist_coeffs: distortioni coeffs, deafult is zero
    @return: returns the rotation and the translation vectors that transform a 3D point
             expressed in the object/3d points' coordinate frame to the camera coordinate frame
    """
    r2cs = []
    rvec = np.zeros(3, dtype=np.float)
    tvec = np.array([0, 0, 1], dtype=np.float)
    for obj_pts, img_pts in zip(obj_pts_lst, img_pts_lst):
        if method is None or method == 'PNP_ITERATIVE':
            _, rvec, tvec = cv2.solvePnP(objectPoints=obj_pts, imagePoints=img_pts,
                                         cameraMatrix=intrinsics, distCoeffs=dist_coeffs,
                                         rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE
                                         )           
        elif method == 'PNP_SQPNP':
            rvec = np.zeros(3, dtype=np.float)
            tvec = np.array([0, 0, 1], dtype=np.float)
            _, rvec, tvec = cv2.solvePnP(objectPoints=obj_pts, imagePoints=img_pts,
                                         cameraMatrix=intrinsics, distCoeffs=dist_coeffs,
                                         rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE
                                         )
        r2cs.append(rtvec2rtmat(rvec, tvec))

        if np.isnan(rvec.sum()):
            rvec = np.zeros(3, dtype=np.float)
            tvec = np.array([0, 0, 1], dtype=np.float)
    r2cs = np.array(r2cs)

    return r2cs

def rtvec2rtmat(rvec, tvec):
    rtmat = np.eye(4)
    rtmat[:3, :3] = cv2.Rodrigues(rvec)[0]
    rtmat[:3, 3] = tvec.flatten()

    return rtmat[:3, :]

def compute_root2cam_ops(all_jts2d, all_jts3d):
    # pnp minimization
    all_r2c_trnsfms = cv2_PNPSolver(all_jts3d, all_jts2d, CAM_INTR)

    # projtd_jts2d = []
    # for r2c, j3d in zip(all_r2c_trnsfms, all_jts3d):
    #     proj_mat = CAM_INTR @ r2c
    #     projtd_jts2d.append(project(P=proj_mat, X=j3d))
    # print(f"Reproj Err: {np.mean(np.linalg.norm(all_jts2d - projtd_jts2d, axis=2), 1).mean()}")
    
    # transform 3d points to cam transform using all_r2c_trnsfms
    all_jts3d_cam = np.array(
        [trnsfm_points(pts=j3d, trnsfm=r2c) for (j3d, r2c) in zip(all_jts3d, all_r2c_trnsfms)]
        )
    
    return all_r2c_trnsfms, all_jts3d_cam

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

def order_sequence(dirname):
    print(dirname)
    seqnames = os.listdir(dirname)
    seqnames.sort()
    # seqnames = seqnames[:1] # TODO comment

    errors = []
    errors_ = []
    all_erros = []
    for seqname in tqdm(seqnames):
        sys.stdout.flush()
        error = np.loadtxt(os.path.join(dirname, seqname, 'cam_pose_error_mean.txt'))
        error_ = np.loadtxt(os.path.join(dirname, seqname, 'cam_pose_error_median.txt'))
        errors.append(error)
        errors_.append(error_)

        fns = os.listdir(os.path.join(dirname, seqname, 'cam_pose_error'))
        all_err = []
        for fn in fns:
            err = np.loadtxt(os.path.join(dirname, seqname, 'cam_pose_error', fn))
            all_err.append(err)
        all_erros.append(np.stack(all_err))
    
    errors = np.stack(errors)
    errors_ = np.stack(errors_)

    thresh = 5
    for i in [0,1]:
        if i == 0:
            print("\nNot sorted:")
            idx_sorted = range(errors.shape[0])
        else:
            print("\nSorted:")
            idx_sorted = np.argsort(errors[:,0])
        
        for idx in idx_sorted:
            sys.stdout.flush()
            # Count number of frame with rot error lowest than 5 degree
            count = (all_erros[idx][:,0] < thresh).sum()
            print(f"{seqnames[idx]} - MEAN: Rot={errors[idx,0]:.4f} , Trans={errors[idx,1]:.4f} - MEDIAN: Rot={errors_[idx,0]:.4f} , Trans={errors_[idx,1]:.4f} - Nb good frames: {count}/{len(all_erros[idx])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training PoseBERT')
    parser.add_argument('--dirname', type=str, default=ANIL_DIR)
    parser.add_argument("--seqname", type=str, default='20220705173214')
    parser.add_argument("--ckpt", type=str, default='/scratch/1/user/fbaradel/ROAR/logs/dae_j3dj2d_smallNoiseBis/checkpoints/last.pt')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=-1)
    parser.add_argument("--outdirname", type=str, default='res/')
    parser.add_argument("--take_left_hands_into_account", type=int, default=0, choices=[0,1])
    parser.add_argument("--debug", type=int, default=0, choices=[0,1])
    parser.add_argument("--render", type=int, default=0, choices=[0,1])
    parser.add_argument("--all_seq", type=int, default=0, choices=[0,1])
    parser.add_argument("--dope_only", type=int, default=0, choices=[0,1])
    parser.add_argument("--refine_posebert", type=int, default=0, choices=[0,1])
    parser.add_argument("--pnp", type=str, default='pnp', choices=['pnp', 'kabsch'])
    parser.add_argument("--shape", type=str, default='dope', choices=['gt', 'dope_median', 'mano', 'dope'])
    parser.add_argument("--method", type=str, default='dope', choices=['dope', 'posebert', 'posebert_shape', 'dope_median_filtering',  'dope_median_filtering_2d'])
    # re-order the result
    parser.add_argument("--order_sequence", type=int, default=0, choices=[0,1])
    parser.add_argument("--res_dir", type=str, default='')
    args = parser.parse_args()

    if args.order_sequence == 1:
        order_sequence(args.res_dir)
        os._exit(0)
    
    model = PoseBERT()

    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # MANO model
    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True).to(device)
    faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32)).to(device)

    if args.dope_only == 1:
        get_dope_metrics(args.shape, args.debug, bm=bm, model=model, method=args.method, refine_posebert=args.refine_posebert==1, pnp=args.pnp)
        os._exit(0)

    outdirname = os.path.join(args.outdirname, args.seqname)
    if args.take_left_hands_into_account:
        outdirname = outdirname + '_transformleftHand'

    if args.all_seq:
        # select all the sids with .tar
        import glob
        all_seqs_tar_pths = glob.glob(f"{ANIL_DIR}/*.tar")
        all_sqns = []
        for spth in all_seqs_tar_pths:
            if os.path.isfile(spth):
                all_sqns.append(os.path.basename(spth.split('.')[0]))
        all_sqns.sort()
        l_all_MPJPE, l_all_MPJPE_PA, l_all_MPJPE_posebert, l_all_MPJPE_PA_posebert = [], [], [], []
        l_all_pck_posebert, l_all_pck = [], []
        for seqname in tqdm(all_sqns):
                # seqname = '20220902110304'
                all_MPJPE, all_MPJPE_PA, all_pck, all_MPJPE_posebert, all_MPJPE_PA_posebert, all_pck_posebert = visu(model, seqname=seqname, debug=args.debug==1, t_start=args.start, t_max=args.seq_len, outdirname=outdirname, take_left_hands_into_account=args.take_left_hands_into_account, render=args.render, bm=bm, faces=faces)
                l_all_pck.extend(all_pck)
                l_all_pck_posebert.extend(all_pck_posebert)
                l_all_MPJPE.extend(all_MPJPE)
                l_all_MPJPE_PA.extend(all_MPJPE_PA)
                l_all_MPJPE_posebert.extend(all_MPJPE_posebert)
                l_all_MPJPE_PA_posebert.extend(all_MPJPE_PA_posebert)

                # pck
                pck_th = 0.03
                pck_final = compute_pck(np.stack(l_all_pck), pck_th) * 100.
                pck_final_posebert = compute_pck(np.stack(l_all_pck_posebert), pck_th) * 100.

                print(f"N={len(l_all_MPJPE_posebert)}")
                print(f"DOPE:     MPJPE: {1000. * np.mean(l_all_MPJPE):.2f} mm - PA-MPJPE: {1000. * np.mean(l_all_MPJPE_PA):.2f} mm - PCK@3cm: {pck_final:.2f}")
                print(f"PoseBERT: MPJPE: {1000. * np.mean(l_all_MPJPE_posebert):.2f} mm - PA-MPJPE: {1000. * np.mean(l_all_MPJPE_PA_posebert):.2f} mm - PCK@3cm: {pck_final_posebert:.2f}")
                sys.stdout.flush()
    else:    
        visu(model, seqname=args.seqname, debug=args.debug==1, t_start=args.start, t_max=args.seq_len, outdirname=outdirname, take_left_hands_into_account=args.take_left_hands_into_account, render=args.render, bm=bm, faces=faces)