from yaml import load
from evaluation.pnps import cv2_PNPSolver
import numpy as np
import os, sys, glob
from ipdb import set_trace as bb
import argparse
import pprint, pickle
from tqdm import tqdm
osp = os.path

from evaluation.pnps import *
from viz_utils import *

CAM_INTR = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
   )

def load_pkl(pkl_file):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


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

    return poses_2d, poses_3d


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


def filter_jts(all_jts, window=5):
    N = len(all_jts)

    all_jts_fltrd = np.zeros(all_jts.shape)
    for idx, jts in enumerate(all_jts):
        if (idx < window) or (idx > N - window):
            all_jts_fltrd[idx] = all_jts[idx]
        else:
            all_jts_fltrd[idx] = np.median(all_jts[idx - window : idx + window], axis=0)
        
    return np.array(all_jts_fltrd)


def plt_err_line(err_lst, save_pth=None, xlabel='Frame index', ylabel='Err magnitude', title='Each frame error plot', leg=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    plt.figure(figsize=(25, 15))
    plt.style.use('dark_background')
    plt.plot(range(len(err_lst)), err_lst, 'r-', markersize=1)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(leg, loc='upper right')
    os.makedirs(os.path.dirname(save_pth), exist_ok=True)
    plt.savefig(save_pth)

    return None

def write_o3d_pcd(file_path, pcd_o3d):
        import open3d as o3d
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        o3d.io.write_point_cloud(file_path, pcd_o3d, write_ascii=True)

        return print(f"saved: {file_path}")

def read_o3d_pcd(file_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)

        return pcd

def select_ref_frm(all_jts3d, method='mean'):
    """
    all_jts3d: (N, 21, 3)
    # 1. compute mean/median/custom criteria
    # 2. select frame that is closest as ref
    """
    ref_frm_idx = 0
    if method == 'mean':
        print('Using mean pose to select ref frame')
        jts3d_mean = np.mean(all_jts3d, axis=0)
        jts3d_close2mean_mje =  (np.linalg.norm((all_jts3d - jts3d_mean), axis=2)).mean(axis=1)
        ref_frm_idx = np.argmin(jts3d_close2mean_mje)
    elif method == 'median':
        print('Using median pose to select ref frame')
        jts3d_med = np.median(all_jts3d, axis=0)
        jts3d_close2med_mje =  (np.linalg.norm((all_jts3d - jts3d_med), axis=2)).mean(axis=1)
        ref_frm_idx = np.argmin(jts3d_close2med_mje)
    else:
        raise NotImplementedError(f"Specified{method} is not implented yet, specify correct method!!")
    
    return ref_frm_idx

if __name__ == "__main__":

    parser = argparse.ArgumentParser('compute relative poses of estimated poses and annotated poses. Then compute error')
    parser.add_argument('--seq_rgbs_dir', type=str, default=None,
                        help='sequnce rgbs dir')
    parser.add_argument('--seq_pose3d_ann_dir', type=str, required=True,
                        help='sequence 3d pose annotation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/icp_res')
    parser.add_argument('--seq_jts2d_est_dir', type=str, required=False,
                        help='sequence 2D key points  estimation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/dope_dets')
    parser.add_argument('--seq_jts3d_est_dir', type=str, required=True,
                        help='sequence 3D key points estimation dir(eg.: /scratch/1/user/aswamy/data/hand-obj/{sqn})/dope_dets')
    parser.add_argument('--bsln_method', type=str, required=True,
                        help='baseline method like DOPE, FRANKMOCAP, etc. ')

    parser.add_argument('--rel_ptype', type=str, default='CONSEQ', help='CONSEQ or REF')   

    parser.add_argument('--sqn', type=str, default=None, help='seq id')   
    

    # Flags                     
    parser.add_argument('--save', type=int, default=0, choices=[0, 1],
                        help='save(1), not_save(0)')
    parser.add_argument('--viz', type=int, default=0, choices=[0, 1],
                        help='viz(1), not_viz(0)')
    parser.add_argument('--filter', type=int, default=0, choices=[0, 1],
                        help='filter(1), not_filter(0)')
    
    args = parser.parse_args()


    save_base_dir = osp.dirname(args.seq_pose3d_ann_dir)
    
    # print the args
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(f"args: {args}")

    ## For estimations
    if args.bsln_method == 'DOPE':   
        # load all frm's 3d joints
        print('Loading dope detections...')
        all_jts2d, all_jts3d = load_dope_poses(dets_pkls_dir=args.seq_jts3d_est_dir)

        # median filtering the dope joints
        if args.filter:
            all_jts2d = filter_jts(all_jts2d, window=5)
            all_jts3d = filter_jts(all_jts3d, window=5)
        # bb()
        if args.rel_ptype == 'CONSEQ':
            ## For Annotations
            # load 3d pose annotatations (Note: ann are inversed while loading)
            all_poses3d_ann_hom = load_poses3d_ann(seq_pose3d_ann_dir=args.seq_pose3d_ann_dir)

            # compute relative pose frame on ann poses
            all_poses3d_ann_rel = np.array(
                [all_poses3d_ann_hom[idx] @ np.linalg.inv(all_poses3d_ann_hom[idx - 1])
                    for idx in range(1, len(all_poses3d_ann_hom))]
            )

            # find root to cam transforms
            all_r2c_trnsfms = cv2_PNPSolver(all_jts3d, all_jts2d, CAM_INTR)
            
            # transform 3d points to cam transform using all_r2c_trnsfms
            all_jts3d_cam = np.array([trnsfm_points(pts=j3d, trnsfm=r2c) for (j3d, r2c) in zip(all_jts3d, all_r2c_trnsfms)])

            # compute rel pose  by procrust aligment
            root_joint_idx = None
            all_jts3d_prcrst_pose_rel = []
            all_jts3d_prcrst_algnd = []
            for idx in tqdm(range(1, len(all_jts3d_cam))):
                pts_trnsfmd, trnsfm = compute_similarity_transform(all_jts3d_cam[idx], all_jts3d_cam[idx - 1], return_transfom=True, mu_idx=root_joint_idx, scale=1)
                all_jts3d_prcrst_algnd.append(pts_trnsfmd)
                all_jts3d_prcrst_pose_rel.append(rotran2homat(trnsfm['rot'], trnsfm['tran'].flatten()))
            all_jts3d_prcrst_algnd = np.array(all_jts3d_prcrst_algnd)
            all_jts3d_prcrst_pose_rel = np.array(all_jts3d_prcrst_pose_rel)

            # Procrust alignment of r2c and ann translations
            all_ann_transl = all_poses3d_ann_hom[:, :3, 3]
            all_r2c_transl = all_r2c_trnsfms[:, :3, 3]
            pts_trnsfmd, trnsfm = compute_similarity_transform(all_r2c_transl, all_ann_transl, return_transfom=True, mu_idx=None, scale=1)
            r2c_to_ann_trnsfm = rotran2homat(trnsfm['rot'], trnsfm['tran'].flatten())
            all_r2c_transl_algnd2ann = np.array(pts_trnsfmd)

            all_ann_transl_rel = np.array(
                [all_ann_transl[idx] - all_ann_transl[idx - 1] for idx in range(1, len(all_ann_transl))]
                )
            all_ann_transl_rel_dist = np.array(
                [np.linalg.norm(all_ann_transl[idx] - all_ann_transl[idx - 1]) for idx in range(1, len(all_ann_transl))]
                )
            
            all_r2c_transl_algnd2ann_rel = np.array(
                [all_r2c_transl_algnd2ann[idx] - all_r2c_transl_algnd2ann[idx - 1] for idx in range(1, len(all_r2c_transl_algnd2ann))]
                )
            all_r2c_transl_algnd2ann_rel_dist = np.array(
                [np.linalg.norm(all_r2c_transl_algnd2ann[idx] - all_r2c_transl_algnd2ann[idx - 1]) for idx in range(1, len(all_r2c_transl_algnd2ann))]
                )

            bb()
            # relative rot magnitude
            all_jts3d_prcrst_pose_rel_rot_dist =  np.array(
            [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i][:3, :3], all_jts3d_prcrst_pose_rel[i - 1][:3, :3]) * (180 / np.pi))
             for i in range(1, len(all_jts3d_prcrst_pose_rel))]
                )
            all_poses3d_ann_rel_rot_dist =  np.array(
            [(geodesic_distance_for_rotations(all_poses3d_ann_rel[i][:3, :3], all_poses3d_ann_rel[i - 1][:3, :3]) * (180 / np.pi))
             for i in range(1, len(all_poses3d_ann_rel))]
                )

            plt.figure(figsize=(25, 15))
            plt.style.use('dark_background')
            plt.plot(range(len(all_jts3d_prcrst_pose_rel_rot_dist)), all_jts3d_prcrst_pose_rel_rot_dist, 'r-', markersize=1)
            plt.plot(range(len(all_poses3d_ann_rel_rot_dist)), all_poses3d_ann_rel_rot_dist, 'g-', markersize=1)
            plt.xlabel('frameid', fontsize=20)
            plt.ylabel('ref-rel rot magnitude', fontsize=20)
            plt.title('rot magnitude error of each frame', fontsize=20)
            plt.legend(['dope', 'ann'])
            save_pth = f'./out/{args.sqn}_filter_{args.filter}_rel_rot_mag.png'
            plt.savefig(save_pth)

        elif args.rel_ptype == 'REF':
            # compute ref frame jts3D
            jts3d_ref_frm_ind = 0 #select_ref_frm(all_jts3d=all_jts3d, method='median')
            print(f"Ref. Frame Selected: {jts3d_ref_frm_ind}")

            ## For Annotations
            # load 3d pose annotatations (Note: ann are inversed while loading)
            all_poses3d_ann_hom = load_poses3d_ann(seq_pose3d_ann_dir=args.seq_pose3d_ann_dir)

            # compute relative pose frame on ann poses
            all_poses3d_ann_rel = np.array(
                [all_poses3d_ann_hom[idx] @ np.linalg.inv(all_poses3d_ann_hom[jts3d_ref_frm_ind])
                    for idx in range(1, len(all_poses3d_ann_hom))]
            )

            # find root to cam transforms
            all_r2c_trnsfms = cv2_PNPSolver(all_jts3d, all_jts2d, CAM_INTR, dist_coeffs=np.zeros(4)) 
            
            # transform 3d points to cam transform using all_r2c_trnsfms
            all_jts3d_cam = np.array([trnsfm_points(pts=j3d, trnsfm=r2c) for (j3d, r2c) in zip(all_jts3d, all_r2c_trnsfms)])

            # compute rel pose  by procrust aligment
            root_joint_idx = None
            all_jts3d_prcrst_pose_rel = []
            all_jts3d_prcrst_algnd = []
            for idx in tqdm(range(1, len(all_jts3d_cam))):
                pts_trnsfmd, trnsfm = compute_similarity_transform(all_jts3d_cam[idx], all_jts3d_cam[jts3d_ref_frm_ind],
                                                                     return_transfom=True, mu_idx=root_joint_idx, scale=1)
                all_jts3d_prcrst_algnd.append(pts_trnsfmd)
                all_jts3d_prcrst_pose_rel.append(rotran2homat(trnsfm['rot'], trnsfm['tran'].flatten()))
            all_jts3d_prcrst_algnd = np.array(all_jts3d_prcrst_algnd)
            all_jts3d_prcrst_pose_rel = np.array(all_jts3d_prcrst_pose_rel)

            # Procrust alignment of r2c and ann translations
            all_ann_transl = all_poses3d_ann_hom[:, :3, 3]
            all_r2c_transl = all_r2c_trnsfms[:, :3, 3]
            pts_trnsfmd, trnsfm = compute_similarity_transform(all_r2c_transl, all_ann_transl, return_transfom=True, mu_idx=None, scale=1)
            r2c_to_ann_trnsfm = rotran2homat(trnsfm['rot'], trnsfm['tran'].flatten())
            all_r2c_transl_algnd2ann = np.array(pts_trnsfmd)

            # relative translation
            all_ann_transl_rel = np.array(
                [all_ann_transl[idx] - all_ann_transl[jts3d_ref_frm_ind] for idx in range(1, len(all_ann_transl))]
                )
            all_ann_transl_rel_dist = np.array(
                [np.linalg.norm(all_ann_transl[idx] - all_ann_transl[jts3d_ref_frm_ind]) for idx in range(1, len(all_ann_transl))]
                )
            
            # relative translation magnitude
            all_r2c_transl_algnd2ann_rel = np.array(
                [all_r2c_transl_algnd2ann[idx] - all_r2c_transl_algnd2ann[jts3d_ref_frm_ind] for idx in range(1, len(all_r2c_transl_algnd2ann))]
                )
            all_r2c_transl_algnd2ann_rel_dist = np.array(
                [np.linalg.norm(all_r2c_transl_algnd2ann[idx] - all_r2c_transl_algnd2ann[jts3d_ref_frm_ind]) for idx in range(1, len(all_r2c_transl_algnd2ann))]
                )
            
            # relative rot magnitude
            all_jts3d_prcrst_pose_rel_rot_dist =  np.array(
            [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i][:3, :3], all_jts3d_prcrst_pose_rel[jts3d_ref_frm_ind][:3, :3]) * (180 / np.pi))
             for i in range(1, len(all_jts3d_prcrst_pose_rel))]
                )
            all_poses3d_ann_rel_rot_dist =  np.array(
            [(geodesic_distance_for_rotations(all_poses3d_ann_rel[i][:3, :3], all_poses3d_ann_rel[jts3d_ref_frm_ind][:3, :3]) * (180 / np.pi))
             for i in range(1, len(all_poses3d_ann_rel))]
                )

            plt.figure(figsize=(25, 15))
            plt.style.use('dark_background')
            plt.plot(range(len(all_jts3d_prcrst_pose_rel_rot_dist)), all_jts3d_prcrst_pose_rel_rot_dist, 'r-', markersize=1)
            plt.plot(range(len(all_poses3d_ann_rel_rot_dist)), all_poses3d_ann_rel_rot_dist, 'g-', markersize=1)
            plt.xlabel('frameid', fontsize=20)
            plt.ylabel('ref-rel rot magnitude', fontsize=20)
            plt.title('rot magnitude error of each frame', fontsize=20)
            plt.legend(['dope', 'ann'])
            save_pth = f'./out/{args.sqn}_filter_{args.filter}_rel_rot_mag.png'
            plt.savefig(save_pth)
            
    else:
        raise ValueError("set correct baseline method")  

    # viz
    if args.viz:
        bb()
        # load all images
        if args.seq_rgbs_dir is not None:
            all_imgs_pths = sorted(glob.glob(osp.join(args.seq_rgbs_dir, '*.png')))
            print('Loading images...')
            all_inp_imgs = np.array(
                [cv2.imread(imp)[:, :, ::-1] for imp in tqdm(all_imgs_pths)]
            )
        else:
            all_inp_imgs = None

        save_pth = os.path.join('./out/vid.mp4')
        create_juxt_vid(filepath=save_pth, inp_imgs=all_inp_imgs, jts_order='DOPE',
                        all_2d_jts=all_jts2d, all_3d_jts_rt=all_jts3d,
                        all_3d_jts_cam=all_jts3d_cam, all_3d_jts_prcst_algnd=None)
    
    if False:
        #Do not use below(camera pose way) to compare relative poses
        #cam centers for dope are very noisy
        all_ann_cam_centers = np.array(
            [-1 * all_poses3d_ann_hom[idx, :3, :3].T @ all_poses3d_ann_hom[idx, :3, 3] for idx in range(len(all_poses3d_ann_hom))]
        )
        all_r2c_cam_centers = np.array(
            [-1 * all_r2c_trnsfms[idx, :3, :3].T @ all_r2c_trnsfms[idx, :3, 3] for idx in range(len(all_r2c_trnsfms))]
        )

    if False:
        import open3d as o3d
        pcd_ann = o3d.geometry.PointCloud()
        pcd_ann.points = o3d.pybind.utility.Vector3dVector(all_r2c_transl_algnd2ann)
        o3d.io.write_point_cloud('./r2c_transl_algnd.ply', pcd_ann, write_ascii=True)
        
        import numpy as np
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(all_ann_transl[:, 0], all_ann_transl[:, 1], all_ann_transl[:, 2], c='g', s=1)
        ax.scatter3D(all_r2c_transl[:, 0], all_r2c_transl[:, 1], all_r2c_transl[:, 2], c='r', s=1)
        plt.savefig('./tmp.png')


    ################ Translation Metrics ##########################
    # Relative translation distance
    print(f"\nTranslation Error/distance between the rel-ann poses and rel-dope procrst algnd poses")
    # all_rel_tran_err = np.linalg.norm((all_jts3d_prcrst_pose_rel[:, :3, 3] - all_poses3d_ann_rel[:, :3, 3]), axis=1)
    # all_rel_tran_err = np.linalg.norm((all_r2c_transl_algnd2ann_rel - all_ann_transl_rel), axis=1)
    all_rel_tran_err = np.abs(all_r2c_transl_algnd2ann_rel_dist - all_ann_transl_rel_dist)

    if args.save:
        save_pth = os.path.join('./out/skeleton_vids/', intent, obj, cam_name, 'method2',
                            'evaluation', 'all_rel_tran_err.npy')
        np.save(save_pth, all_rel_tran_err)
    
    #print
    print(f"Mean: {np.mean(all_rel_tran_err):04f} m") 
    print(f"Median: {np.median(all_rel_tran_err):04f} m")
    print(f"Std: {np.std(all_rel_tran_err):04f} m")
    print(f"Min: {np.min(all_rel_tran_err):04f} m")
    print(f"Max: {np.max(all_rel_tran_err):04f} m")

    all_rel_tran_err_prcntls = np.percentile(all_rel_tran_err, [25, 50, 75, 90])

    print(f"Percentile - 25th: {all_rel_tran_err_prcntls[0]:04f} m")
    print(f"Percentile - 50th: {all_rel_tran_err_prcntls[1]:04f} m")
    print(f"Percentile - 75th: {all_rel_tran_err_prcntls[2]:04f} m")
    print(f"Percentile - 90th: {all_rel_tran_err_prcntls[3]:04f} m")

    # bb()

    ################ Rotation Metrics ##########################
    # Relative rotation distance
    # all_rel_rot_err = np.array(
    #         [(geodesic_distance_for_rotations(all_poses3d_ann_rel[i][:3, :3], all_jts3d_prcrst_pose_rel[i][:3, :3]) * (180 / np.pi))
    #          for i in range(len(all_poses3d_ann_rel))]
    # )
    all_rel_rot_err = np.abs(all_jts3d_prcrst_pose_rel_rot_dist - all_poses3d_ann_rel_rot_dist)

    if args.save:
        save_pth = os.path.join('./out/skeleton_vids/', intent, obj, cam_name, 'method2',
                                'evaluation', 'all_rel_rot_err.npy')
        np.save(save_pth, all_rel_rot_err)
        
    print(f"\nRot mat diff b/w rel-ann poses and rel-dope procrst algnd poses")
    print(f"Mean: {np.mean(all_rel_rot_err):04f} °")
    print(f"Median: {np.median(all_rel_rot_err):04f} °")
    print(f"Std: {np.std(all_rel_rot_err):04f} °")
    print(f"Min: {np.min(all_rel_rot_err):04f} °")
    print(f"Max: {np.max(all_rel_rot_err):04f} °")

    all_rel_rot_err_prcntls = np.percentile(all_rel_rot_err, [25, 50, 75, 90])

    print(f"Percentiles - 25th: {all_rel_rot_err_prcntls[0]:04f} °")
    print(f"Percentiles - 50th: {all_rel_rot_err_prcntls[1]:04f} °")
    print(f"Percentiles - 75th: {all_rel_rot_err_prcntls[2]:04f} °")
    print(f"Percentiles - 90th: {all_rel_rot_err_prcntls[3]:04f} °")

    # plot errors
    fpth_tran_errr = f"./out/{args.sqn}_tran_err_filter_{args.filter}.png" 
    plt_err_line(all_rel_tran_err, save_pth=fpth_tran_errr, xlabel='Frame index',
                ylabel='rel tran err in m', title='Each frame error plot', leg='RTE')
    fpth_rot_errr = f"./out/{args.sqn}_rot_err_filter_{args.filter}.png" 
    plt_err_line(all_rel_rot_err, save_pth=fpth_rot_errr, xlabel='Frame index',
                ylabel='rel rot err in °', title='Each frame error plot', leg='RRE')
    # bb()
    print('Done!!')

    





