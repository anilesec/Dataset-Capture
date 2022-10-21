from dis import dis
from tkinter.messagebox import NO
import numpy as np
import os, sys, glob
from ipdb import set_trace as bb
import argparse
import pprint, pickle
from tqdm import tqdm
import pathlib
osp = os.path

from evaluation.pnps import *
from evaluation.eval_utils import *
from evaluation.viz_utils import *

CAM_INTR = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
   )
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')


def compute_relp_cam_from_jts3d(all_jts3d_cam, rel_type=None):
    if rel_type == 'CONSEQ':
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
            pts_trnsfmd, trnsfm = compute_similarity_transform(
                all_jts3d_cam[0], all_jts3d_cam[idx], return_transfom=True, mu_idx=None, scale=1
                )
            all_jts3d_prcrst_algnd = np.append(all_jts3d_prcrst_algnd, pts_trnsfmd.reshape(1, 21, 3), axis=0)
            all_jts3d_prcrst_pose_rel =  np.append(all_jts3d_prcrst_pose_rel, rotran2homat(trnsfm['rot'],
             trnsfm['tran'].flatten()).reshape(1, 4, 4), axis=0)
    else:
        raise ValueError(f"wrong {rel_type} value")

    return all_jts3d_prcrst_pose_rel,  all_jts3d_prcrst_algnd

def compute_root2cam_ops(all_jts2d, all_jts3d):
    # pnp minimization
    all_r2c_trnsfms = cv2_PNPSolver(all_jts3d, all_jts2d, CAM_INTR)

    projtd_jts2d = []
    for r2c, j3d in zip(all_r2c_trnsfms, all_jts3d):
        proj_mat = CAM_INTR @ r2c
        projtd_jts2d.append(project(P=proj_mat, X=j3d))

    print(f"Reproj Err: {np.mean(np.linalg.norm(all_jts2d - projtd_jts2d, axis=2), 1).mean()}")
    
    # transform 3d points to cam transform using all_r2c_trnsfms
    all_jts3d_cam = np.array(
        [trnsfm_points(pts=j3d, trnsfm=r2c) for (j3d, r2c) in zip(all_jts3d, all_r2c_trnsfms)]
        )
    
    return all_r2c_trnsfms, all_jts3d_cam

def disp_rot_err(all_rot_err):
    print("==============================================================================================")
    print(f"\nRotmat Error/distance between ref-rel of ann jts3d and ref-rel dope jts3d in cam spaces")
    print(f"Mean: {np.mean(all_rot_err):04f} °")
    print(f"Median: {np.median(all_rot_err):04f} °")
    print(f"Std: {np.std(all_rot_err):04f} °")
    print(f"Min: {np.min(all_rot_err):04f} °")
    print(f"Max: {np.max(all_rot_err):04f} °")

    all_rot_err_prcntls = np.percentile(all_rot_err, [25, 50, 75, 90])

    print(f"Percentiles - 25th: {all_rot_err_prcntls[0]:04f} °")
    print(f"Percentiles - 50th: {all_rot_err_prcntls[1]:04f} °")
    print(f"Percentiles - 75th: {all_rot_err_prcntls[2]:04f} °")
    print(f"Percentiles - 90th: {all_rot_err_prcntls[3]:04f} °")
    print("==============================================================================================")

    return None

def disp_tran_err(all_tran_err):
    print("==============================================================================================")
    print(f"\nTranslation Error/distance between ref-rel of ann jts3d and ref-rel dope jts3d in cam spaces")
    print(f"Mean: {np.mean(all_tran_err):04f} m") 
    print(f"Median: {np.median(all_tran_err):04f} m")
    print(f"Std: {np.std(all_tran_err):04f} m")
    print(f"Min: {np.min(all_tran_err):04f} m")
    print(f"Max: {np.max(all_tran_err):04f} m")

    all_tran_err_prcntls = np.percentile(all_tran_err, [25, 50, 75, 90])

    print(f"Percentile - 25th: {all_tran_err_prcntls[0]:04f} m")
    print(f"Percentile - 50th: {all_tran_err_prcntls[1]:04f} m")
    print(f"Percentile - 75th: {all_tran_err_prcntls[2]:04f} m")
    print(f"Percentile - 90th: {all_tran_err_prcntls[3]:04f} m")
    print("==============================================================================================")

    return None

if __name__ == "__main__":
    print("compute relative poses")
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None, help='seq id')
    # parser.add_argument('--bsln_method', type=str, required=True,
    #                     help='baseline method like DOPE, FRANKMOCAP, etc. ')
    parser.add_argument('--add_noise', type=int, default=0, help='add(1) noise or not(0) to ann 3d jts')

    args = parser.parse_args()

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    if args.sqn in all_sqns:
        if args.sqn in all_sqns:
            # load ann poses
            print('Loading annotation poses...')
            seq_pose3d_ann_dir = osp.join(RES_DIR, args.sqn, 'icp_res')
            all_poses_ann = load_poses3d_ann(seq_pose3d_ann_dir)

            # load ann 2d and 3d jts
            print('Loading annotation 2d & 3d jts...')
            sqn_dir = osp.join(RES_DIR, args.sqn)
            all_jts2d, all_jts3d = load_ann_jts(sqn_dir)

            # add gaus noise
            if args.add_noise:
                noise = np.random.normal(loc=0., scale=0.005, size=all_jts3d.shape)
                all_jts3d = all_jts3d + noise

            all_r2c_trnsfms, all_jts3d_cam = compute_root2cam_ops(all_jts2d, all_jts3d)

            all_jts3d_prcrst_pose_rel, all_jts3d_prcrst_algnd = compute_relp_cam_from_jts3d(all_jts3d_cam, rel_type='REF')

            rel_type = 'REF'
            if rel_type == 'CONSEQ':
                pass 
            elif rel_type == 'REF':
                all_poses_ann_rel = np.array(
                    [all_poses_ann[idx] @ np.linalg.inv(all_poses_ann[0])
                    for idx in range(1, len(all_poses_ann))]
                    )
                all_poses_ann_prcst_rel, all_jts3d_ann_prcst_algnd = compute_relp_cam_from_jts3d(all_jts3d, rel_type='REF')
            
            ## Relative rotation erorr
            # using rel-pose obtained from ann poses
            all_rel_rot_err = np.array(
                [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i, :3, :3], all_poses_ann_rel[i-1, :3, :3]) * (180 / np.pi))
             for i in range(1, len(all_jts3d_prcrst_pose_rel))]
             )
            # using rel-pose obtained from ann 3d jts prcst
            all_jts3d_prcrst_pose_rel_rot_err = np.array(
                [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i, :3, :3], all_poses_ann_prcst_rel[i, :3, :3]) * (180 / np.pi))
             for i in range(1, len(all_jts3d_prcrst_pose_rel))]
             )
            
            # print rot err
            print("all_jts3d_prcrst_pose_rel_rot_err:")
            _ = disp_rot_err(all_jts3d_prcrst_pose_rel_rot_err)
            print("all_rel_rot_err:")
            _ = disp_rot_err(all_rel_rot_err)
            
            ## Relative translation error
            # using rel-pose obtained from ann poses
            all_rel_tran_err = np.linalg.norm((all_jts3d_prcrst_pose_rel[1:, :3, 3] - all_poses_ann_rel[:, :3, 3]), axis=1)
            # using rel-pose obtained from ann 3d jts prcst
            all_jts3d_prcrst_pose_rel_tran_err = np.linalg.norm((all_jts3d_prcrst_pose_rel[:, :3, 3] - all_poses_ann_prcst_rel[:, :3, 3]), axis=1)

            # print tran err
            print("all_jts3d_prcrst_pose_rel_tran_err:")
            _ = disp_tran_err(all_jts3d_prcrst_pose_rel_tran_err)
            print("all_rel_tran_err:")
            _ = disp_tran_err(all_rel_tran_err)
            

            bb()
            if False:
                # plot ann pose hand centers and r2c pose hand centers
                all_jts3d_prcrst_pose_rel_tran = all_jts3d_prcrst_pose_rel[:, :3, 3]
                all_poses_ann_rel_tran = all_poses_ann_rel[:, :3, 3]
                import open3d as o3d
                pcd_all_poses_ann_rel_tran = o3d.geometry.PointCloud()
                pcd_all_poses_ann_rel_tran.points = o3d.pybind.utility.Vector3dVector(all_poses_ann_rel_tran)
                o3d.io.write_point_cloud('out/anns/all_poses_ann_rel_tran.ply', pcd_all_poses_ann_rel_tran, write_ascii=True)

                pcd_all_jts3d_prcrst_pose_rel_tran = o3d.geometry.PointCloud()
                pcd_all_jts3d_prcrst_pose_rel_tran.points = o3d.pybind.utility.Vector3dVector(all_jts3d_prcrst_pose_rel_tran)
                o3d.io.write_point_cloud('out/anns/all_jts3d_prcrst_pose_rel_tran.ply', pcd_all_jts3d_prcrst_pose_rel_tran, write_ascii=True)

                pts_trnsfmd, trnsfm = compute_similarity_transform(all_jts3d_prcrst_pose_rel_tran[1:], all_poses_ann_rel_tran, return_transfom=True, mu_idx=None, scale=1)
                r2c_to_ann_trnsfm = rotran2homat(trnsfm['rot'], trnsfm['tran'].flatten())
                all_r2c_transl_algnd2ann = np.array(pts_trnsfmd)
                pcd_pts_trnsfmd = o3d.geometry.PointCloud()
                pcd_pts_trnsfmd.points = o3d.pybind.utility.Vector3dVector(pts_trnsfmd)
                o3d.io.write_point_cloud('out/anns/pts_trnsfmd.ply', pcd_pts_trnsfmd, write_ascii=True)

            if False:    
                import torch, roma
                res1 = roma.rotmat_geodesic_distance_naive(torch.tensor(all_jts3d_prcrst_pose_rel[1:, :3, :3]), torch.tensor(all_poses_ann_rel[:, :3, :3]))
                print(f"{res1.mean() * (180 / np.pi)} deg")
            
            # juxt viz
            if False: 
                all_imgs_pths = sorted(glob.glob(osp.join(RES_DIR, args.sqn, 'rgb/*.png')))
                print('Loading images...')
                all_inp_imgs = np.array(
                    [cv2.imread(imp)[:, :, ::-1] for imp in tqdm(all_imgs_pths)]
                )
                save_pth = os.path.join('./out/vid_ann_noise_4mm.mp4')
                create_juxt_vid(filepath=save_pth, inp_imgs=all_inp_imgs, jts_order='OURS',
                            all_2d_jts=all_jts2d, all_3d_jts_rt=all_jts3d,
                            all_3d_jts_cam=all_jts3d_cam, all_3d_jts_prcst_algnd=None)

            # .ply files
            if False:
                # plot ann pose hand centers and r2c pose hand centers
                all_ann_transl = all_poses_ann[:, :3, 3]
                all_r2c_transl = all_r2c_trnsfms[:, :3, 3]
                import open3d as o3d
                pcd_ann = o3d.geometry.PointCloud()
                pcd_ann.points = o3d.pybind.utility.Vector3dVector(all_ann_transl)
                o3d.io.write_point_cloud('out/anns/ann_transl_.ply', pcd_ann, write_ascii=True)

                pcd_r2c = o3d.geometry.PointCloud()
                pcd_r2c.points = o3d.pybind.utility.Vector3dVector(all_r2c_transl)
                o3d.io.write_point_cloud('out/anns/ann_r2c_transl_.ply', pcd_r2c, write_ascii=True)

                # plot ann cam centers(-R.T@t) and r2c cam centers (-R.T@t)
                all_ann_cam_centers = np.array(
                    [-1 * all_poses_ann_hom[idx, :3, :3].T @ all_poses_ann_hom[idx, :3, 3] for idx in range(len(all_poses_ann_hom))]
                )
                all_r2c_cam_centers = np.array(
                    [-1 * all_r2c_trnsfms[idx, :3, :3].T @ all_r2c_trnsfms[idx, :3, 3] for idx in range(len(all_r2c_trnsfms))]
                )
                import open3d as o3d
                pcd_ann_cc = o3d.geometry.PointCloud()
                pcd_ann_cc.points = o3d.pybind.utility.Vector3dVector(all_ann_cam_centers)
                o3d.io.write_point_cloud('out/anns/ann_cam_centers_.ply', pcd_ann_cc, write_ascii=True)

                pcd_r2c_cc = o3d.geometry.PointCloud()
                pcd_r2c_cc.points = o3d.pybind.utility.Vector3dVector(all_r2c_cam_centers)
                o3d.io.write_point_cloud('out/anns/ann_r2c_cam_centers.ply', pcd_r2c_cc, write_ascii=True)

                bb()

    else:
        raise ValueError(f"{args.sqn} is not present in selected all_sqns list, check the sqn value!!")



    
# res1 = roma.rotmat_geodesic_distance_naive(torch.tensor(all_jts3d_prcrst_pose_rel[1:, :3, :3]), torch.tensor(all_poses_ann_rel[:, :3, :3])
