from telnetlib import XDISPLOC
from evaluation.eval_recon import saveas_pkl
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

def filter_jts(all_jts, window=5):
    N = len(all_jts)

    all_jts_fltrd = np.zeros(all_jts.shape)
    for idx, jts in enumerate(all_jts):
        if (idx < window) or (idx > N - window):
            all_jts_fltrd[idx] = all_jts[idx]
        else:
            all_jts_fltrd[idx] = np.median(all_jts[idx - window : idx + window], axis=0)
        
    return np.array(all_jts_fltrd)

def disp_rot_err(all_rot_err, just_mean=None):
    print("==============================================================================================")
    print(f"\nRotmat Error/distance between ref-rel of ann jts3d and ref-rel dope jts3d in cam spaces")
    print(f"Mean: {np.mean(all_rot_err):04f} °")
    if just_mean is None:
        print(f"Median: {np.median(all_rot_err):04f} °")
        print(f"Std: {np.std(all_rot_err):04f} °")
        print(f"Min: {np.min(all_rot_err):04f} °")
        print(f"Max: {np.max(all_rot_err):04f} °")

        all_rot_err_prcntls = np.percentile(all_rot_err, [25, 50, 75, 90])

        print(f"Percentiles - 25th: {all_rot_err_prcntls[0]:04f} °")
        print(f"Percentiles - 50th: {all_rot_err_prcntls[1]:04f} °")
        print(f"Percentiles - 75th: {all_rot_err_prcntls[2]:04f} °")
        print(f"Percentiles - 90th: {all_rot_err_prcntls[3]:04f} °")
    # print("==============================================================================================")

    return None

def disp_tran_err(all_tran_err, just_mean=None):
    print("==============================================================================================")
    print(f"\nTranslation Error/distance between ref-rel of ann jts3d and ref-rel dope jts3d in cam spaces")
    print(f"Mean: {np.mean(all_tran_err):04f} m") 
    if just_mean is None:
        print(f"Median: {np.median(all_tran_err):04f} m")
        print(f"Std: {np.std(all_tran_err):04f} m")
        print(f"Min: {np.min(all_tran_err):04f} m")
        print(f"Max: {np.max(all_tran_err):04f} m")

        all_tran_err_prcntls = np.percentile(all_tran_err, [25, 50, 75, 90])

        print(f"Percentile - 25th: {all_tran_err_prcntls[0]:04f} m")
        print(f"Percentile - 50th: {all_tran_err_prcntls[1]:04f} m")
        print(f"Percentile - 75th: {all_tran_err_prcntls[2]:04f} m")
        print(f"Percentile - 90th: {all_tran_err_prcntls[3]:04f} m")
    # print("==============================================================================================")

    return None

if __name__ == "__main__":
    print("compute relative poses")
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None,
                        help='seq id')
    # parser.add_argument('--bsln', type=str, required=True,
    #                     help='baseline method like DOPE, FRANKMOCAP, etc. ')
    # parser.add_argument('--add_noise', type=int, default=0,
    #                     help='add(1) noise or not(0) to ann 3d jts')
    parser.add_argument('--filter', type=int, default=0, choices=[0, 1],
                        help='filter(1), not_filter(0)')
    parser.add_argument('--write', action='store_true', help='flag to write')
    parser.add_argument('--export_poses', action='store_true', help='flag to export poses')

    args = parser.parse_args()

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        all_sqns = [args.sqn]

    all_seqs_mean_RRE = []
    all_seqs_mean_RTE = []
    
    for sqn in tqdm(all_sqns):
        print(f"sqn: {sqn}")
        
        # all_poses_ann_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'icp_res/*/f_trans.txt')))
        all_dope_pkls_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'dope_dets/*.pkl')))
        all_jts2d_ann_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'proj_jts/*.txt')))
        all_jts3d_ann_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'jts3d/*.txt')))

        all_jts2d_dope = []
        all_jts3d_dope = []
        all_jts2d_ann = []
        all_jts3d_ann = []
        all_poses_ann = []

        missing_dets_frms = []
        save_dict = dict()
        print('Load Dope dets and Annotated jts...')
        for j2d_annp, j3d_annp in zip(all_jts2d_ann_pths, all_jts3d_ann_pths):
            dope_pkl_pth = pathlib.Path((j3d_annp.replace('jts3d', 'dope_dets')).replace('txt', 'pkl'))
            if not dope_pkl_pth.exists():
                print(f'Missing dope det: {dope_pkl_pth}')
                missing_dets_frms.append(dope_pkl_pth) 
                continue
            jts2d_dope, jts3d_dope = load_dope_det_frm_pkl(dope_pkl_pth)
            all_jts2d_dope.append(jts2d_dope)
            all_jts3d_dope.append(jts3d_dope)

            jts2d_ann, jts3d_ann = np.loadtxt(j2d_annp), np.loadtxt(j3d_annp)
            all_jts2d_ann.append(jts2d_ann)
            all_jts3d_ann.append(jts3d_ann)

            pose_ann_pth = pathlib.Path(osp.join(RES_DIR, sqn, f"icp_res/{osp.basename(j3d_annp).split('.')[0]}/f_trans.txt"))
            all_poses_ann.append(np.loadtxt(pose_ann_pth))
        
        all_jts2d_dope = np.array(all_jts2d_dope)
        all_jts3d_dope = np.array(all_jts3d_dope)

        all_jts2d_ann = np.array(all_jts2d_ann)
        all_jts3d_ann = np.array(all_jts3d_ann)

        all_poses_ann = np.array(all_poses_ann)

        all_missing_dets_info = [missing_dets_frms, len(missing_dets_frms), len(missing_dets_frms) / (len(all_jts3d_ann)+len(missing_dets_frms))]

        assert len(all_jts3d_dope) == len(all_jts3d_ann), f"missamatch b/w dope {all_jts3d_dope.shape} and ann 3Djts {all_jts3d_ann.shape}"
        assert len(all_jts2d_dope) == len(all_jts2d_ann), f"missamatch b/w dope {all_jts2d_dope.shape} and ann 2Djts {all_jts3d_ann.shape}"\

        # median filtering the dope joints
        if args.filter:
            all_jts2d = filter_jts(all_jts2d_dope, window=5)
            all_jts3d = filter_jts(all_jts3d_dope, window=5)

        # REL-POSE for DOPE
        all_r2c_trnsfms, all_jts3d_cam = compute_root2cam_ops(all_jts2d, all_jts3d)
        all_jts3d_prcrst_pose_rel, all_jts3d_prcrst_algnd = compute_relp_cam_from_jts3d(all_jts3d_cam, rel_type='REF')

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.pybind.utility.Vector3dVector(all_jts3d_cam[0])
        # fpth = osp.join(RES_DIR, sqn, 'dope_jts3d_cam_frm1.ply')
        # o3d.io.write_point_cloud(fpth, pcd, write_ascii=True)
        # np.savetxt(fpth.replace('ply', 'txt'), all_jts3d_cam[0])
        
        # REPRJ Error on r2cs
        projtd_jts2d = []
        for r2c, j3d in zip(all_r2c_trnsfms, all_jts3d):
            proj_mat = CAM_INTR @ r2c
            projtd_jts2d.append(project(P=proj_mat, X=j3d))
        mean_rep_err = np.mean(np.linalg.norm(all_jts2d - projtd_jts2d, axis=2), 1).mean()
        print(f"Reproj Err: {mean_rep_err} pixs")

        # REL-POSE for Anno
        rel_type = 'REF'
        if rel_type == 'CONSEQ':
            raise NotImplementedError 
        elif rel_type == 'REF':
            all_poses_ann_rel = np.array(
                [all_poses_ann[idx] @ np.linalg.inv(all_poses_ann[0])
                for idx in range(1, len(all_poses_ann))]
                )
            all_poses_ann_prcst_rel, all_jts3d_ann_prcst_algnd = compute_relp_cam_from_jts3d(all_jts3d_ann, rel_type='REF')
        
        # Translation error: (added block to evaluate same as SRA)
        abs_tran_err = np.linalg.norm((all_poses_ann_prcst_rel[:, :3, 3] - all_jts3d_prcrst_pose_rel[:, :3, 3]), axis=1)
        print(f'Absolute Translation Error: {abs_tran_err.mean():.3f} m')
        rel_tran_gt = all_poses_ann_prcst_rel[:, None, :3, 3] - all_poses_ann_prcst_rel[:, :3, 3]
        rel_tran_est = all_jts3d_prcrst_pose_rel[:, None, :3, 3] - all_jts3d_prcrst_pose_rel[:, :3, 3]
        rel_tran_err = np.linalg.norm((rel_tran_gt-rel_tran_est), axis=2)
        print(f'Rel Tran. Err.: {rel_tran_err.mean():.3f} m')
        import roma, torch
        def compute_relrot_mat_geodist(gt_rots, est_rots):
            # relrot_mat_estim = np.linalg.inv(est_rots[:, None, :3, :3]) @ est_rots[:, :3, :3]
            # relrot_mat_gt = np.linalg.inv(gt_rots[:, None, :3, :3]) @ gt_rots[:, :3, :3]
            relrot_mat_estim = est_rots[:, :3, :3] @ np.linalg.inv(est_rots[:, None, :3, :3])
            relrot_mat_gt = gt_rots[:, :3, :3] @ np.linalg.inv(gt_rots[:, None, :3, :3])
            relrot_diff = roma.rotmat_geodesic_distance_naive(torch.tensor(relrot_mat_gt), torch.tensor(relrot_mat_estim)) * 180 / np.pi
            print(f"RR Mat. Geod. Dist. w.r.t ALL ref frms : {(relrot_diff).mean():.5f} deg")
            return relrot_diff
        def compute_relrot_amp_geodist(gt_rots, est_rots, ref_frm_ind=None):
            # err wrt all ref frames
            relrot_allrefs_gt = np.array(
                roma.rotmat_geodesic_distance_naive(torch.tensor(gt_rots[:, None, :3, :3]), torch.tensor(gt_rots[:, :3, :3]))
                ) * 180 / np.pi
            relrot_allrefs_est = np.array(
                roma.rotmat_geodesic_distance_naive(torch.tensor(est_rots[:, None, :3, :3]), torch.tensor(est_rots[:, :3, :3]))
                ) * 180 / np.pi
            rel_rot_diff = np.abs(relrot_allrefs_gt - relrot_allrefs_est)
            print(f"RR Amp. Geod. Dist. w.r.t ALL ref frms : {rel_rot_diff.mean():.5f} deg")

            # err with single ref frame
            if ref_frm_ind is not None:
                rel_rot_estim = np.array(
                        [roma.rotmat_geodesic_distance_naive(torch.tensor(est_rots[idx, :3, :3]), torch.tensor(est_rots[ref_frm_ind, :3, :3]))
                        for idx in range(len(est_rots))]
                        ) * 180 / np.pi
                rel_rot_gt = np.array(
                        [roma.rotmat_geodesic_distance_naive(torch.tensor(gt_rots[idx, :3, :3]), torch.tensor(gt_rots[ref_frm_ind, :3, :3]))
                        for idx in range(len(gt_rots))]
                        ) * 180 / np.pi
                rel_rot_diff = np.abs(rel_rot_estim - rel_rot_gt)
                print(f"RR Geod. Dist. w.r.t ref_frm_ind-{ref_frm_ind} : {(np.abs(rel_rot_diff).mean()):.5f} deg")

        print(f'Dope detection rate: {(len(all_jts3d_dope)/len(all_jts3d_ann_pths))*100:.2f}')
        relrot_err_mat = compute_relrot_mat_geodist(gt_rots=all_poses_ann_prcst_rel[:, :3, :3], est_rots=all_jts3d_prcrst_pose_rel[:, :3, :3]);
        relrot_err_amp = compute_relrot_amp_geodist(gt_rots=all_poses_ann_prcst_rel[:, :3, :3], est_rots=all_jts3d_prcrst_pose_rel[:, :3, :3], ref_frm_ind=None);
        # bb()
        ## Relative rotation erorr
        # using rel-pose obtained from ann poses
        # all_rel_rot_err = np.array(
        #     [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i, :3, :3], all_poses_ann_rel[i-1, :3, :3]) * (180 / np.pi))
        #     for i in range(1, len(all_jts3d_prcrst_pose_rel))]
        #     )
        # using rel-pose obtained from ann 3d jts prcst
        # all_jts3d_prcrst_pose_rel_rot_err = np.array(
        #     [(geodesic_distance_for_rotations(all_jts3d_prcrst_pose_rel[i, :3, :3], all_poses_ann_prcst_rel[i, :3, :3]) * (180 / np.pi))
        #     for i in range(1, len(all_jts3d_prcrst_pose_rel))]
        #     )
        
        ## Relative translation error
        # using rel-pose obtained from ann poses
        # all_rel_tran_err = np.linalg.norm((all_jts3d_prcrst_pose_rel[1:, :3, 3] - all_poses_ann_rel[:, :3, 3]), axis=1)
        # using rel-pose obtained from ann 3d jts prcst
        # all_jts3d_prcrst_pose_rel_tran_err = np.linalg.norm((all_jts3d_prcrst_pose_rel[:, :3, 3] - all_poses_ann_prcst_rel[:, :3, 3]), axis=1)

        all_seqs_mean_RRE.append(relrot_err_mat.mean())
        all_seqs_mean_RTE.append(rel_tran_err.mean())

        # print errs
        print("all_jts3d_prcrst_pose_rel_rot_err:", relrot_err_mat.mean(), 'deg')
        print("all_jts3d_prcrst_pose_rel_tran_err:", rel_tran_err.mean(), 'm')

        print('Saving the rel pose error for a give rot and tran threshold')
        save_dict = {
            'sqn' : sqn,
            'RRE_mat_valid_pairs' : relrot_err_mat,
            'RTE_mat_valid_pairs' : rel_tran_err,
            'RRE_valid_pairs_mean' : (relrot_err_mat).mean().detach().cpu().numpy().item(),
            'RTE_valid_pairs_mean' : (rel_tran_err).mean().item(),
            'DET_rate' : len(all_jts3d_dope)/len(all_jts3d_ann_pths)*100,
            'missing_det_info': all_missing_dets_info
        }
        n_D = len(all_jts3d_dope) # no of dope frames
        n_N = len(all_jts3d_ann_pths) # no of total frames
        n_F = n_N - n_D # no of failed frames
        off = (n_F * n_N) + n_F * (n_N - n_F)
        total_num_pairs = n_D**2 + off
        assert total_num_pairs == n_N**2

        ths = [[5.0, 0.05], [10.0, 0.10], [15.0, 0.15], [20.0, 0.20], [25.0, 0.25], [30.0, 0.30]]
        for th in ths:  
            rre_th =  (relrot_err_mat < th[0]).type(torch.float32)
            rte_th =  (rel_tran_err < th[1]).astype(np.uint8)
            rre_rte_th = np.logical_and(rre_th.detach().cpu().numpy(), rte_th)
            save_dict.update({
                f"RRE@{th[0]}" : (rre_th.sum() / total_num_pairs).detach().cpu().numpy().item() * 100,
                f"RTE@{th[1]}" : (rte_th.sum() / total_num_pairs) * 100,
                f"RRE@{th[0]}_RTE@{th[1]}" : (rre_rte_th.sum() / total_num_pairs) * 100
            })
        
        # bb()
        if args.write:
            save_dict_pth = osp.join(RES_DIR, sqn, 'eval_rel_pose_dope_latest.pkl')
            saveas_pkl(save_dict, save_dict_pth)
            print(f"save here: {save_dict_pth}")
        # bb()
        if args.export_poses:
            # bb()
            for idx, relp in tqdm(enumerate(all_jts3d_prcrst_pose_rel)):
                relp_fpth = osp.join(RES_DIR, sqn, 'dope_rel_poses', f"{idx:010d}.txt")
                os.makedirs(osp.dirname(relp_fpth), exist_ok=True)
                np.savetxt(relp_fpth, relp)                            
            print(f'saved rel-poses here: {osp.dirname(relp_fpth)}')
        

    all_seqs_mean_RRE = np.array(all_seqs_mean_RRE)
    all_seqs_mean_RTE = np.array(all_seqs_mean_RTE)

    print('Avg of mean-RRE of all seqs:', all_seqs_mean_RRE.mean())
    print('Avg of mean-RTE of all seqs:', all_seqs_mean_RTE.mean())

    print('Done!')


