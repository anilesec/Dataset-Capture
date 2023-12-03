import os
import numpy  as np
import glob
from tqdm import tqdm
# from io3d import *
import pathlib
import torch, roma
from ipdb import set_trace
from evaluation.eval_utils import *
from evaluation.viz_utils import *

osp = os.path
SRATRA_BASE_DIR = pathlib.Path("/scratch/2/user/aswamy/projects/hhor_evaluation/HHOR/data/DEMO_OUT") # 20220912160620/relpose_sra_tra_relpose_clas/camera_poses
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

def compute_relrot_mat_geodist(gt_rots, est_rots):
    relrot_mat_estim = np.linalg.inv(est_rots[:, None, :3, :3]) @ est_rots[:, :3, :3]
    relrot_mat_gt = np.linalg.inv(gt_rots[:, None, :3, :3]) @ gt_rots[:, :3, :3]
    # relrot_mat_estim = est_rots[:, :3, :3] @ np.linalg.inv(est_rots[:, None, :3, :3])
    # relrot_mat_gt = gt_rots[:, :3, :3] @ np.linalg.inv(gt_rots[:, None, :3, :3])
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
    return rel_rot_diff

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("--sqn", type=str, default=None, help='seq no.')
    parser.add_argument('--write', action='store_true', help='flag to write')
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

    
    est_pose_missing_sqns = []
    est_pose_valid_sqns = []
    all_seqs_mean_RRE = []
    all_seqs_mean_RTE = []
    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)
        # get cam poses
        cps_pths_est = sorted(glob.glob(osp.join(SRATRA_BASE_DIR, sqn, 'relpose_sra_tra_relpose_clas/camera_poses/*.txt')))

        all_gt_poses_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'icp_res/*/f_trans.txt')))
        print(f"SRA TRA yields {len(cps_pths_est)} / {len(all_gt_poses_pths)} poses")
        print(f'Detection rate: {len(cps_pths_est) / len(all_gt_poses_pths)*100:.2f}')

        cps_pths_ann = []
        for pth in cps_pths_est:
            path = osp.join(RES_DIR, sqn, f"icp_res/{osp.basename(pth).split('.')[0]}/f_trans.txt")
            cps_pths_ann.append(path)
            est_pose_valid_sqns.append(sqn)

        cps_est = np.array([np.loadtxt(cp_pth_est) for cp_pth_est in tqdm(cps_pths_est)])
        cps_ann = np.array([np.loadtxt(cp_pth_ann) for cp_pth_ann in tqdm(cps_pths_ann)])

        # Relative rotation error
        relrot_err_mat = compute_relrot_mat_geodist(gt_rots=cps_ann[:, :3, :3], est_rots=cps_est[:, :3, :3]);
        relrot_err_amp = compute_relrot_amp_geodist(gt_rots=cps_ann[:, :3, :3], est_rots=cps_est[:, :3, :3],   ref_frm_ind=None);

        # translation error
        gt_tran = cps_ann[:, :3, 3]
        est_tran = cps_est[:, :3, 3]
        est_tran_algnd = compute_similarity_transform(est_tran, gt_tran)
        abs_tran_err = np.linalg.norm((gt_tran - est_tran_algnd), axis=1)
        print(f'Absolute Translation Error: {abs_tran_err.mean():.3f} m')

        rel_tran_gt = cps_ann[:, None, :3, 3] - cps_ann[:, :3, 3]
        rel_tran_est = est_tran_algnd[:, None, :] - est_tran_algnd
        rel_tran_err = np.linalg.norm((rel_tran_gt-rel_tran_est), axis=2)
        print(f'Rel Tran. Err.: {rel_tran_err.mean():.3f} m')

        all_seqs_mean_RRE.append(relrot_err_mat.mean())
        all_seqs_mean_RTE.append(rel_tran_err.mean())
        
        print('Saving the rel pose error for a give rot and tran threshold')
        save_dict = {
            'sqn' : sqn,
            'RRE_mat_valid_pairs' : relrot_err_mat,
            'RTE_mat_valid_pairs' : rel_tran_err,
            'RRE_valid_pairs_mean' : (relrot_err_mat).mean().detach().cpu().numpy().item(),
            'RTE_valid_pairs_mean' : (rel_tran_err).mean().item(),
            'DET_rate' : len(cps_pths_est) / len(all_gt_poses_pths)*100
        }

        n_C = len(cps_pths_est) # no of colmap frames
        n_N = len(all_gt_poses_pths) # no of total frames
        n_F = n_N - n_C # no of failed frames
        off = (n_F * n_N) + n_F * (n_N - n_F)
        total_num_pairs = n_C**2 + off
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
        if args.write:
            save_dict_pth = osp.join(RES_DIR, sqn, 'eval_rel_pose_sratra_latest.pkl')
            saveas_pkl(save_dict, save_dict_pth)
            print(f"save here: {save_dict_pth}")

    print(f"clmp_pose_missing_sqns: {est_pose_missing_sqns} len: {len(est_pose_missing_sqns)}")
    print(f"clmp_pose_valid_sqns: {est_pose_valid_sqns} len: {len(est_pose_valid_sqns)}")

    all_seqs_mean_RRE = np.array(all_seqs_mean_RRE)
    all_seqs_mean_RTE = np.array(all_seqs_mean_RTE)
    print('Avg of mean-RRE of all seqs:', all_seqs_mean_RRE.mean())
    print('Avg of mean-RTE of all seqs:', all_seqs_mean_RTE.mean())
    print('Done')


