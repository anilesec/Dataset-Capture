import os
import numpy  as np
import glob
from tqdm import tqdm
# from io3d import *
import pathlib
from ipdb import set_trace
from evaluation.eval_utils import *
from evaluation.viz_utils import *

osp = os.path
COLMAP_BASE_DIR = pathlib.Path("/scratch/1/user/aswamy/data/colmap-hand-obj")
RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')

def create_o3d_pcd(verts, clr=np.array([[0., 0., 1.]])):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(verts))

    clrs = np.repeat(clr, len(verts), axis=0)
    pcd.colors = o3d.utility.Vector3dVector(np.array(verts))
    return pcd

def write_o3d_pcd(file_path, pcd_o3d):
        import open3d as o3d
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        o3d.io.write_point_cloud(file_path, pcd_o3d, write_ascii=True)

        return print(f"saved: {file_path}")

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

    clmp_pose_missing_sqns = []
    clmp_pose_valid_sqns = []
    all_seqs_mean_RRE = []
    all_seqs_mean_RTE = []
    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)
        # get cam poses
        cps_pths_clmp = sorted(glob.glob(osp.join(COLMAP_BASE_DIR, sqn, 'cam_poses/*.txt')))

        all_gt_poses_pths = sorted(glob.glob(osp.join(RES_DIR, sqn, 'icp_res/*/f_trans.txt')))
        print(f"colmap yields {len(cps_pths_clmp)} / {len(all_gt_poses_pths)} poses")
        print(f'detection rate: {len(cps_pths_clmp) / len(all_gt_poses_pths)*100:.2f}')

        cps_pths_ann = []
        for pth in cps_pths_clmp:
            path = osp.join(RES_DIR, sqn, f"icp_res/{osp.basename(pth).split('.')[0]}/f_trans.txt")
            cps_pths_ann.append(path)
        
        clmp_pose_valid_sqns.append(sqn)
        cps_clmp = np.array([np.loadtxt(cp_pth_clmp) for cp_pth_clmp in tqdm(cps_pths_clmp)])
        cps_ann = np.array([np.linalg.inv(np.loadtxt(cp_pth_ann)) for cp_pth_ann in tqdm(cps_pths_ann)])

        # save camera centers
        if False:
            import trimesh
            base_savedir = f"/scratch/2/user/aswamy/projects/HandObjectRelPose/out/cam_poses_viz/showme/{sqn}/cam_centers"
            os.makedirs(base_savedir, exist_ok=True)
            pcd_ann = trimesh.points.PointCloud(np.linalg.inv(cps_ann)[:, :3, 3])
            # pcd_ann.export(f'{base_savedir}/cam_centers_ann.ply');
            pcd_clmp = trimesh.points.PointCloud(np.linalg.inv(cps_clmp)[:, :3, 3]) # -R.T @ t (=inv(pose))
            # pcd_clmp.export(f'{base_savedir}/cam_centers_clmp.ply');
            tran_algnd, trnsfm = compute_similarity_transform(pcd_clmp.vertices, pcd_ann.vertices, return_transfom=True)
            pcd_clmp_algnd = trimesh.points.PointCloud(tran_algnd)
            # pcd_clmp_algnd.export(f'{base_savedir}/cam_centers_clmp_algnd.ply');
            abs_cam_centers_err = np.linalg.norm((pcd_ann.vertices - pcd_clmp_algnd.vertices), axis=1)
            print(f'Camera Centers Error: {abs_cam_centers_err.mean():.3f} m')

        # plot translations as its
        cps_clmp_tran = cps_clmp[:, :3, 3]
        cps_ann_tran = cps_ann[:, :3, 3]

        # pcd_clmp = create_o3d_pcd(cps_clmp_tran, [[1., 0., 0.]])
        # pcd_clmp_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_tran.ply')
        # write_o3d_pcd(pcd_clmp_fpth, pcd_clmp)

        # pcd_ann = create_o3d_pcd(cps_ann_tran, [[0., 1., 0.]])
        # pcd_ann_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_ann_tran.ply')
        # write_o3d_pcd(pcd_ann_fpth, pcd_ann)

        # bb()

        # plot cam centers -R.T@t
        cps_clmp_cam_center = np.array([-1 * (cp[:3, :3].T @ cp[:3, 3]) for cp in cps_clmp])
        cps_ann_cam_center = np.array([-1 * (ap[:3, :3].T @ ap[:3, 3]) for ap in cps_ann])

        # pcd_clmp_cam_center = create_o3d_pcd(cps_clmp_cam_center, [[1., 0., 0.]])
        # pcd_clmp_cc_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_cam_center.ply')
        # write_o3d_pcd(pcd_clmp_cc_fpth, pcd_clmp_cam_center)

        # pcd_ann_cam_center = create_o3d_pcd(cps_ann_cam_center, [[0., 1., 0.]])
        # pcd_ann_cc_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_ann_cam_center.ply')
        # write_o3d_pcd(pcd_ann_cc_fpth, pcd_ann)

        # bb()
        # prcst algnmnt b/w cam pose transl
        cps_clmp_tran_algnd, tform_tran = compute_similarity_transform(S1=cps_clmp_tran.T, S2=cps_ann_tran.T, return_transfom=True)
        # pcd_clmp_tran_algnd = create_o3d_pcd(cps_clmp_tran_algnd.T, [[1., 0., 0.]])
        # pcd_clmp_tran_algnd_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_tran_algnd.ply')
        # write_o3d_pcd(pcd_clmp_tran_algnd_fpth, pcd_clmp_tran_algnd)

        # prcst algnmnt b/w cam centers
        # cps_clmp_cam_center_algnd, tform_cam_center = compute_similarity_transform(S1=cps_clmp_cam_center.T, S2=cps_ann_cam_center.T, return_transfom=True)
        # pcd_clmp_cam_center_algnd = create_o3d_pcd(cps_clmp_cam_center_algnd.T, [[1., 0., 0.]])
        # pcd_clmp_cc_algnd_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_cam_center_algnd.ply')
        # write_o3d_pcd(pcd_clmp_cc_algnd_fpth, pcd_clmp_cam_center_algnd)  

        # ref relative tran
        cps_clmp_tran_algnd = cps_clmp_tran_algnd.T
        # all_rel_tran_clmp = np.array([(cps_clmp_tran_algnd[i] - cps_clmp_tran_algnd[0]) for i in range(1, len(cps_clmp_tran_algnd))])
        # all_rel_tran_ann = np.array([(cps_ann_tran[i] - cps_ann_tran[0]) for i in range(1, len(cps_ann_tran))])


        # translation error: (uncomment below block to compute error wrt to all ref-frame)
        abs_tran_err = np.linalg.norm((cps_ann_tran - cps_clmp_tran_algnd), axis=1)
        print(f'Absolute Translation Error: {abs_tran_err.mean():.3f} m')
        rel_tran_gt = cps_ann_tran[:, None, :] - cps_ann_tran
        rel_tran_est = cps_clmp_tran_algnd[:, None, :] - cps_clmp_tran_algnd
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

        relrot_err_mat = compute_relrot_mat_geodist(gt_rots=cps_ann[:, :3, :3], est_rots=cps_clmp[:, :3, :3]);
        # relrot_err_amp = compute_relrot_amp_geodist(gt_rots=cps_ann[:, :3, :3], est_rots=cps_clmp[:, :3, :3], ref_frm_ind=None);

        # ref-rel tran error(after alignment)
        # all_rel_tran_err = np.linalg.norm((all_rel_tran_clmp - all_rel_tran_ann), axis=1)

        all_seqs_mean_RRE.append(relrot_err_mat.mean())
        all_seqs_mean_RTE.append(rel_tran_err.mean())
        
        print('Saving the rel pose error for a give rot and tran threshold')
        save_dict = {
            'sqn' : sqn,
            'RRE_mat_valid_pairs' : relrot_err_mat,
            'RTE_mat_valid_pairs' : rel_tran_err,
            'RRE_valid_pairs_mean' : (relrot_err_mat).mean().detach().cpu().numpy().item(),
            'RTE_valid_pairs_mean' : (rel_tran_err).mean().item(),
            'DET_rate' : len(cps_pths_clmp) / len(all_gt_poses_pths)*100
        }

        n_C = len(cps_pths_clmp) # no of colmap frames
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
            save_dict_pth = osp.join(RES_DIR, sqn, 'eval_rel_pose_clmp_latest.pkl')
            saveas_pkl(save_dict, save_dict_pth)
            print(f"save here: {save_dict_pth}")

    print(f"clmp_pose_missing_sqns: {clmp_pose_missing_sqns} len: {len(clmp_pose_missing_sqns)}")
    print(f"clmp_pose_valid_sqns: {clmp_pose_valid_sqns} len: {len(clmp_pose_valid_sqns)}")

    all_seqs_mean_RRE = np.array(all_seqs_mean_RRE)
    all_seqs_mean_RTE = np.array(all_seqs_mean_RTE)
    print('Avg of mean-RRE of all seqs:', all_seqs_mean_RRE.mean())
    print('Avg of mean-RTE of all seqs:', all_seqs_mean_RTE.mean())

    print('Done')




