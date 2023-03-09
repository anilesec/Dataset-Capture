import os
import numpy  as np
import glob
from tqdm import tqdm
from io3d import *
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
    # pcd.colors = o3d.utility.Vector3dVector(np.array(verts))

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

    all_sqns_str = "20220805164755 20220829155218 20220902153221 20220824154342 20220830162330 20220809161015 20220905151029 20220905155551 20220905153946 20220824104203 20220913145135 20220812180133 20220907153810 20220905154829 20220905111237 20220902154737 20220913151554 20220909114359 20220912144751 20220909145039 20220909140016 20220902163904 20220811170459 20220705173214 20220912155637 20220824142508 20220905140306 20220824160141 20220805165947 20220902114024 20220830161218 20220902111535 20220902104048 20220909151546 20220824152850 20220912161700 20220909111450 20220824150228 20220913153520 20220824105341 20220811172657 20220912160620 20220909113237 20220823113402 20220902111409 20220809163444 20220819155412 20220824181949 20220909142411 20220912151849 20220902151726 20220811165540 20220811163525 20220907155615 20220909134639 20220909120614 20220912143756 20220905105332 20220902170443 20220905112733 20220913144436 20220823115809 20220902110304 20220902163950 20220912164407 20220819162529 20220823114538 20220905142354 20220812170512 20220809171854 20220829154032 20220912165455 20220913154643 20220811171507 20220909115705 20220824155144 20220830163143 20220909152911 20220824144438 20220902164854 20220905112623 20220907152036 20220905141444 20220812174356 20220912161552 20220909141430 20220824180652 20220909121541 20220819164041 20220912142017 20220912152000 20220809170847 20220824102636 20220902115034 20220812172414 20220811154947"
    all_sqns = all_sqns_str.split(' ')

    clmp_pose_missing_sqns = []
    clmp_pose_valid_sqns = []
    all_seqs_mean_RRE = []
    all_seqs_mean_RTE = []
    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)

        # get cam poses
        cps_pths_clmp = sorted(glob.glob(osp.join(COLMAP_BASE_DIR, sqn, 'cam_poses/*.txt')))
        cps_pths_ann = sorted(glob.glob(osp.join(RES_DIR, sqn, 'icp_res/*/f_trans.txt')))

        if len(cps_pths_clmp) != len(cps_pths_ann):
            print("cps_pths_clmp:", len(cps_pths_clmp))
            print("cps_pths_ann:", len(cps_pths_ann))
            clmp_pose_missing_sqns.append(sqn)
            continue
        else:
            clmp_pose_valid_sqns.append(sqn)
            cps_clmp = np.array([np.loadtxt(cp_pth_clmp) for cp_pth_clmp in tqdm(cps_pths_clmp)])
            cps_ann = np.array([np.linalg.inv(np.loadtxt(cp_pth_ann)) for cp_pth_ann in tqdm(cps_pths_ann)])

            # plot translations as its
            cps_clmp_tran = cps_clmp[:, :3, 3]
            cps_ann_tran = cps_ann[:, :3, 3]

            pcd_clmp = create_o3d_pcd(cps_clmp_tran, [[1., 0., 0.]])
            pcd_clmp_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_tran.ply')
            write_o3d_pcd(pcd_clmp_fpth, pcd_clmp)

            pcd_ann = create_o3d_pcd(cps_ann_tran, [[0., 1., 0.]])
            pcd_ann_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_ann_tran.ply')
            write_o3d_pcd(pcd_ann_fpth, pcd_ann)

            # bb()

            # plot cam centers -R.T@t
            cps_clmp_cam_center = np.array([-1 * (cp[:3, :3].T @ cp[:3, 3]) for cp in cps_clmp])
            cps_ann_cam_center = np.array([-1 * (ap[:3, :3].T @ ap[:3, 3]) for ap in cps_ann])

            pcd_clmp_cam_center = create_o3d_pcd(cps_clmp_cam_center, [[1., 0., 0.]])
            pcd_clmp_cc_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_cam_center.ply')
            write_o3d_pcd(pcd_clmp_cc_fpth, pcd_clmp_cam_center)

            pcd_ann_cam_center = create_o3d_pcd(cps_ann_cam_center, [[0., 1., 0.]])
            pcd_ann_cc_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_ann_cam_center.ply')
            write_o3d_pcd(pcd_ann_cc_fpth, pcd_ann)

            # bb()
            # prcst algnmnt b/w cam pose transl
            cps_clmp_tran_algnd, tform_tran = compute_similarity_transform(S1=cps_clmp_tran.T, S2=cps_ann_tran.T, return_transfom=True)
            pcd_clmp_tran_algnd = create_o3d_pcd(cps_clmp_tran_algnd.T, [[1., 0., 0.]])
            pcd_clmp_tran_algnd_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_tran_algnd.ply')
            write_o3d_pcd(pcd_clmp_tran_algnd_fpth, pcd_clmp_tran_algnd)

            # prcst algnmnt b/w cam centers
            cps_clmp_cam_center_algnd, tform_cam_center = compute_similarity_transform(S1=cps_clmp_cam_center.T, S2=cps_ann_cam_center.T, return_transfom=True)
            pcd_clmp_cam_center_algnd = create_o3d_pcd(cps_clmp_cam_center_algnd.T, [[1., 0., 0.]])
            pcd_clmp_cc_algnd_fpth = osp.join(COLMAP_BASE_DIR, sqn, 'pcd_clmp_cam_center_algnd.ply')
            write_o3d_pcd(pcd_clmp_cc_algnd_fpth, pcd_clmp_cam_center_algnd)  

            # ref relative tran
            cps_clmp_tran_algnd = cps_clmp_tran_algnd.T
            all_rel_tran_clmp = np.array([(cps_clmp_tran_algnd[i] - cps_clmp_tran_algnd[0]) for i in range(1, len(cps_clmp_tran_algnd))])
            all_rel_tran_ann = np.array([(cps_ann_tran[i] - cps_ann_tran[0]) for i in range(1, len(cps_ann_tran))])

            # ref relative rot
            all_rel_rot_clmp = np.array(
                [cps_clmp[idx, :3, :3] @ np.linalg.inv(cps_clmp[0, :3, :3])
                for idx in range(1, len(cps_clmp))]
                )
            all_rel_rot_ann = np.array(
                [cps_ann[idx, :3, :3] @ np.linalg.inv(cps_ann[0, :3, :3])
                for idx in range(1, len(cps_ann))]
                )
            
            # ref-rel tran error(after alignment)
            all_rel_tran_err = np.linalg.norm((all_rel_tran_clmp - all_rel_tran_ann), axis=1)

            # rel-rel rot error
            all_rel_rot_err = np.array(
            [(geodesic_distance_for_rotations(all_rel_rot_clmp[i, :3, :3], all_rel_rot_ann[i, :3, :3]) * (180 / np.pi))
            for i in range(1, len(all_rel_rot_clmp))]
            )

            disp_rot_err(all_rel_rot_err)
            disp_tran_err(all_rel_tran_err)

            all_seqs_mean_RRE.append(all_rel_rot_err.mean())
            all_seqs_mean_RTE.append(all_rel_tran_err.mean())

            print('Saving the rel pose error')
            save_dict = {
                'sqn' : sqn,
                'REPE_mean': '-',
                'RRE': all_rel_rot_err,
                'RRE_mean' : np.mean(all_rel_rot_err),
                'RTE' : all_rel_tran_err,
                'RTE_mean' : np.mean(all_rel_tran_err),
            }
            save_dict_pth = osp.join(RES_DIR, sqn, 'eval_rel_pose_clmp.pkl')
            saveas_pkl(save_dict, save_dict_pth)
            print(f"save here: {save_dict_pth}")

    print(f"clmp_pose_missing_sqns: {clmp_pose_missing_sqns} len: {len(clmp_pose_missing_sqns)}")
    print(f"clmp_pose_valid_sqns: {clmp_pose_valid_sqns} len: {len(clmp_pose_valid_sqns)}")

    all_seqs_mean_RRE = np.array(all_seqs_mean_RRE)
    all_seqs_mean_RTE = np.array(all_seqs_mean_RTE)

    print('Avg of mean-RRE of all seqs:', all_seqs_mean_RRE.mean())
    print('Avg of mean-RTE of all seqs:', all_seqs_mean_RTE.mean())

    print('Done')








