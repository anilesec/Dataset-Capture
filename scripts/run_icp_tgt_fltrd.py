
if True:
    import open3d as o3d
    # from scripts.io3d import *
    from io3d import *
    import numpy as np
    import os, sys
    import time
    osp = os.path

    def remove_farther_tgtpts(src, tgt, dist_thresh=0.01):
        "src and tgt must be open3d pcd objects"
        tgt_dists = np.array(tgt.compute_point_cloud_distance(src))
        inds_near = np.where(tgt_dists < dist_thresh)[0]
        inds_far = np.where(tgt_dists > dist_thresh)[0]
        
        tgt_near = tgt.select_by_index(inds_near)
        tgt_far = tgt.select_by_index(inds_far)

        return tgt_near, tgt_far
        
    def initialize(curr_frm_pcd_pth, prev_frm_f_pth, curr_frm_init_pcd_write_pth):
        # 1. Load curr_frm_pcd
        curr_frm_pcd = o3d.io.read_point_cloud(curr_frm_pcd_pth)

        # 2. Load prev_frm_f transform
        curr_frm_i = np.loadtxt(prev_frm_f_pth)

        # 3. Apply curr_frm_i/prev_frm_f to curr_frm_pcd
        curr_frm_pcd.transform(curr_frm_i)
        curr_frm_pcd.estimate_normals()
        # o3d.geometry.orient_normals_towards_camera_location(curr_frm_pcd)

        write_ply(curr_frm_init_pcd_write_pth, verts=np.array(curr_frm_pcd.points), trias=None,
         color=np.array(curr_frm_pcd.colors)*255, normals=np.array(curr_frm_pcd.normals), binary=False)

        return print(curr_frm_init_pcd_write_pth)

    def filtr_tgt_pcd(tgt_pcd_pth, curr_frm_init_pcd_pth, tgt_fltrd_pcd_write_pth, dist_thresh):
        # bb()
        tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_pth)
        curr_frm_pcd = o3d.io.read_point_cloud(curr_frm_init_pcd_pth)

        tgt_near_pcd, tgt_far_pcd = remove_farther_tgtpts(src=curr_frm_pcd, tgt=tgt_pcd, dist_thresh=dist_thresh)

        write_ply(tgt_fltrd_pcd_write_pth, verts=np.array(tgt_near_pcd.points), trias=None,
        color=np.array(tgt_near_pcd.colors)*255, normals=np.array(tgt_near_pcd.normals), binary=False)

        return print(tgt_fltrd_pcd_write_pth)

    def FRICP(icp_dir, tgt_fltrd_pcd_pth, src_init_pcd_pth, res_dir, icp_enum=3):
        # icp_dir = '/scratch/github_repos/Fast-Robust-ICP'
        print('Running bash commands...')
        print((f"{osp.join(icp_dir, 'build/FRICP')} {tgt_fltrd_pcd_pth} {src_init_pcd_pth} {res_dir}/ {icp_enum}"))
        os.system(f"{osp.join(icp_dir, 'build/FRICP')} {tgt_fltrd_pcd_pth} {src_init_pcd_pth} {res_dir}/ {icp_enum}")

    def finalize(curr_frm_pcd_pth, curr_frm_o_pth, curr_frm_i_pth, curr_frm_f_write_pth, curr_frm_f_pcd_write_pth):
        # 4. load current frame icp transfomration cur_frm_o 
        curr_frm_o = np.loadtxt(curr_frm_o_pth)
        
        curr_frm_i = np.loadtxt(curr_frm_i_pth)

        # 5. compute current frame final transformation curr_frm_f
        curr_frm_f = curr_frm_o @ curr_frm_i

        # 6. save current frame final tranformation path
        np.savetxt(curr_frm_f_write_pth, curr_frm_f)

        # 7. save curr frm final transformation pcd
        curr_frm_pcd = o3d.io.read_point_cloud(curr_frm_pcd_pth)
        curr_frm_pcd.transform(curr_frm_f)
        write_ply(curr_frm_f_pcd_write_pth, verts=np.array(curr_frm_pcd.points), trias=None,
         color=np.array(curr_frm_pcd.colors)*255, normals=np.array(curr_frm_pcd.normals), binary=False)

        return print(f'final pcd:{curr_frm_f_pcd_write_pth}, final trnsfm:{curr_frm_f_write_pth}')


    seq = '20220824144438'

    if seq == '20220705173214':
        for frm_idx in range(800, 900):
            print(f"\n frm_idx: {frm_idx}")
            # Step1 (create src initialzation)
            curr_frm_pcd_pth = f"/scratch/github_repos/Fast-Robust-ICP/data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/0000000{frm_idx}.ply"
            prev_frm_f_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx-1}/f_trans.txt'
            curr_frm_init_pcd_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/init_0000000{frm_idx}.ply'
            initialize(curr_frm_pcd_pth, prev_frm_f_pth, curr_frm_init_pcd_write_pth)
            
            # Step2 (filter target pcd)
            icp_dir = '/scratch/github_repos/Fast-Robust-ICP/'
            tgt_pcd_pth = f"{osp.join(icp_dir, f'data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/tgt_pcd.ply')}"
            curr_frm_init_pcd_pth = f"{osp.join(icp_dir, f'data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/init_0000000{frm_idx}.ply')}"
            tgt_fltrd_pcd_write_pth = f"{osp.join(icp_dir, f'data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/tgt_fltrd_0000000{frm_idx}.ply')}"
            filtr_tgt_pcd(tgt_pcd_pth, curr_frm_init_pcd_pth, tgt_fltrd_pcd_write_pth, dist_thresh=0.01)

            # Step3 (run FRICP between src_init and tgt_fltrd)  
            icp_dir = '/scratch/github_repos/Fast-Robust-ICP'
            res_dir = osp.join(icp_dir ,f'res/{seq}/frm{frm_idx}')
            os.makedirs(res_dir, exist_ok=True)
            FRICP(tgt_fltrd_pcd_pth=tgt_fltrd_pcd_write_pth, src_init_pcd_pth=curr_frm_init_pcd_write_pth, res_dir=res_dir)

            # Step4 (save final registered mesh and transformation)
            curr_frm_pcd_pth = f"/scratch/github_repos/Fast-Robust-ICP/data/{seq}/fgnd_xyz_wc_wn_wo_outlr_iter2_cleaned/0000000{frm_idx}.ply"
            curr_frm_o_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/m3trans.txt'
            curr_frm_i_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx-1}/f_trans.txt'
            curr_frm_f_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/f_trans.txt'
            curr_frm_f_pcd_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/f_pcd.ply'
            finalize(curr_frm_pcd_pth, curr_frm_o_pth, curr_frm_i_pth, curr_frm_f_write_pth, curr_frm_f_pcd_write_pth)
        
    elif seq == '20220715182504':
        for frm_idx in range(800, 852):
            # bb()
            print(f"\n frm_idx: {frm_idx}")
            # Step1 (create src initialzation)
            curr_frm_pcd_pth = f"/scratch/github_repos/Fast-Robust-ICP/data/{seq}/data_for_reg/0000000{frm_idx}.ply"
            prev_frm_f_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx-1}/f_trans.txt'
            curr_frm_init_pcd_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/data/{seq}/data_for_reg/init_0000000{frm_idx}.ply'
            initialize(curr_frm_pcd_pth, prev_frm_f_pth, curr_frm_init_pcd_write_pth)
            
            # Step2 (filter target pcd)
            icp_dir = '/scratch/github_repos/Fast-Robust-ICP/'
            tgt_pcd_pth = f"{osp.join(icp_dir, f'data/{seq}/data_for_reg/tgt_pcd.ply')}"
            curr_frm_init_pcd_pth = f"{osp.join(icp_dir, f'data/{seq}/data_for_reg/init_0000000{frm_idx}.ply')}"
            tgt_fltrd_pcd_write_pth = f"{osp.join(icp_dir, f'data/{seq}/data_for_reg/tgt_fltrd_0000000{frm_idx}.ply')}"
            filtr_tgt_pcd(tgt_pcd_pth, curr_frm_init_pcd_pth, tgt_fltrd_pcd_write_pth, dist_thresh=0.01)

            # Step3 (run FRICP between src_init and tgt_fltrd)  
            icp_dir = '/scratch/github_repos/Fast-Robust-ICP'
            res_dir = osp.join(icp_dir ,f'res/{seq}/frm{frm_idx}')
            os.makedirs(res_dir, exist_ok=True)
            FRICP(tgt_fltrd_pcd_pth=tgt_fltrd_pcd_write_pth, src_init_pcd_pth=curr_frm_init_pcd_write_pth, res_dir=res_dir)

            # Step4 (save final registered mesh and transformation)
            curr_frm_pcd_pth = f"/scratch/github_repos/Fast-Robust-ICP/data/{seq}/data_for_reg/0000000{frm_idx}.ply"
            curr_frm_o_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/m3trans.txt'
            curr_frm_i_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx-1}/f_trans.txt'
            curr_frm_f_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/f_trans.txt'
            curr_frm_f_pcd_write_pth = f'/scratch/github_repos/Fast-Robust-ICP/res/{seq}/frm{frm_idx}/f_pcd.ply'
            finalize(curr_frm_pcd_pth, curr_frm_o_pth, curr_frm_i_pth, curr_frm_f_write_pth, curr_frm_f_pcd_write_pth)

    elif seq == '20220824144438':
        for frm_idx in range(1, 1140    ):
            start = time.time()
            ICP_DIR = '/scratch/1/user/aswamy/github_repos/Fast-Robust-ICP'
            SAVE_DIR = '/scratch/1/user/aswamy/data/hand-obj'
            # if frm_idx < 156     :
            #     continue
            # bb()
            print(f"\n frm_idx: {frm_idx}")
            # Step1 (create src initialzation)
            curr_frm_pcd_pth = f"/{ICP_DIR}/data/{seq}/data_for_reg/{frm_idx:010d}.ply"
            prev_frm_f_pth = f'/{SAVE_DIR}/{seq}/icp_res/{(frm_idx - 1):010d}/f_trans.txt'
            curr_frm_init_pcd_write_pth = f'{ICP_DIR}/data/{seq}/data_for_reg/init_{frm_idx:010d}.ply'
            initialize(curr_frm_pcd_pth, prev_frm_f_pth, curr_frm_init_pcd_write_pth)
            
            # Step2 (filter target pcd)
            # icp_dir = '/scratch/github_repos/Fast-Robust-ICP/'
            tgt_pcd_pth = f"{osp.join(SAVE_DIR, f'{seq}/gt_mesh/xyz/tgt_pcd.ply')}"
            curr_frm_init_pcd_pth = f"{osp.join(ICP_DIR, f'data/{seq}/data_for_reg/init_{frm_idx:010d}.ply')}"
            tgt_fltrd_pcd_write_pth = f"{osp.join(ICP_DIR, f'data/{seq}/data_for_reg/tgt_fltrd_{frm_idx:010d}.ply')}"
            filtr_tgt_pcd(tgt_pcd_pth, curr_frm_init_pcd_pth, tgt_fltrd_pcd_write_pth, dist_thresh=0.01)

            # Step3 (run FRICP between src_init and tgt_fltrd)  
            # icp_dir = '/scratch/github_repos/Fast-Robust-ICP'
            res_dir = osp.join(SAVE_DIR ,f'{seq}/icp_res/{frm_idx:010d}')
            os.makedirs(res_dir, exist_ok=True)
            FRICP(icp_dir=ICP_DIR, tgt_fltrd_pcd_pth=tgt_fltrd_pcd_write_pth, src_init_pcd_pth=curr_frm_init_pcd_write_pth, res_dir=res_dir)
            # bb()
            # Step4 (save final registered mesh and transformation)
            curr_frm_pcd_pth = f"{ICP_DIR}/data/{seq}/data_for_reg/{frm_idx:010d}.ply"
            curr_frm_o_pth = f'{SAVE_DIR}/{seq}/icp_res/{frm_idx:010d}/m3trans.txt'
            curr_frm_i_pth = f'{SAVE_DIR}/{seq}/icp_res/{(frm_idx - 1):010d}/f_trans.txt'
            curr_frm_f_write_pth = f'{SAVE_DIR}/{seq}/icp_res/{frm_idx:010d}/f_trans.txt'
            curr_frm_f_pcd_write_pth = f'{SAVE_DIR}/{seq}/icp_res/{frm_idx:010d}/f_pcd.ply'
            finalize(curr_frm_pcd_pth, curr_frm_o_pth, curr_frm_i_pth, curr_frm_f_write_pth, curr_frm_f_pcd_write_pth)
            print(f'Frame {frm_idx} Time:: {(time.time() - start):0.4f}s')
            # bb()
    # run metrics script
    metrics_cmd = f"python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/metrics.py --sqn {seq}"
    os.system(metrics_cmd)
    # run img projection script
    proj_gt2frms_cmd = f"python /gfs-ssd/user/aswamy/github_repos/Dataset-Capture/scripts/proj_gt2frms.py --sqn {seq}"
    os.system(proj_gt2frms_cmd)
    print('Done!')
