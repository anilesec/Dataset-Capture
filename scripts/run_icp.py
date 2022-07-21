if True:
    def initialize(curr_frm_pcd_pth, prev_frm_f_pth, curr_frm_init_pcd_write_pth):
        import open3d as o3d
        import numpy as np

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

        return print('curr_frm_init_pcd_write_pth')
    
    #stage2: run ICP on CLI

    # Stage3
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

        return print(f'final pcd:{curr_frm_f_pcd_write_pth}, final trnsfm:{curr_frm_f_write_pth}')\



import open3d as o3d
from scripts.io3d import *
import numpy as np
import os, sys
osp = os.path

for frm_idx in range(460, 900):
    print(f'frame_idx: {frm_idx}')
    initialize(curr_frm_pcd_pth=f'/scratch/github_repos/Fast-Robust-ICP/data/20220705173214/data_for_reg/0000000{frm_idx}.ply',
        prev_frm_f_pth=f'/scratch/github_repos/Fast-Robust-ICP/res/20220705173214/frm{frm_idx-1}/f_trans.txt',
        curr_frm_init_pcd_write_pth=f'/scratch/github_repos/Fast-Robust-ICP/data/20220705173214/data_for_reg/init_0000000{frm_idx}.ply')
    
    base_dir = '/scratch/github_repos/Fast-Robust-ICP'
    print('Running bash commands...')
    print(f"{osp.join(base_dir, 'build/FRICP')} {osp.join(base_dir, 'data/20220705173214/data_for_reg/tgt_pcd.ply')} {osp.join(base_dir, f'data/20220705173214/data_for_reg/init_0000000{frm_idx}.ply')} {osp.join(base_dir, f'res/20220705173214/frm{frm_idx}/')} 3")
    os.system(f"{osp.join(base_dir, 'build/FRICP')} {osp.join(base_dir, 'data/20220705173214/data_for_reg/tgt_pcd.ply')} {osp.join(base_dir, f'data/20220705173214/data_for_reg/init_0000000{frm_idx}.ply')} {osp.join(base_dir, f'res/20220705173214/frm{frm_idx}/')} 3")
    # print(f"{osp.join(base_dir, 'build/FRICP')} {osp.join(base_dir, 'data_for_reg/tgt_pcd.ply')} {osp.join(base_dir, f'data/20220705173214/data_for_reg/init_0000000{frm_idx}.ply')}")

    finalize(curr_frm_pcd_pth=f'/scratch/github_repos/Fast-Robust-ICP/data/20220705173214/data_for_reg/0000000{frm_idx}.ply',
        curr_frm_o_pth=f'/scratch/github_repos/Fast-Robust-ICP/res/20220705173214/frm{frm_idx}/m3trans.txt', 
        curr_frm_i_pth=f'/scratch/github_repos/Fast-Robust-ICP/res/20220705173214/frm{frm_idx-1}/f_trans.txt',
        curr_frm_f_write_pth=f'/scratch/github_repos/Fast-Robust-ICP/res/20220705173214/frm{frm_idx}/f_trans.txt',
        curr_frm_f_pcd_write_pth=f'/scratch/github_repos/Fast-Robust-ICP/res/20220705173214/frm{frm_idx}/f_pcd.ply')

print('Done!')

