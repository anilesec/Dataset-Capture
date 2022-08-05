import open3d as o3d
import numpy as np
import os
osp = os.path
read_o3d_pcd = o3d.io.read_point_cloud
import polyscope as ps
from ipdb import set_trace as bb

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("polyscoe viz")
    parser.add_argument('--sqn', type=str, default=20220705173214,
                        help='seq name')
    parser.add_argument('--start_ind', type=int, required=True,
                        help='start index of a seq')
    parser.add_argument('--end_ind', type=int, required=True,
                        help='End index of a seq')
    args = parser.parse_args()

    print("args:", args)     

    base_dir = '/Users/aswamy/My_drive/github_repos/L515_seqs/20220705173214/icp_res/'
    tgt_pth = '/Users/aswamy/My_drive/github_repos/L515_seqs/20220705173214/gt_mesh/xyz/tgt_pcd.ply'
    tgt_pcd = read_o3d_pcd(tgt_pth)

    ps.init()        
    ps_cloud = ps.register_point_cloud(f"GT", np.array(tgt_pcd.points), enabled=True, radius=0.003)
    ps_cloud.add_color_quantity("rgb", np.array(tgt_pcd.colors), enabled=True)

    for frm_idx in range(args.start_ind, args.end_ind):
        src_pth = osp.join(base_dir, f'frm{frm_idx}/f_pcd.ply')
        src_pcd = read_o3d_pcd(src_pth)
        src_dists = np.array(src_pcd.compute_point_cloud_distance(tgt_pcd))
        ps_src = ps.register_point_cloud(f"frm{frm_idx}", np.array(src_pcd.points), enabled=True, radius=0.003)
        ps_src.add_color_quantity("rgb", np.array(src_pcd.colors), enabled=True)
        ps_src.add_scalar_quantity("dist", src_dists, enabled=True, cmap='reds')
    ps.show()
    print('Done')

    