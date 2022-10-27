import open3d as o3d
import numpy as np
from ipdb import set_trace as bb
import os, glob
osp = os.path
from tqdm import tqdm
from io3d import *

def display_inlier_outlier(cloud, ind):
	inlier_cloud = cloud.select_by_index(ind)
	outlier_cloud = cloud.select_by_index(ind, invert=True)
	print("Showing outliers (red) and inliers (gray): ")
	outlier_cloud.paint_uniform_color([1, 0, 0])
	# inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
	o3d.visualization.draw([inlier_cloud, outlier_cloud])

def rm_outlier(inp_pcds_dir, seq_dir, vox_size, radius, num_nibrs, down_sample=True, start_ind=None, end_ind=None):
    pcds_pths = sorted(glob.glob(osp.join(inp_pcds_dir, '*.ply')))
    for pcdp in tqdm(pcds_pths[start_ind : end_ind]):
        pcd = o3d.io.read_point_cloud(pcdp)

        if down_sample:
            print(f"Downsample the point cloud with a voxel of {vox_size}")
            voxel_down_pcd = pcd.voxel_down_sample(voxel_size=vox_size)
        else:
            voxel_down_pcd = pcd

        print("Radius oulier removal...")
        cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=num_nibrs, radius=radius)
        # cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=400, std_ratio=0.01)

        # display_inlier_outlier(voxel_down_pcd, ind)

        # write inliers
        fn_pcd = osp.join(seq_dir, 'fgnd_xyz_wc_wn_wo_outlr', osp.basename(pcdp))
        os.makedirs(osp.dirname(fn_pcd), exist_ok=True)
        voxel_down_pcd.estimate_normals()
        # bb()
        # o3d.io.write_point_cloud(fname_pcd, voxel_down_pcd.select_by_index(ind))
        write_ply(fn_pcd, verts=np.array(voxel_down_pcd.select_by_index(ind).points), trias=None,
            color=(np.array(voxel_down_pcd.select_by_index(ind).colors)*255).astype(np.uint8),
            normals=np.array(voxel_down_pcd.select_by_index(ind).normals), binary=False)

        # write inliers + outliers
        fn_pcd_w_outlr = osp.join(seq_dir, 'xyz_outlr_viz', osp.basename(pcdp))
        os.makedirs(osp.dirname(fn_pcd_w_outlr), exist_ok=True)

        inlier_cloud = voxel_down_pcd.select_by_index(ind)
        outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        o3d.io.write_point_cloud(fn_pcd_w_outlr, inlier_cloud + outlier_cloud)
        # bb()
    print(f"saved here: {fn_pcd} \n {fn_pcd_w_outlr}")

    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("pcd outlier removal")
    
    parser.add_argument('--inp_pcds_dir', type=str, required=True,
                        help='path to the input pcd .ply files')
    parser.add_argument('--seq_dir', type=str, required=True,
                        help='base path to save outputs(path to seq dir)')
    parser.add_argument('--vox_size', type=float, default=0.0005,
                        help='voxel size for down sampling')
    parser.add_argument('--radius', type=float, default=0.01,
                        help='outlier search radius of sphere')
    parser.add_argument('--num_nibrs', type=int, default=150,
                        help='min no of points to be present in the sphere for inlier cond.')
    parser.add_argument('--start_ind', type=int, default=None,
                        help='start index of a seq')
    parser.add_argument('--end_ind', type=int, default=None,
                        help='End index of a seq')
    args = parser.parse_args()
    print("args:", args)   
    
    rm_outlier(args.inp_pcds_dir, args.seq_dir, args.vox_size, args.radius, args.num_nibrs, down_sample=True, start_ind=args.start_ind, end_ind=args.end_ind)

    print('Done!')
