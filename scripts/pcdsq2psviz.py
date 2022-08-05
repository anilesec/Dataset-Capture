import polyscope as ps
import open3d as o3d 
import numpy  as np
import os, sys, glob
osp = os.path
from ipdb import set_trace as bb

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("polyscoe viz")
    parser.add_argument('--pcds_dir', type=str, default=None,
                        help='seq name')
    parser.add_argument('--start_ind', type=int, default=None,
                        help='start index of a seq')
    parser.add_argument('--end_ind', type=int, default=None,
                        help='End index of a seq')
    args = parser.parse_args()

    print("args:", args)     
    
    # Initialize polyscope
    ps.init()
    pcd_pths = sorted(glob.glob(osp.join(args.pcds_dir, '*.ply')))
    for idx, pcdp in enumerate(pcd_pths[args.start_ind : args.end_ind]):
        pcd = o3d.io.read_point_cloud(pcdp)
        pts = np.array(pcd.points)
        clrs = np.array(pcd.colors)
        print(f'{osp.basename(pcdp)}')
        ps_cloud = ps.register_point_cloud(f"{osp.basename(pcdp)}", pts, enabled=True, radius=0.001)
        ps_cloud.add_color_quantity("rgb", clrs, enabled=True)
    # show
    ps.show()

# for idx, pcdp in enumerate(pcd_pths[0 : 10]):
#     pcd = o3d.io.read_point_cloud(pcdp)
#     pts = np.array(pcd.points)
#     clrs = np.array(pcd.colors)
#     print(f'{osp.basename(pcdp)}')
#     ps_cloud = ps.register_point_cloud(f"{osp.basename(pcdp)}", pts, enabled=True, radius=0.001)
#     ps_cloud.add_color_quantity("rgb", clrs, enabled=True)
# # show
# ps.show()


# python scripts/