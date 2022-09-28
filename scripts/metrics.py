import open3d as o3d
import numpy as np
import os, sys, glob
# import polyscope as ps
from ipdb import set_trace as bb
from tqdm import tqdm
import matplotlib.pyplot as plt
osp = os.path
read_o3d_pcd = o3d.io.read_point_cloud

class Metrics:
    def __init__(self, tgt_pcd_pth, all_src_pcds_pths, all_algnd_pcds_pths, all_src2tgt_poses_pths):
        self.tgt_pcd_pth = tgt_pcd_pth
        self.all_src_pcds_pths = all_src_pcds_pths
        self.all_algnd_pcds_pths = all_algnd_pcds_pths
        self.all_src2tgt_poses_pths = all_src2tgt_poses_pths

    def compute_allsrcs2tgt_metrics(self):
        tgt_pcd = read_o3d_pcd(self.tgt_pcd_pth)
        all_algnd_pcd_dist_dicts = []
        for algnd_pcdp in tqdm(self.all_algnd_pcds_pths):
            algnd_pcd = read_o3d_pcd(algnd_pcdp)
            algnd_pcd_dist = np.array(algnd_pcd.compute_point_cloud_distance(tgt_pcd))
            dist_dict = {
                'dist': algnd_pcd_dist
            }
            all_algnd_pcd_dist_dicts.append(dist_dict)

        return all_algnd_pcd_dist_dicts 
    
    def compute_reproj_err(self):
        raise NotImplementedError("yet to be implemented")

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("metrics for a given sequence")
    parser.add_argument('--sqn', type=str, required=True,
                        help='seq name')
    args = parser.parse_args()
    print("args:", args)     

    ICP_DIR = '/scratch/1/user/aswamy/github_repos/Fast-Robust-ICP'
    RES_DIR = '/scratch/1/user/aswamy/data/hand-obj'
    tgt_pcd_pth = f'{ICP_DIR}/data/{args.sqn}/data_for_reg/tgt_pcd.ply'
    all_algnd_pcds_pths = sorted(glob.glob(f'{RES_DIR}/{args.sqn}/icp_res/*/f_pcd.ply'))
    # bb()
    pcd_metrics = Metrics(tgt_pcd_pth, None, all_algnd_pcds_pths, None)

    res_dicts = pcd_metrics.compute_allsrcs2tgt_metrics()

    all_frms_dist_mu = []
    for rdict in res_dicts:
        all_frms_dist_mu.append(rdict['dist'].mean())
    all_frms_dist_mu = np.array(all_frms_dist_mu)

    # print stats
    print(f'Mean: {np.round(all_frms_dist_mu.mean(), 5)}')
    print(f'Min: {np.round(all_frms_dist_mu.min(), 5)}')
    print(f'Max: {np.round(all_frms_dist_mu.max(), 5)}')
    print(f'Std: {np.round(all_frms_dist_mu.std(), 5)}')
    print(f'Percentile 25th: {np.round(np.percentile(all_frms_dist_mu, 25), 5)}')
    print(f'Percentile 50th: {np.round(np.percentile(all_frms_dist_mu, 50), 5)}')
    print(f'Percentile 75th: {np.round(np.percentile(all_frms_dist_mu, 75), 5)}')
    print(f'Percentile 90th: {np.round(np.percentile(all_frms_dist_mu, 90), 5)}')
    print(f'Percentile 95th: {np.round(np.percentile(all_frms_dist_mu, 95), 5)}')
    # plot
    plt.figure(figsize=(20, 10))
    plt.plot(all_frms_dist_mu, 'go-', markersize=1, linewidth=1)
    plt.xlabel('Frame No.')
    plt.ylabel('Lidar to GT dist(m)')
    plt.title('Distance error of each frame')
    plt.savefig(f'/gfs-ssd/user/aswamy/github_repos/Dataset-Capture/imgs/dist_err_{args.sqn}.png')
    # bb()
    print('Done!')


        


