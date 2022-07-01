import open3d as o3d
import numpy as np
import copy
from copy import deepcopy as dcopy
import time
import os
import sys
from utils import *
from ipdb import set_trace as bb
import roma, torch
"""
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 5
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 10
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, src, tgt):
    print(":: Load two point clouds and disturb initial pose")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    src.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(src, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(tgt, voxel_size)
    return src, tgt, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2.0
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if True:
    src = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_pcd_cntrd2tgt.ply')
    tgt = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/tgt_scnr_win.ply')

    voxel_size = 0.005  # means 0.5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, src, tgt)

    ITERATIONS = 500
    result = o3d.pipelines.registration.RegistrationResult()
    for iter in range(ITERATIONS):
        print(f'Iter: {iter}')
        curr_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh, voxel_size)
        
        if (curr_ransac.fitness > result.fitness) or \
            (curr_ransac.fitness == result.fitness and
            curr_ransac.inlier_rmse < result.inlier_rmse):
            print('Better solution found')
            result = curr_ransac

    print(f"result: {result}")
    print('Recovered transform:')
    print(result.transformation)
    bb()
    src.transform(result.transformation)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_global_reg_500iter.ply', src)

    src_inp = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_pcd_cntrd2tgt.ply')
    src_trnsfmd = src

    src_inp_pts = torch.tensor(np.array(src_inp.points))
    src_trnsfmd_pts = torch.tensor(np.array(src_trnsfmd.points))

    R, t = roma.rigid_points_registration(src_inp_pts, src_trnsfmd_pts)
    final_global_reg_transfm = np.eye(4)
    final_global_reg_transfm[:3, :3] = np.array(R)
    final_global_reg_transfm[:3, 3] = np.array(t)
    bb()
    print(final_global_reg_transfm)
"""

class GlobalRegistration:
    def __init__(self, src_pcd_o3d, tgt_pcd_o3d, voxel_size=0.005, iter=500):
        """
        src_pcd: o3d source(frame's) pcd
        tg_pcd: o3d target(scanner's) pcd
        voxel_size: downsampling voxel size for pcd feature computation
        iter: number of ransac iterations to find the better global registration 
        """
        self.src_pcd = src_pcd_o3d
        self.tgt_pcd = tgt_pcd_o3d
        self.voxel_size = voxel_size
        self.iter = iter

    def preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 5
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 10
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh

    def prepare_dataset(self, voxel_size, src, tgt):
        print(":: Load two point clouds and disturb initial pose.")
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        src.transform(trans_init)
        # draw_registration_result(source, target, np.identity(4))
        source_down, source_fpfh = self.preprocess_point_cloud(src, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(tgt, voxel_size)

        return src, tgt, source_down, target_down, source_fpfh, target_fpfh
    
    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 2.0
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        return result

    def center_align_src2tgt(self, src_pcd, tgt_pcd):
        src_pts = np.array(src_pcd.points)
        tgt_pts = np.array(tgt_pcd.points)

        # center the first frame pcd to target pcd 
        src_pts_mu = np.mean(src_pcd.points, axis=0)
        tgt_pts_mu = np.mean(tgt_pcd.points, axis=0)

        # initial translation for mean alignment of src to tgt
        init_tran =  tgt_pts_mu - src_pts_mu
        init_trans_trnsfm = np.eye(4)
        init_trans_trnsfm[:3, 3] = init_tran

        # center 1st frm to tgt pcds and write
        src_pts_cntrd = src_pts + init_tran
        src_pcd_cntrd = o3d.geometry.PointCloud()
        src_pcd_cntrd.points = o3d.pybind.utility.Vector3dVector(src_pts_cntrd)
        src_pcd_cntrd.colors = o3d.pybind.utility.Vector3dVector(np.array(src_pcd.colors))
        src_pcd_cntrd.estimate_normals()

        return src_pcd_cntrd, init_trans_trnsfm

    def get_initial_transfm(self):
        # get center aligned translation 
        src_tranl_trnsfmd, tranl_trnsfm = self.center_align_src2tgt(
            src_pcd=dcopy(self.src_pcd), tgt_pcd=dcopy(self.tgt_pcd)
            )

        # get global reg trnsfm from "registration_ransac_based_on_feature_matching"
        source, target, source_down, \
        target_down, source_fpfh, target_fpfh = self.prepare_dataset(
            voxel_size=self.voxel_size, src=dcopy(src_tranl_trnsfmd), tgt=dcopy(self.tgt_pcd)
        )

        # loop to find better soln
        result = o3d.pipelines.registration.RegistrationResult()
        for iter in range(self.iter):
            print(f'Iter: {iter}')
            curr_ransac = self.execute_global_registration(
                dcopy(source_down), dcopy(target_down), dcopy(source_fpfh),
                dcopy(target_fpfh), self.voxel_size
                )
            
            if (curr_ransac.fitness > result.fitness) or \
                (curr_ransac.fitness == result.fitness and
                curr_ransac.inlier_rmse < result.inlier_rmse):
                print('Better solution found')
                result = curr_ransac

        print(f"Result: {result}")
        print('Recovered transform:')
        print(result.transformation)

        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        final_trnsfm = result.transformation @ trans_init @ tranl_trnsfm

        return dcopy(self.src_pcd).transform(final_trnsfm), final_trnsfm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("global registration for inital transformation")

    parser.add_argument('--src_pcd_pth', type=str, default='/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000090_crop.ply',
                        help='path to the source .ply path')
    parser.add_argument('--tgt_pcd_pth', type=str, default='/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/tgt_scnr_win.ply',
                        help='path to the target .ply path')
    parser.add_argument('--vox_size', type=float, default=0.005,
                        help='downsampling voxel size for pcd feature computation')
    parser.add_argument('--iter', type=int, default=500,
                        help='number of ransac iterations to find the better global registration')

    args = parser.parse_args()

    src_pcd_o3d = read_o3d_pcd(args.src_pcd_pth)
    tgt_pcd_o3d = read_o3d_pcd(args.tgt_pcd_pth)

    greg = GlobalRegistration(src_pcd_o3d, tgt_pcd_o3d, voxel_size=0.005, iter=500)
    pcd, T = greg.get_initial_transfm()
    
    bb()
    print('Done!')
