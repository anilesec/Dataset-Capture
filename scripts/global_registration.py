import open3d as o3d
import numpy as np
import copy
import time
import os
import sys
from utils import *
from ipdb import set_trace as bb
import roma, torch

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
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
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


src = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm1_pcd_cntrd2tgt.ply')
tgt = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/tgt_scnr_win.ply')

bb()
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
write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm1_global_reg_500iter.ply', src)

src_inp = read_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm1_pcd_cntrd2tgt.ply')
src_trnsfmd = src

src_inp_pts = torch.tensor(np.array(src_inp.points))
src_trnsfmd_pts = torch.tensor(np.array(src_trnsfmd.points))

R, t = roma.rigid_points_registration(src_inp_pts, src_trnsfmd_pts)
final_global_reg_transfm = np.eye(4)
final_global_reg_transfm[:3, :3] = np.array(R)
final_global_reg_transfm[:3, 3] = np.array(t)
bb()
print(final_global_reg_transfm)

