import os, sys, glob, copy
import numpy as np
osp = os.path
import open3d as o3d
from utils import *
from ipdb import set_trace as bb
import torch, roma

if __name__ == '__main__':

    pcds_dir = '/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr'
    pcds_pths = glob.glob(osp.join(pcds_dir, '*_crop.ply'))

    frm1_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000090_crop.ply')
    tgt_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/tgt_scnr_win.ply')

    frm1_pts = np.array(frm1_pcd.points)
    tgt_pts = np.array(tgt_pcd.points)

    # step1: center the first frame pcd to target pcd 
    frm1_pts_mu = np.mean(frm1_pcd.points, axis=0)
    tgt_pts_mu = np.mean(tgt_pcd.points, axis=0)

    # initial translation for mean alignment of frm1 to tgt
    init_tran =  tgt_pts_mu - frm1_pts_mu
    init_trnsfm1 = np.eye(4)  
    init_trnsfm1[:3, 3] = init_tran

    # center 1st frm to tgt pcds and write
    frm1_pts_cntrd = frm1_pts + init_tran
    frm1_pcd_cntrd = o3d.geometry.PointCloud()
    frm1_pcd_cntrd.points = o3d.pybind.utility.Vector3dVector(frm1_pts_cntrd)
    frm1_pcd_cntrd.colors = o3d.pybind.utility.Vector3dVector(np.array(frm1_pcd.colors))
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm1_pcd_cntrd2tgt.ply', frm1_pcd_cntrd)
    
    bb()
   #  frm1_pcd.transform(init_trnsfm1)
    # step2: Find the init_rot (initial rotation) or init_trnfsm from global registration
    final_global_reg_trnsfm = np.array(
      [[ 0.35001961,  0.06403435, -0.93455116,  0.02084637],
      [-0.91386341, -0.19580574, -0.35568776,  0.0797344 ],
      [-0.20576672,  0.9785498,  -0.0100172,  -0.06569094],
      [ 0.,          0.,          0.,          1.        ]])
    frm1_pcd_cntrd.transform(final_global_reg_trnsfm)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm1_final_init_pcd.ply', frm1_pcd_cntrd)
    
    final_init_trnsfm = final_global_reg_trnsfm @ init_trnsfm1
    print('final_init_trnsfm:', final_init_trnsfm)
    """
    final_init_trnsfm = np.array([[ 0.35001961,  0.06403435, -0.93455116, 0.33575756],
     [-0.91386341, -0.19580574, -0.35568776,  0.42589591],
     [-0.20576672,  0.9785498,  -0.0100172,  -0.01586967],
     [ 0.,          0.,          0.,          1.,        ]])
    """

    frm1_reg_res = np.array(
      [[0.677321, -0.0159027,  -0.735516,  0.0143123],
      [0.120476, 0.988667,  0.0895678,  0.0126468],
      [0.725756,  -0.149278,   0.671561,  0.0711044],
      [0.,          0.,          0.,          1.]])

    frm2_init = frm1_reg_res @ final_init_trnsfm 
    print('frm2_init:', frm2_init)
    """
    frm2_init = np.array([[ 0.40295324 -0.67325338 -0.61996692  0.24662744]
                          [-0.87976771 -0.09822552 -0.46514495  0.47274535]
                           [ 0.25226363  0.73285869 -0.63188692  0.24054812]
                           [ 0.          0.          0.          1.        ]])
    """
    frm2_reg_res = np.array([
      [0.999864, -0.0121181,    0.0111608, -0.000223661],
      [0.0122807,     0.999818,   -0.0146233,  -0.00252517],
      [-0.0109815,    0.0147584,     0.999831,  0.000990775],
      [0.,          0.,          0.,          1.]]
      )
    
    frm3_init = frm2_reg_res @ final_init_trnsfm
    print('frm3_init:', frm3_init)
    """
    frm3_init = np.array([[ 0.35874977,  0.07731983, -0.9302256 ,  0.33015007],
       [-0.90638961, -0.20929334, -0.36695348,  0.42764863],
       [-0.22306285,  0.97479145, -0.00500212, -0.01227779],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm3_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000092_crop.ply')
    frm3_pcd.transform(frm3_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm3_final_init_pcd.ply', frm3_pcd)
    frm3_reg_res = np.array(
      [[  0.667303, -0.0123426,  -0.744684,  0.0142548],
      [  0.108157,    0.99087,  0.0804953, 0.00924337],
      [  0.736892,  -0.134258,   0.662545,  0.0744248],
      [0.,          0.,          0.,          1.]],
    )

    frm4_init = frm3_reg_res @ final_init_trnsfm
    print('frm4_init:', frm4_init)
    """
   array([[ 0.39807977, -0.68356331, -0.61177903,  0.24486805],
       [-0.88422602, -0.10832361, -0.45432492,  0.46628795],
       [ 0.24429041,  0.72180816, -0.64754619,  0.25414756],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm4_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000093_crop.ply')
    frm4_pcd.transform(frm4_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm4_final_init_pcd.ply', frm4_pcd)
    frm4_reg_res = np.array(
      [[    0.999308,  -0.0362487, -0.00835069, -0.00151162 ],
      [  0.0361323,    0.999253,  -0.0137028, -0.00379666],
      [  0.00884116,   0.0133916,    0.999871,  0.00376864],
      [0.,          0.,          0.,          1.]],
    )

    frm5_init = frm4_reg_res @ final_init_trnsfm
    print('frm5_init:', frm5_init)
    """ 
       array([[ 0.38462205,  0.06291618, -0.92092758,  0.31870794],
       [-0.89771416, -0.20675464, -0.38905228,  0.43413026],
       [-0.21488369,  0.97636755, -0.02304165, -0.00342707],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm5_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000094_crop.ply')
    frm5_pcd.transform(frm5_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm5_final_init_pcd.ply', frm5_pcd)
    frm5_reg_res = np.array(
      [[   0.923256,   0.0733892,   -0.377109,  0.00401154],
      [ -0.0279776,    0.991822,    0.124522, -0.00184138],
      [0.383164,   -0.104415,     0.91776,   0.0498135],
      [0.,          0.,          0.,          1.]],
    )

    frm6_init = frm5_reg_res @ final_init_trnsfm
    print('frm6_init:', frm6_init)
    """
      array([[ 0.33368648, -0.32426987, -0.88515603,  0.35124247],
       [-0.94180503, -0.07414499, -0.32787981,  0.40920174],
       [ 0.0406915 ,  0.94305458, -0.33014061,  0.11942924],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm6_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000095_crop.ply')
    frm6_pcd.transform(frm6_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm6_final_init_pcd.ply', frm6_pcd)
    frm6_reg_res = np.array(
      [[0.999903,  -0.0132282,  0.00435146, -0.00583715],
      [0.013111,    0.999578,   0.0259381, -0.00584217],
      [-0.00469274,  -0.0258785,    0.999654,  0.00493773],
      [0,           0,           0,           1]],
    )


    frm7_init = frm6_reg_res @ final_init_trnsfm
    print('frm7_init:', frm7_init)
    """
      array([[ 0.36117904,  0.07087642, -0.92979899,  0.32418495],
       [-0.91422585, -0.16950183, -0.36805039,  0.4238645 ],
       [-0.18368866,  0.98297788,  0.00357654, -0.02352362],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm7_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000096_crop.ply')
    frm7_pcd.transform(frm7_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm7_final_init_pcd.ply', frm7_pcd)
    frm7_reg_res = np.array(
      [[  0.916632,  0.0912634,  -0.389174, 0.00197435],
      [   -0.035007,   0.988175,    0.14928, -0.0033702],
      [  0.398195,  -0.123211,   0.908988,  0.0569448],
      [0,           0,           0,           1]],
    )


    frm8_init = frm7_reg_res @ final_init_trnsfm
    print('frm8_init:', frm8_init)
    """
      array([[ 0.31751595, -0.3400001 , -0.88520234,  0.35478524],
       [-0.94602697, -0.04965407, -0.32026129,  0.40336661],
       [ 0.0649346 ,  0.9391136 , -0.33741447,  0.12374138],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm8_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000097_crop.ply')
    frm8_pcd.transform(frm8_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm8_final_init_pcd.ply', frm8_pcd)
    frm8_reg_res = np.array(
      [[   0.999827,  -0.0168063,  0.00794756, -0.00619642],
      [0.0166715,    0.999721,    0.016731, -0.00753726],
      [-0.00822653,  -0.0165956,    0.999828,  0.00312071],
      [0,           0,           0,           1]],
    )

    frm9_init = frm8_reg_res @ final_init_trnsfm
    print('frm9_init:', frm9_init)
    """
      array([[ 0.36368238,  0.07509113, -0.9284913 ,  0.32221919],
       [-0.91121577, -0.17831144, -0.37133649,  0.4235719 ],
       [-0.19344466,  0.98110422,  0.00357549, -0.02257634],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm9_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/0000000098_crop.ply')
    frm9_pcd.transform(frm9_init)
    write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm9_final_init_pcd.ply', frm9_pcd)
    bb()
    frm9_reg_res = np.array(
      [[   0.999827,  -0.0168063,  0.00794756, -0.00619642],
      [0.0166715,    0.999721,    0.016731, -0.00753726],
      [-0.00822653,  -0.0165956,    0.999828,  0.00312071],
      [0,           0,           0,           1]],
    )


    bb()
    print('Done!')







