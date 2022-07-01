import os, sys, glob, copy
import numpy as np

from io3d import write_ply
osp = os.path
import open3d as o3d
from utils import *
from ipdb import set_trace as bb
import torch, roma

if __name__ == '__main__':

    pcds_dir = '/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr'
    pcds_pths = glob.glob(osp.join(pcds_dir, '*_crop.ply'))

    frm1_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000090_crop.ply')
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
    frm1_pcd_cntrd.estimate_normals()
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_pcd_cntrd2tgt.ply', frm1_pcd_cntrd)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_pcd_cntrd2tgt.ply',
              verts=np.array(frm1_pcd_cntrd.points), color=np.array(frm1_pcd_cntrd.colors)*255, normals=np.array(frm1_pcd_cntrd.normals))
    
    bb()
   #  frm1_pcd.transform(init_trnsfm1)
    # step2: Find the init_rot (initial rotation) or init_trnfsm from global registration
    # final_global_reg_trnsfm = np.array(
    #   [[ 0.35001961,  0.06403435, -0.93455116,  0.02084637],
    #   [-0.91386341, -0.19580574, -0.35568776,  0.0797344 ],
    #   [-0.20576672,  0.9785498,  -0.0100172,  -0.06569094],
    #   [ 0.,          0.,          0.,          1.        ]])
    final_global_reg_trnsfm = np.array(
      [[ 0.27342721, -0.62018529, -0.73526034,  0.07331152],
       [-0.91267567,  0.07410396, -0.40191009,  0.06495868],
       [ 0.30374443,  0.78094738, -0.54576599,  0.01866109],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )
    frm1_pcd_cntrd.transform(final_global_reg_trnsfm)
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_final_init_pcd.ply', frm1_pcd_cntrd)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm1_final_init_pcd.ply',
              verts=np.array(frm1_pcd_cntrd.points), color=np.array(frm1_pcd_cntrd.colors)*255, normals=np.array(frm1_pcd_cntrd.normals))
    
    final_init_trnsfm = final_global_reg_trnsfm @ init_trnsfm1
    print('final_init_trnsfm:', final_init_trnsfm)
    """
    final_init_trnsfm = np.array([[ 0.35001961,  0.06403435, -0.93455116, 0.33575756],
     [-0.91386341, -0.19580574, -0.35568776,  0.42589591],
     [-0.20576672,  0.9785498,  -0.0100172,  -0.01586967],
     [ 0.,          0.,          0.,          1.,        ]])
    """

    frm1_reg_res = np.array(
      [[ 0.983707  , -0.167356  , -0.0656716 ,  0.0153381 ],
       [ 0.161278  ,  0.982891  , -0.0889616 ,  0.014014  ],
       [ 0.0794363 ,  0.0769207 ,  0.993868  ,  0.00101522],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )
    
    frm2_init = frm1_reg_res @ final_init_trnsfm 
    print('frm2_init:', frm2_init)
    """
    frm2_init = np.array([[ 0.40295324 -0.67325338 -0.61996692  0.24662744]
                          [-0.87976771 -0.09822552 -0.46514495  0.47274535]
                           [ 0.25226363  0.73285869 -0.63188692  0.24054812]
                           [ 0.          0.          0.          1.        ]])
    """
    frm2_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000091_crop.ply')
    frm2_pcd.transform(frm2_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm2_final_init_pcd.ply',
              verts=np.array(frm2_pcd.points), color=np.array(frm2_pcd.colors)*255, normals=np.array(frm2_pcd.normals))

    frm2_reg_res = np.array(
      [[ 9.99890e-01, -1.10645e-02,  9.91398e-03, -3.16182e-04],
       [ 1.12084e-02,  9.99831e-01, -1.45805e-02, -2.58106e-03],
       [-9.75098e-03,  1.46900e-02,  9.99845e-01,  1.07921e-03],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]]
       )
    
    frm3_init = frm2_reg_res @ final_init_trnsfm
    print('frm3_init:', frm3_init)
    """
    frm3_init = np.array([[ 0.35874977,  0.07731983, -0.9302256 ,  0.33015007],
       [-0.90638961, -0.20929334, -0.36695348,  0.42764863],
       [-0.22306285,  0.97479145, -0.00500212, -0.01227779],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm3_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000092_crop.ply')
    frm3_pcd.transform(frm3_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm3_final_init_pcd.ply',
              verts=np.array(frm3_pcd.points), color=np.array(frm3_pcd.colors)*255, normals=np.array(frm3_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm3_final_init_pcd.ply', frm3_pcd)

    frm3_reg_res = np.array(
      [[ 0.981934  , -0.173387  , -0.0757796 ,  0.0145006 ],
       [ 0.165953  ,  0.981511  , -0.095367  ,  0.0122701 ],
       [ 0.090914  ,  0.0810683 ,  0.992554  ,  0.00419817],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )

    frm4_init = frm3_reg_res @ final_init_trnsfm
    print('frm4_init:', frm4_init)
    """
   array([[ 0.39807977, -0.68356331, -0.61177903,  0.24486805],
       [-0.88422602, -0.10832361, -0.45432492,  0.46628795],
       [ 0.24429041,  0.72180816, -0.64754619,  0.25414756],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm4_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000093_crop.ply')
    frm4_pcd.transform(frm4_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm4_final_init_pcd.ply',
              verts=np.array(frm4_pcd.points), color=np.array(frm4_pcd.colors)*255, normals=np.array(frm4_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm4_final_init_pcd.ply', frm4_pcd)

    frm4_reg_res = np.array(
      [[ 9.99516e-01, -2.86418e-02, -1.21245e-02,  9.91307e-06],
       [ 2.83518e-02,  9.99323e-01, -2.34499e-02, -4.10541e-03],
       [ 1.27879e-02,  2.30948e-02,  9.99651e-01,  3.83875e-03],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]]
       )

    frm5_init = frm4_reg_res @ final_init_trnsfm
    print('frm5_init:', frm5_init)
    """ 
       array([[ 0.38462205,  0.06291618, -0.92092758,  0.31870794],
       [-0.89771416, -0.20675464, -0.38905228,  0.43413026],
       [-0.21488369,  0.97636755, -0.02304165, -0.00342707],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm5_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000094_crop.ply')
    frm5_pcd.transform(frm5_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm5_final_init_pcd.ply',
              verts=np.array(frm5_pcd.points), color=np.array(frm5_pcd.colors)*255, normals=np.array(frm5_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm5_final_init_pcd.ply', frm5_pcd)

    frm5_reg_res = np.array(
      [[ 0.979369  , -0.190043  , -0.0687091 ,  0.012598  ],
       [ 0.183591  ,  0.97883   , -0.0904742 ,  0.0125379 ],
       [ 0.0844485 ,  0.0759932 ,  0.993526  ,  0.00775137],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )

    frm6_init = frm5_reg_res @ final_init_trnsfm
    print('frm6_init:', frm6_init)
    """
      array([[ 0.33368648, -0.32426987, -0.88515603,  0.35124247],
       [-0.94180503, -0.07414499, -0.32787981,  0.40920174],
       [ 0.0406915 ,  0.94305458, -0.33014061,  0.11942924],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm6_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000095_crop.ply')
    frm6_pcd.transform(frm6_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm6_final_init_pcd.ply',
              verts=np.array(frm6_pcd.points), color=np.array(frm6_pcd.colors)*255, normals=np.array(frm6_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm6_final_init_pcd.ply', frm6_pcd)
    
    frm6_reg_res = np.array([[ 0.999627  , -0.015834  , -0.0222699 , -0.00120821],
       [ 0.015313  ,  0.99961   , -0.0233729 , -0.00720199],
       [ 0.0226312 ,  0.0230232 ,  0.999479  ,  0.00533083],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )


    frm7_init = frm6_reg_res @ final_init_trnsfm
    print('frm7_init:', frm7_init)
    """
      array([[ 0.36117904,  0.07087642, -0.92979899,  0.32418495],
       [-0.91422585, -0.16950183, -0.36805039,  0.4238645 ],
       [-0.18368866,  0.98297788,  0.00357654, -0.02352362],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm7_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000096_crop.ply')
    frm7_pcd.transform(frm7_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm7_final_init_pcd.ply',
              verts=np.array(frm7_pcd.points), color=np.array(frm7_pcd.colors)*255, normals=np.array(frm7_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm7_final_init_pcd.ply', frm7_pcd)

    frm7_reg_res = np.array(
      [[ 0.974219 , -0.213222 , -0.0737127,  0.0122366],
       [ 0.205805 ,  0.973794 , -0.0968015,  0.0131234],
       [ 0.0924211,  0.0791354,  0.99257  ,  0.0106401],
       [ 0.       ,  0.       ,  0.       ,  1.       ]]
       )

    frm8_init = frm7_reg_res @ final_init_trnsfm
    print('frm8_init:', frm8_init)
    """
      array([[ 0.31751595, -0.3400001 , -0.88520234,  0.35478524],
       [-0.94602697, -0.04965407, -0.32026129,  0.40336661],
       [ 0.0649346 ,  0.9391136 , -0.33741447,  0.12374138],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm8_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000097_crop.ply')
    frm8_pcd.transform(frm8_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm8_final_init_pcd.ply',
              verts=np.array(frm8_pcd.points), color=np.array(frm8_pcd.colors)*255, normals=np.array(frm8_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm8_final_init_pcd.ply', frm8_pcd)

    frm8_reg_res = np.array([[ 9.99863e-01,  6.84240e-04, -1.65561e-02, -2.21014e-03],
       [-9.79712e-04,  9.99840e-01, -1.78452e-02, -1.09712e-02],
       [ 1.65412e-02,  1.78589e-02,  9.99704e-01,  6.17600e-03],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])

    frm9_init = frm8_reg_res @ final_init_trnsfm
    print('frm9_init:', frm9_init)
    """
      array([[ 0.36368238,  0.07509113, -0.9284913 ,  0.32221919],
       [-0.91121577, -0.17831144, -0.37133649,  0.4235719 ],
       [-0.19344466,  0.98110422,  0.00357549, -0.02257634],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm9_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000098_crop.ply')
    frm9_pcd.transform(frm9_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm9_final_init_pcd.ply',
              verts=np.array(frm9_pcd.points), color=np.array(frm9_pcd.colors)*255, normals=np.array(frm9_pcd.normals))
    # write_o3d_pcd('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/frm9_final_init_pcd.ply', frm9_pcd)

    frm9_reg_res = np.array([[ 0.976129  , -0.206105  , -0.0685087 ,  0.00832399],
       [ 0.199932  ,  0.975912  , -0.0873058 ,  0.0132036 ],
       [ 0.0848527 ,  0.0715246 ,  0.993823  ,  0.013328  ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
       )

    frm10_init = frm9_reg_res @ final_init_trnsfm
    print('frm10_init:', frm10_init)
    """
      array([[ 0.36368238,  0.07509113, -0.9284913 ,  0.32221919],
       [-0.91121577, -0.17831144, -0.37133649,  0.4235719 ],
       [-0.19344466,  0.98110422,  0.00357549, -0.02257634],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    frm10_pcd = o3d.io.read_point_cloud('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/0000000099_crop.ply')
    frm10_pcd.transform(frm10_init)
    write_ply('/scratch/github_repos/Fast-Robust-ICP/data/20220614171547_grn_dstr/outlr_fltr/frm10_final_init_pcd.ply',
              verts=np.array(frm10_pcd.points), color=np.array(frm10_pcd.colors)*255, normals=np.array(frm10_pcd.normals))

    bb()
    frm10_reg_res = np.array([[ 9.99650e-01, -5.41223e-03, -2.59078e-02, -8.00775e-04],
       [ 4.69802e-03,  9.99609e-01, -2.75491e-02, -1.18831e-02],
       [ 2.60468e-02,  2.74177e-02,  9.99285e-01,  5.74482e-03],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])
       
    print('Done!')







