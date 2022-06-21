import os, sys, glob 
import numpy as np
osp = os.path
import open3d as o3d
from utils import *

if __name__ == '__main__':

    pcds_dir = ''
    pcds_pths = glob.glob(osp.join(pcds_dir, '*_crop.ply'))

    frm1_pcd = o3d.io.read_point_cloud('/scarat')
    tgt_pcd = 