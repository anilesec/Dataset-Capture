from copy import deepcopy
from lib2to3.pgen2.pgen import DFAState
from re import L
from xml.etree.ElementTree import TreeBuilder
from evaluation.eval_utils import RES_DIR
from evaluation.viz_utils import saveas_json
import numpy as np
import matplotlib.pyplot as plt
import glob, copy
import os
osp = os.path
from tqdm import tqdm
import ipdb
bb = ipdb.set_trace
import cv2
# import open3d as o3d

def imgs2vid_ffmpeg(imgs_dir, file_pth, ext='png',  frm_rate=10):
    import os
    print(f"ffmpeg creating video...")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate {frm_rate} -pattern_type glob -i '{imgs_dir}/*.{ext}' -c:v " f"libx264 -vf fps=30 -pix_fmt yuv420p {file_pth} -y "
    os.system(cmd)
   # os.system(f"rm {imgs_dir}/*.jpg")
    return print(f"video saved here: {file_pth}")

def read_o3d_pcd(file_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)

        return pcd

def create_target_pcd(src_pcd_pth, tran=[0., 0., 0.], rot=[0., 0., 0.]):
        "creates a target point cloud by transforming source pointcloud src_pcd using rot(in deg along x,y and z axis) and tran"

        pcd = o3d.io.read_point_cloud(src_pcd_pth)
        from scipy.spatial.transform import Rotation as R
        r = R.from_rotvec(np.array(rot), degrees=True)
        trnsfm = np.eye(4)
        trnsfm[:3, :3] = r.as_matrix()
        trnsfm[:3, 3] = tran

        pcd.transform(trnsfm)

        return pcd, trnsfm

 
def write_o3d_pcd(file_path, pcd_o3d):
        import open3d as o3d
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        o3d.io.write_point_cloud(file_path, pcd_o3d, write_ascii=True)

        return print(f"saved: {file_path}")

# Convert Midas depth into masks 
if False:
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser("")
        parser.add_argument('--sqn', type=str, required=True,
                            help='seq name')

        args = parser.parse_args()

        print("args:", args)
        seq = args.sqn
        base_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}'
        imgs_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}/rgb'
        dimgs_dir = f'/scratch/1/user/aswamy/data/hand-obj/{seq}/midas_depth'
        dimgs_pths = sorted(glob.glob(osp.join(dimgs_dir, '*.png')))
        imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.png')))
        for ithresh in np.linspace(0.1, 0.6, 6):
            for dimp, imp in tqdm(zip(dimgs_pths, imgs_pths)):
                # get mask
                dimg = plt.imread(dimp)
                mask = (dimg > ithresh).astype(np.uint8)
                save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}')
                fname_mask = osp.join(save_dir, osp.basename(dimp))
                os.makedirs(osp.dirname(fname_mask), exist_ok=True)
                plt.imsave(fname_mask, mask)

                # get masked image
                img = plt.imread(imp)
                mskd_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
                save_dir = osp.join(base_dir, f'{osp.basename(dimgs_dir)}_masks_th_{ithresh:.1f}_mskd_img')
                fname_mskd_img = osp.join(save_dir, osp.basename(imp))
                os.makedirs(osp.dirname(fname_mskd_img), exist_ok=True)
                plt.imsave(fname_mskd_img, mskd_img)
                # bb()

# Convert RGBD data to colored pcds(.ply) 
if False:
    # get colored pointclouds from RGB and depth info
    base_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud'
    imgs_dir =  '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud'
    pcds_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud'
    depth_save_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/image'

    imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.bgr.npz')))
    pcds_pths = sorted(glob.glob(osp.join(pcds_dir, '*.xyz.npz')))
    # bb()
    for (imp, pcdp) in tqdm(zip(imgs_pths, pcds_pths)):
        im = dict(np.load(imp))['arr_0']
        pcd = dict(np.load(pcdp))['arr_0']
        # bb()
        if False: # convert pcd to depth maps
            # replace nan values with zero/max depth values
            points = np.nan_to_num(pcd)
            # x_max = points[:, :, 0].max()
            # y_max = points[:, :, 1].max()
            z_max = points[:, :, 2].max()
            # bb()
            # points[:, :, 0] = np.nan_to_num(pcd[:, :, 0], nan=x_max)
            # points[:, :, 1] = np.nan_to_num(pcd[:, :, 1], nan=y_max)
            points[:, :, 2] = np.nan_to_num(pcd[:, :, 2], nan=z_max)
            depth = points[:, :, 2]
            depth = (1 - (depth / (depth.max() - depth.min()))) * 255.
            fname_depth = osp.join(depth_save_dir, osp.basename(imp).replace('.bgr.npz', '_depth.jpg'))
            os.makedirs(osp.dirname(fname_depth), exist_ok=True)
            cv2.imwrite(fname_depth, depth.astype(np.uint8))
            # bb()
        if True: # ply file computation
            # create open3d point cloud
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.cpu.pybind.utility.Vector3dVector(pcd.reshape(-1, 3) * 1000.)
            pcd_o3d.colors = o3d.cpu.pybind.utility.Vector3dVector(im[:, :, ::-1].reshape(-1, 3) / 255.)
            pcd_o3d.remove_non_finite_points()
            # pcd_o3d.scale(1000, center=pcd_o3d.get_center())
            # bb()
            fname = osp.join(base_dir, osp.basename(imp).replace('bgr.npz', 'ply'))
            os.makedirs(osp.dirname(fname), exist_ok=True)
            print(fname)
            o3d.io.write_point_cloud(fname, pcd_o3d, write_ascii=True)
    print('Done!!')


# select the largest companent and writing it back
if False:
    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser("")
        parser.add_argument('--sqn', type=str, required=True,
                            help='seq name')

        parser.add_argument('--start_ind', type=int, default=None,
                        help='start index of a seq')
        parser.add_argument('--end_ind', type=int, default=None,
                        help='End index of a seq')

        args = parser.parse_args()

        def getLargestCC(segmentation):
            from skimage.measure import label   
            "get largest connected components"
            labels = label(segmentation)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

            return largestCC

        imgs_pths = sorted(glob.glob(f'/scratch/1/user/aswamy/data/hand-obj/{args.sqn}/slvless_img/*.png'))

        for imp in tqdm(imgs_pths[args.start_ind : args.end_ind]):
            im = cv2.imread(imp)
            im_norm = 255. * im / (np.sum(im,axis=-1)[:, :, None] + 1e-6)
            im_mask = (im_norm.sum(2) > 0.0).astype(np.uint8)
            # bb()
            lcc = getLargestCC(im_mask)
            im_lcc = cv2.bitwise_and(im, im, mask=lcc.astype(np.uint8))
            cv2.imwrite(imp, im_lcc)
            # bb()




            


# Segmenting hand pcds from background pcds
if False:
    # get colored pointclouds from RGB and depth info
    base_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud_seg'
    imgs_dir =  '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud'
    pcds_dir = '/Users/aswamy/My_drive/github_repos/jerome-yves/20220608165458_green_duster/pointcloud'

    imgs_pths = sorted(glob.glob(osp.join(imgs_dir, '*.bgr.npz')))
    pcds_pths = sorted(glob.glob(osp.join(pcds_dir, '*.xyz.npz')))

    for (imp, pcdp) in tqdm(zip(imgs_pths, pcds_pths)):
        im = dict(np.load(imp))['arr_0']
        raw_pts = dict(np.load(pcdp))['arr_0']

        pts = raw_pts.reshape(-1, 3)
        pts_colors = im[:, :, ::-1].reshape(-1, 3) / 255.
        pts_depth = np.linalg.norm(raw_pts.reshape(-1, 3), axis=1)
        tmp_depth = np.nan_to_num(pts_depth)
        depth_max = tmp_depth.max()

        # replace all  nan values with max dist
        pts_depth_wo_nan = np.nan_to_num(pts_depth, nan=depth_max)

        DEPTH_THRESH = 1.0 # in meters 
        sel_inds = (pts_depth_wo_nan < DEPTH_THRESH)

        sel_pts = pts[sel_inds]
        sel_pts_colors = pts_colors[sel_inds]

        # skip if selected points are less than 1000 points
        if sum(sel_inds) < 20000:
            print(f"Warning! Frame {osp.basename(pcdp)} has less than 20000 point points. Frame dropped!!")
            continue
        
        # bb()
        # create open3d point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.cpu.pybind.utility.Vector3dVector(sel_pts * 1000.) # convert vertices unit to mm
        pcd_o3d.colors = o3d.cpu.pybind.utility.Vector3dVector(sel_pts_colors)
        fname = osp.join(base_dir, osp.basename(imp).replace('bgr.npz', 'ply'))
        os.makedirs(osp.dirname(fname), exist_ok=True)
        o3d.io.write_point_cloud(fname, pcd_o3d, write_ascii=True)
    print('Done!!') 




if False:
    pth = "/scratch/1/user/aswamy/data/colmap-hand-obj"
    sqns = os.listdir(pth)
    import pathlib
    miss_sqns = []
    for sqn in sqns:
        p = pathlib.Path(osp.join(pth, sqn, 'sparse/0/cameras.bin'))
        if not p.exists():
            miss_sqns.append(sqn)

    print('Miss sqns:', miss_sqns)


# dataset sqs with subj anmes
if True:
    RES_DIR = "/scratch/1/user/aswamy/data/hand-obj"
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar") 

    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))

    all_dicts =  []
    all_objs = []
    all_objs_names = []
    for sqn in all_sqns:
        print(sqn)
        gt_mesh_pth = sorted(glob.glob(osp.join(RES_DIR, sqn, 'gt_mesh/*.obj')))[0]
        
        print(gt_mesh_pth)
        m_name = osp.basename(gt_mesh_pth)
        print(m_name)
        new_dict = {
            'sqn' : sqn,
            'subj' : m_name[:4],
            # 'obj' : m_name.split('_')[1] if '_' in m_name else m_name,
            'obj_name' : m_name,
        }
        # all_objs.append(m_name.split('_')[1])
        all_objs_names.append(m_name)

        import pandas as pd
        df = pd.DataFrame.from_dict([new_dict])
        all_dicts.append(df)
         
    df_comb = pd.concat(all_dicts)
    df_comb_save_pth = osp.join(RES_DIR, 'dataset_log.csv')
    bb()
    df_comb.to_csv(df_comb_save_pth)
    print(f"Saved here: {df_comb_save_pth}")

    my_dict = {i : all_objs.count(i) for i in all_objs}

    for k in my_dict:
        if my_dict[k] > 1:
            print(k, my_dict[k])

    print('Done!!')

# colmap poses stats
if False:
    COLMAP_RESP_DIR = "/scratch/1/user/aswamy/data/colmap-hand-obj"
    all_sqns = os.listdir(COLMAP_RESP_DIR)

    all_dfs = []
    for sqn in all_sqns:
        print('sqn', sqn)
        cps_pths = sorted(glob.glob(osp.join(COLMAP_RESP_DIR, sqn, 'cam_poses/*.txt')))
        imgs_pths = sorted(glob.glob(osp.join(COLMAP_RESP_DIR, sqn, 'images/*.jpg')))
    
        df = {
            'sqn' :sqn,
            'miss_frms_num' : f"{len(cps_pths)}",
            'total_frms' : f"{len(imgs_pths)}",
            'det_ratio' : f"{len(cps_pths)/len(imgs_pths)}"
        }
        import pandas as pd
        df = pd.DataFrame.from_dict([df])
        all_dfs.append(df)

    bb()
    df_comb = pd.concat(all_dfs)
    df_comb_save_pth = osp.join(RES_DIR, 'colmap_det_stats.csv')
    df_comb.to_csv(df_comb_save_pth)
    print('Done')



# small_objs = ['20220905111237', '20220902154737', '20220902111535', '20220909151546', '20220824105341', '20220902111409', '20220902151726', '20220909134639', '20220909120614', '20220905105332', '20220905112733', '20220913144436', '20220823115809', '20220902110304', '20220823114538', '20220829154032', '20220913154643', '20220909115705', '20220909152911', '20220909121541', '20220812172414']
if False:
    RES_DIR = "/scratch/1/user/aswamy/data/hand-obj"
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar") 
    all_sqns = []
    for spth in tqdm(all_seqs_tar_pths):
        if os.path.isfile(spth):
            sqn = osp.basename(spth.split('.')[0])
            all_sqns.append(sqn)
        imgs_dir = osp.join(RES_DIR, sqn, 'rgb')
        file_pth = osp.join(RES_DIR, sqn, 'vid_rgb.mp4')
        imgs2vid_ffmpeg(imgs_dir, file_pth, ext='png', frm_rate=30)
    print('Done!')


if False:
    from moviepy.editor import VideoFileClip, clips_array

    vid_pths = glob.glob(osp.join(RES_DIR, f'*/vid_rgb.mp4'))

    # Load all your videos
    videos = [VideoFileClip(vpth) for vpth in vid_pths]

    # Make sure all clips have the same duration as the shortest clip
    duration = max(clip.duration for clip in videos)
    
    videos = [clip.subclip(0, duration) for clip in videos]

    # Create a 8x12 grid
    video_grid = [videos[n:n+8] for n in range(0, len(videos), 8)]

    # Use clips_array to stitch them together
    final_clip = clips_array(video_grid)

    # Write the result to a file
    final_clip.write_videofile("./output_8x12.mp4", codec='libx264', audio_codec='aac')
