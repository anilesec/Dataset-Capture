# adapted from https://github.com/cvg/nice-slam
from dis import dis
import random, os
import numpy as np
import argparse
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
import cv2
from ipdb import set_trace as bb
import pickle
import torch    
osp = os.path
from tqdm import tqdm
import pprint, glob
import pathlib

def saveas_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc, distances


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp, distances

def write_vis_pcd(file, points, colors):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """    
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances, indices


def calc_3d_metric(rec_meshfile, gt_meshfile, align=False, sampl_num=100000, dist_thresh=None, save_pkl=None):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    # found the aligned bbox for the mesh
    to_align, _ = trimesh.bounds.oriented_bounds(mesh_gt)
    mesh_gt.vertices = (to_align[:3, :3] @ mesh_gt.vertices.T + to_align[:3, 3:]).T
    mesh_rec.vertices = (to_align[:3, :3] @ mesh_rec.vertices.T + to_align[:3, 3:]).T

    min_points = mesh_gt.vertices.min(axis=0) * 1.05
    max_points = mesh_gt.vertices.max(axis=0) * 1.05

    mask_min = (mesh_rec.vertices - min_points[None]) > 0
    mask_max = (mesh_rec.vertices - max_points[None]) < 0

    mask = np.concatenate((mask_min, mask_max), axis=1).all(axis=1)
    face_mask = mask[mesh_rec.faces].all(axis=1)

    mesh_rec.update_vertices(mask)
    mesh_rec.update_faces(face_mask)
    # bb()
    rec_pc = trimesh.sample.sample_surface(mesh_rec, sampl_num)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, sampl_num)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    # ch_dist = chamf_dist(np.array(gt_pc_tri.vertices)[: int(sample_num / 10)], np.array(rec_pc_tri.vertices)[:int(sample_num / 10)])
    accuracy_rec, dist_d2s = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, dist_s2d = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices, dist_thresh)
    
    precision_ratio_rec = completion_ratio(
        rec_pc_tri.vertices, gt_pc_tri.vertices, dist_thresh)
    
    fscore = 2 * precision_ratio_rec * completion_ratio_rec / (completion_ratio_rec + precision_ratio_rec)
    
    # normal consistency
    N = sampl_num
    pointcloud_pred, idx = mesh_rec.sample(N, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_rec.face_normals[idx]

    pointcloud_trgt, idx = mesh_gt.sample(N, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_gt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_avg = (normal_acc + normal_comp) * 0.5

    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    precision_ratio_rec *= 100  # convert to %
    fscore *= 100
    normal_acc *= 100
    normal_comp *= 100
    normal_avg *= 100

    tosave_dict = dict()
    tosave_dict =  {
            'accuracy_rec' : accuracy_rec,
            'completion_rec' :  completion_rec,
            'precision_ratio_rec' :precision_ratio_rec,
            'completion_ratio_rec' : completion_ratio_rec,
            'fscore' : fscore,
            'normal_acc' : normal_acc,
            'normal_comp': normal_comp,
            'normal_avg' : normal_avg,
            # 'ch_dist': ch_dist
    }

    pprint.pprint(f'Evalutaion in cms and %: {tosave_dict}')
    if save_pkl:
        pkl_save_pth = osp.join(osp.dirname(osp.dirname(rec_meshfile)), f"briac_recon_eval_sample{sampl_num}_dth_{dist_thresh:.4f}.pkl") 
        saveas_pkl(tosave_dict, pkl_save_pth)
        print(f'saved here: {pkl_save_pth}')

    # print(accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg)
    
    if False:
        # add visualization and save the mesh output
        vis_dist = 0.005 # dist_th = 5mm
        data_alpha = (dist_d2s.clip(max=vis_dist) / vis_dist).reshape(-1, 1)
        #data_color = R * data_alpha + W * (1-data_alpha)
        im_gray = (data_alpha * 255).astype(np.uint8)
        data_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)[:,0,[2, 0, 1]] / 255.
        write_vis_pcd(f'{rec_meshfile}_d2s_sampl{sampl_num}_dth{dist_thresh}.ply', rec_pc_tri.vertices, data_color)

        stl_alpha = (dist_s2d.clip(max=vis_dist) / vis_dist).reshape(-1, 1)
        #stl_color = R * stl_alpha + W * (1-stl_alpha)
        im_gray = (stl_alpha * 255).astype(np.uint8)
        stl_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)[:,0,[2, 0, 1]] / 255.
        write_vis_pcd(f'{rec_meshfile}_s2d_sampl{sampl_num}_dth{dist_thresh}.ply', gt_pc_tri.vertices, stl_color)


def chamf_dist(gt_pts, rec_pts):
    gt_pts = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
    rec_pts = torch.tensor(rec_pts, dtype=torch.float32).unsqueeze(0)
    # from pytorch3d.loss import chamfer_distance
    # loss_chamfer, _ = chamfer_distance(gt_pts, rec_pts)
    dist1, dist2, idx1, idx2 = chamfer_dist(gt_pts, rec_pts)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))   

    return loss

def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, align=False, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 720
    W = 1080
    # focal = 300
    fx = 899.783
    fy = 900.019
    cx = 653.768
    cy = 362.143

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    bb()
    unseen_gt_pointcloud_file = gt_meshfile.replace('.obj', '_pc_unseen.npy')
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in range(n_imgs):
        while True:
            # sample view, and check if unseen region is not inside the camera view, if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target)-np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if (~seen):
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(gt_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(gt_mesh, reset_bounding_box=True,)

        vis.add_geometry(rec_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True,)

        errors += [np.abs(gt_depth-ours_depth).mean()]

    errors = np.array(errors)
    # from m to cm
    print('Depth L1: ', errors.mean()*100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the reconstruction.'
    )
    # parser.add_argument('--rec_mesh', type=str,
    #                     help='reconstructed mesh file path')
    # parser.add_argument('--gt_mesh', type=str,
    #                     help='ground truth mesh file path')
    # parser.add_argument('--viz', type=str, default=1, 
    #                     help='viz the errors on pcds')
    # parser.add_argument('--save', type=str, default=0, 
    #                     help='save or not')
    parser.add_argument('--sample_num', type=int, default=200000, 
                        help='no of pts to sample on surface')
    parser.add_argument('--sqn', type=str, default=None, help='')
    args = parser.parse_args()


    BRIAC_RECON_RES_DIR = '/scratch/1/user/aswamy/data/briac_baseline'
    RES_DIR = '/scratch/1/user/aswamy/data/hand-obj'
    SAMPLE_NUM = 200000

    # get all sqn ids
    all_sqn_ids = os.listdir(BRIAC_RECON_RES_DIR)

    if args.sqn is not None:
        if args.sqn not in all_sqn_ids:
            print(f"{args.sqn} is not present in listed sequences!!!")
            print(f"No briac recons for {args.sqn}")
            with open('/scratch/1/user/aswamy/data/briac_miss_sqns.txt', 'a') as f:
                f.write(args.sqn+'\n')
            exit()
        all_sqn_ids = [args.sqn]
    
    miss_gt_meshes = []
    for sqn in tqdm(all_sqn_ids):
        print(f'sqn: {sqn}')

        rec_mesh_pth = pathlib.Path(osp.join(BRIAC_RECON_RES_DIR, sqn, 'LOD0/mesh_0.ply'))
        gt_mesh_pth = pathlib.Path(glob.glob(osp.join(RES_DIR, sqn, 'gt_mesh/*.obj'))[0])
        
        if not gt_mesh_pth.exists():
            miss_gt_meshes.append(sqn)
            print('Missing GT Mesh:', sqn)
            continue

        print(f'rec_mesh_pth: {rec_mesh_pth}')
        print(f'gt_mesh_pth: {gt_mesh_pth}')

        for dth in tqdm(np.linspace(0.001, 0.015, 15)):
            print(f'dist_thresh = {dth:.5f}')
            calc_3d_metric(rec_mesh_pth, gt_mesh_pth, sampl_num=SAMPLE_NUM, dist_thresh=dth, save_pkl=True)
    
    print("all_sqn_ids", all_sqn_ids)
    print('Done')

    
