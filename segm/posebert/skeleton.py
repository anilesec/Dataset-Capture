import math
import torch
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import ImageDraw, Image

def get_dope_hand_joint_names():
    return ['wrist',
            'thumb1',
            'index1',
            'middle1',
            'ring1',
            'pinky1',
            'thumb2',
            'thumb3',
            'thumb4',
            'index2',
            'index3',
            'index4',
            'middle2',
            'middle3',
            'middle4',
            'ring2',
            'ring3',
            'ring4',
            'pinky2',
            'pinky3',
            'pinky4']
    
def get_mano_joint_names():
    return [
        'wrist',  # 0
        'index1',  # 1
        'index2',  # 2
        'index3',  # 3
        'middle1',  # 4
        'middle2',  # 5
        'middle3',  # 6
        'pinky1',  # 7
        'pinky2',  # 8
        'pinky3',  # 9
        'ring1',  # 10
        'ring2',  # 11
        'ring3',  # 12
        'thumb1',  # 13
        'thumb2',  # 14
        'thumb3',  # 15
        # Adding the top joints for each joints
        'thumb4',  # 16
        'index4',  # 17
        'middle4',  # 18
        'ring4',  # 19
        'pinky4'  # 20
    ]

def get_contactpose_joint_names():
    return [
        'wrist',
        'thumb1',
        'thumb2',
        'thumb3',
        'thumb4',
        'index1',
        'index2',
        'index3',
        'index4',
        'middle1',
        'middle2',
        'middle3',
        'middle4',
        'ring1',
        'ring2',
        'ring3',
        'ring4',
        'pinky1',
        'pinky2',
        'pinky3',
        'pinky4'        
    ]
    
def update_mano_joints(jts, verts, tips_idx=[745, 317, 444, 556, 673]):
    """
    Add top joints for each finger
    Moving from 16 joints to 21 joints
    :param jts: [b,16,3]
    :param verts: [b,78,3]
    :return: out: [b,21,3
    """
    tips = verts[:, tips_idx]
    out = torch.cat([jts, tips], 1)
    return out

def get_mano_skeleton():
    return np.array(
        [
            [
                # thumb
                [0, 13],
                [13, 14],
                [14, 15],
                [15, 16]
            ],
            [
                # index
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 17]
            ],
            [
                # middle
                [0, 4],
                [4, 5],
                [5, 6],
                [6, 18]
            ]
            ,
            [
                # ring
                [0, 10],
                [10, 11],
                [11, 12],
                [12, 19]
            ],
            [
                # pinky
                [0, 7],
                [7, 8],
                [8, 9],
                [9, 20]
            ]
        ]
    )

def convert_jts(jts, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    list_out = []
    for idx, jn in enumerate(dst_names):
        idx = src_names.index(jn)
        list_out.append(jts[:, idx])
    if isinstance(jts, torch.Tensor):
        out = torch.stack(list_out, 1)
    else:
        out = np.stack(list_out, 1)
    return out

def visu_pose3d(x, res=512, bones=get_mano_skeleton(), lw_line=1, lw_dot=1, colors=['b', 'g', 'c', 'm', 'y'], factor=6.):
    """
    x: tensor of shape [N,K,3]
    """
    
    x *= factor

    plt.style.use('dark_background')
    fig = Figure((res / 100., res / 100.))
    canvas = FigureCanvas(fig)

    # 3D
    ax = fig.gca(projection='3d')
    for i, pose3d in enumerate(x):
        if isinstance(pose3d, torch.Tensor):
            pose3d = pose3d.detach().cpu().numpy()
        if isinstance(pose3d, np.ndarray):
            pass
        else:
            raise NotImplementedError

        pose3d = np.stack([
            - pose3d[:, 0],
            - pose3d[:, 2],
            - pose3d[:, 1]
        ], 1)

        for k, part in enumerate(bones):
            for i, j in part:
                x = [pose3d[i, 0], pose3d[j, 0]]
                y = [pose3d[i, 1], pose3d[j, 1]]
                z = [pose3d[i, 2], pose3d[j, 2]]
                ax.plot(x, y, z, colors[k], scalex=None, scaley=None, lw=lw_line)

        # red circle for all joints
        ax.scatter3D(pose3d[:, 0], pose3d[:, 1], pose3d[:, 2], c='red', lw=lw_dot)

    # legend and ticks
    ax.set_aspect('auto')
    # ax.elev = 20  # 45
    # ax.azim = -90
    ax.view_init(15, 45)  # 0 45 90 315
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-12)
    ax.set_ylabel('Y axis', labelpad=-12)
    ax.set_zlabel('Z axis', labelpad=-12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()

    if img.shape[0] != res:
        img = np.asarray(Image.fromarray(img).resize((res, res)))

    return img

def visu_pose2d(img, pose2d, bones=get_mano_skeleton(), colors=['blue', 'green', 'cyan', 'magenta', 'yellow'], lw_line= 3, lw_dot = 3, color_dot='red'):
    
    img_ = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(img_)
    
    for k, part in enumerate(bones):
        for i, j in part:
            x1y1x2y2 = (pose2d[i, 0], pose2d[i, 1],pose2d[j, 0], pose2d[j, 1])
            draw.line(x1y1x2y2, fill=colors[k], width=lw_line)

    for k in range(pose2d.shape[0]):
        p0 = pose2d[k]
        p0_ = (p0[0] - lw_dot, p0[1] - lw_dot, p0[0] + lw_dot, p0[1] + lw_dot)
        draw.ellipse(p0_, fill=color_dot, outline=color_dot)
    
    return np.asarray(img_)

def normalize_skeleton_by_bone_length(x, y, traversal, parents):
    """
    Args:
        - pred: [k,3]
        - gt: [k,3]
        - traversal: list of len==k
        - parents: list of len==k
    """
    x_norm = x.copy()

    for i in range(len(traversal)):
        i_joint = traversal[i]
        i_parent = parents[i]
        y_len = np.linalg.norm(y[i_joint] - y[i_parent])
        x_vec = x[i_joint] - x[i_parent]
        x_len = np.linalg.norm(x_vec)
        # import ipdb
        # ipdb.set_trace()
        if x_len > 0:
            x_norm[i_joint] = x_norm[i_parent] + x_vec * y_len / x_len
    return x_norm

def normalize_skeleton_by_bone_length_updated(x, y, traversal, parents, return_bone_lengths=False, required_bone_lengths=None):
    """
    Args:
        - pred, gt: [k,3] if numpy world, [bs,k,3] in torch world
        - traversal: list of len==k
        - parents: list of len==k
        - required_bone_lengths: np.array [k] or torch.Tensor [bs,k]
    """
    if isinstance(x, np.ndarray):
        x_norm = x.copy()
    elif isinstance(x, torch.Tensor):
        assert len(x.shape) == 3
        x_norm = x.clone()
        # x_norm = x

    l_bone_length = []
    for i in range(len(traversal)):
        i_joint = traversal[i]
        i_parent = parents[i]

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            y_len = np.linalg.norm(y[i_joint] - y[i_parent]) # float
            if required_bone_lengths is not None:
                y_len = required_bone_lengths[i]
            x_vec = x[i_joint] - x[i_parent]
            x_len = np.linalg.norm(x_vec)
            if x_len > 0:
                x_norm[i_joint] = x_norm[i_parent] + x_vec * y_len / x_len
        elif isinstance(x, torch.Tensor) and  isinstance(y, torch.Tensor):
            y_len = torch.linalg.norm(y[:,i_joint] - y[:,i_parent], dim=1) # [bs]
            if required_bone_lengths is not None:
                y_len = required_bone_lengths[:,i] # [bs]
            x_vec = x[:,i_joint] - x[:,i_parent]
            x_len = torch.linalg.norm(x_vec, dim=1) # [bs]
            x_norm[:,i_joint] = x_norm[:, i_parent] + x_vec * y_len.unsqueeze(1) / (x_len.unsqueeze(1) + 1e-5)
        else:
            raise
        l_bone_length.append(x_len)
    
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        bone_lengths = np.asarray(l_bone_length)
    elif isinstance(x, torch.Tensor) and  isinstance(y, torch.Tensor):
        bone_lengths = torch.stack(l_bone_length, 1) # [bs,20]
    else:
        raise
    
    if return_bone_lengths:
        return x_norm, bone_lengths
    
    return x_norm

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Args:
        - axis: np.array - [3]
        - theta: scalar
    Return
        - out: np.array - [3,3]
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    out = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return out

def rotation_matrix_torch(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Args:
        - axis: torch.Tensor - [B,3]
        - theta: torch.Tensor - [B]
    Return
        - out: torch.Tensor - [B,3,3]
    """
    axis = axis / (torch.sqrt((axis * axis).sum(dim=-1, keepdim=True)) + 1e-5)
    # axis = axis / torch.sqrt((axis * axis).sum(dim=-1, keepdim=True))
    a = torch.cos(theta / 2.0)
    bcd = -axis * torch.sin(theta.reshape(-1, 1) / 2.0)
    b, c, d = bcd[:, 0], bcd[:, 1], bcd[:, 2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    row1 = torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], 1)
    row2 = torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], 1)
    row3 = torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], 1)
    out = torch.stack([row1, row2, row3], 1)
    return out

def unit_vector(vector):
    """ Returns the unit vector of the vector.  
    Args:
        - vector: np.array [3]
    Return:
        - out: np.array [3]
    """
    out = vector / np.linalg.norm(vector)
    return out

def unit_vector_torch(vector):
    """ Returns the unit vector of the vector.  
    Args:
        - vector: torch.Tensor [B,3]
    Return:
        - out: torch.Tensor [B,3]
    """
    out = vector / (torch.linalg.norm(vector, dim=1, keepdim=True) + 1e-5)
    # out_init = vector / torch.linalg.norm(vector, dim=1, keepdim=True)
    return out

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    Args:
        - v1, v2: np.array [3]
    Return:
        - out: scalar
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    out = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return out

def angle_between_torch(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
    Args:
        - v1, v2: torch.tensor [B,3]
    Return:
        - out: torch.Tensor [B]
    """
    v1_u = unit_vector_torch(v1)
    v2_u = unit_vector_torch(v2)
    # dot = torch.bmm(v1_u.unsqueeze(1), v2_u.unsqueeze(2)).reshape(-1)
    dot = (v1_u * v2_u).sum(-1)
    out = torch.arccos(torch.clamp(dot, -1.0, 1.0))

    return out

def preprocess_skeleton_torch(pose, center_joint=[0], xaxis=[10, 1], yaxis=[4, 0], iter=1, norm_x_axis=True, norm_y_axis=True):

    pose_rel = pose.clone()

    # Sub the center joint (pelvis 17)
    pose_center = pose_rel[:, center_joint].mean(1, keepdims=True)
    pose_rel = pose_rel - pose_center

    # Compute the relative pose
    matrices = []
    for _ in range(iter):
        # Y axis
        if norm_y_axis:
            joint_diff = pose_rel[:, yaxis[1]] - pose_rel[:, yaxis[0]]
            ref = torch.Tensor([[0, 1, 0]]).float().repeat(joint_diff.shape[0], 1).type_as(pose)
            axis = torch.cross(joint_diff, ref)
            angle = angle_between_torch(joint_diff, ref)
            matrix = rotation_matrix_torch(axis, angle)
            pose_rel = torch.matmul(matrix.unsqueeze(1), pose_rel.unsqueeze(-1)).squeeze(-1)
            matrices.append(matrix)
        
        # X axis
        if norm_x_axis:
            joint_diff = pose_rel[:, xaxis[1]] - pose_rel[:, xaxis[0]]
            ref = torch.Tensor([[1, 0, 0]]).float().repeat(joint_diff.shape[0], 1).type_as(pose)
            axis = torch.cross(joint_diff, ref)
            angle = angle_between_torch(joint_diff, ref)
            matrix = rotation_matrix_torch(axis, angle)
            pose_rel = torch.matmul(matrix.unsqueeze(1), pose_rel.unsqueeze(-1)).squeeze(-1)
            matrices.append(matrix)
    
    # compute the center orient rotmat
    matrices.reverse()
    orient_center = matrices[0]
    for x in matrices[1:]:
        orient_center = torch.matmul(orient_center, x)
    
    return pose_rel, pose_center[:, 0], orient_center

def preprocess_skeleton(pose, center_joint=[0], xaxis=[1, 4], yaxis=[7, 0], iter=5, sanity_check=True,
                        norm_x_axis=True, norm_y_axis=True):
    """
    Preprocess skeleton such that we disentangle the root orientation and the relative pose
    Following code from https://github.com/lshiwjx/2s-AGCN/blob/master/data_gen/preprocess.py
    Default values are for h36m_plus skeleton (center=hip, xaxis=left_shoulder/right_shoulder, yaxis=spine/hip
    Args:
        - pose: [t,k,3] np.array
        - center_joint: list
        - xaxis: list
        - yaxis: list
        - iter: int
    Return:
        - pose_rel: [t,k,3] np.array
        - pose_center: [t,3] np.array
        - matrix: [t,3,3] np.array
    """
    pose_rel = pose.copy()

    # Sub the center joint (pelvis 17)
    pose_center = pose_rel[:, center_joint].mean(1, keepdims=True)
    pose_rel = pose_rel - pose_center

    list_matrix = []
    list_diff = []
    for t in range(pose_rel.shape[0]):

        matrix = []
        inv_matrix = []
        for _ in range(iter):
            # parallel the bone between hip(jpt 0) and spine(jpt 7) to the Y axis
            if norm_y_axis:
                joint_bottom = pose_rel[t, yaxis[0]]
                joint_top = pose_rel[t, yaxis[1]]
                joint_diff = joint_top - joint_bottom
                ref = np.asarray([0, 1, 0])

                # bs = 600
                # pose_rel_t = torch.from_numpy(pose_rel[t]).unsqueeze(0).repeat(bs, 1, 1)
                # ipdb.set_trace()
                #
                # joint_bottom = pose_rel[:, yaxis[0]]
                # joint_top = pose_rel[:, yaxis[1]]
                # axis = np.cross(joint_top - joint_bottom, [0, 1, 0]).astype(np.float32)
                #
                axis = np.cross(joint_diff, ref).astype(np.float32)
                angle = angle_between(joint_diff, ref).astype(np.float32)                
                matrix_x = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_x.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_x)

                # # torch
                # joint_diff_ = torch.from_numpy(joint_diff).unsqueeze(0).float().repeat(bs, 1)
                # ref_ = torch.from_numpy(ref).unsqueeze(0).float().repeat(bs, 1)
                # axis_ = torch.cross(joint_diff_, ref_)
                # print(axis_[0] - axis)
                # angle_ = angle_between_torch(joint_diff_, ref_)
                # print(angle_[0] - angle)
                # matrix_x_ = rotation_matrix_torch(axis_, angle_)
                # # ipdb.set_trace()
                # print((matrix_x_[0] - matrix_x_).abs().sum())
                # pose_rel_t_ = torch.matmul(matrix_x_.unsqueeze(1), pose_rel_t.unsqueeze(-1)).squeeze(-1)
                # ipdb.set_trace()
                # print((pose_rel_t_[0] - pose_rel[t]).abs().sum())

            # parallel the bone between right_shoulder(jpt 0) and left_shoulder(jpt 7) to the X axis
            if norm_x_axis:
                joint_rshoulder = pose_rel[t, xaxis[0]]
                joint_lshoulder = pose_rel[t, xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0]).astype(np.float32)
                angle = angle_between(joint_rshoulder - joint_lshoulder, np.asarray([1, 0, 0])).astype(np.float32)
                matrix_y = rotation_matrix(axis, angle).astype(np.float32)
                pose_rel[t] = (matrix_y.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
                matrix.append(matrix_y)

        # compute the center orient rotmat
        matrix.reverse()
        mat = matrix[0]
        for x in matrix[1:]:
            mat = mat @ x
        list_matrix.append(mat)

        if sanity_check:
            # sanity check for computing the inverse matrix step by step
            matrix.reverse()
            inv_mat = np.linalg.inv(matrix[0])
            for x in matrix[1:]:
                inv_mat = inv_mat @ np.linalg.inv(x)
            pose_centered_t_bis = (inv_mat.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            pose_centered_t = pose[t] - pose_center[t]
            err = np.abs(pose_centered_t_bis - pose_centered_t).sum()
            # print(err)
            assert err < 1e-5
            inv_matrix.append(inv_mat)

            # sanity check for matrix multiplication
            pose_rel_bis = pose.copy() - pose_center
            pose_rel_t_bis = (mat.reshape(1, 3, 3) @ pose_rel_bis[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_rel_t_bis - pose_rel[t]).sum()
            # print(err)
            assert err < 1e-5

            # inv bis
            inv_mat_bis = np.linalg.inv(mat)
            pose_centered_t_bis_bis = (inv_mat_bis.reshape(1, 3, 3) @ pose_rel[t].reshape(-1, 3, 1)).reshape(-1, 3)
            err = np.abs(pose_centered_t_bis_bis - pose_centered_t).sum()
            # print(err)
            assert err < 1e-5

    orient_center = np.stack(list_matrix)
    return pose_rel, pose_center.reshape(-1, 3), orient_center

def get_mano_traversal():
    traversal = ['thumb1', 'thumb2', 'thumb3', 'thumb4',
                 'index1', 'index2', 'index3', 'index4',
                 'middle1', 'middle2', 'middle3', 'middle4',
                 'ring1', 'ring2', 'ring3', 'ring4',
                 'pinky1', 'pinky2', 'pinky3', 'pinky4'
                 ]
    parents = ['wrist', 'thumb1', 'thumb2', 'thumb3',
               'wrist', 'index1', 'index2', 'index3',
               'wrist', 'middle1', 'middle2', 'middle3',
               'wrist', 'ring1', 'ring2', 'ring3',
               'wrist', 'pinky1', 'pinky2', 'pinky3',
               ]

    names = get_mano_joint_names()
    traversal_idx = []
    parents_idx = []
    for i in range(len(traversal)):
        traversal_idx.append(names.index(traversal[i]))
        parents_idx.append(names.index(parents[i]))

    assert len(traversal_idx) == len(parents_idx)

    return traversal_idx, parents_idx

def perspective_projection(points, c_x, c_y, f_x, f_y, no_nan=True):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Input:
        points (bs, N, 3): 3D points
        c_x, c_y, f_x, f_y (int): Focal length and camera center
    """
    batch_size = points.shape[0]

    # Apply perspective distortion
    if no_nan:
        projected_points = points / (torch.abs(points[:, :, -1].unsqueeze(-1)) + 1e-5)  # 1(bs, N, 3)
    else:
        projected_points = points / points[:, :, -1].unsqueeze(-1)  # 1(bs, N, 3)
    
    # print("projected_points: ", torch.isnan(projected_points).float().sum().item())

    # Camera intrinsic params
    K = torch.zeros([batch_size, 3, 3], device=points.device)  # (bs, 3, 3)
    K[:, 0, 0] = f_x
    K[:, 1, 1] = f_y
    K[:, 2, 2] = 1.
    K[:, 0, -1] = c_x
    K[:, 1, -1] = c_y

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)  # (bs, N, 3)

    return projected_points[:, :, :2]

def inverse_projection_to_3d(y, camera_size, K_inverse):
    """
    Given the 2d location and the distance it predicts the 3d location in the camera space
    Args:
        - y: [batch_size, seq_len, 3] - 2d location in pixel space normalized to 0-1 and nearness
    """
    # Transform input
    points = y[..., :2] * camera_size
    nearness = y[..., -1:]
    distance = 1. / torch.exp(nearness)

    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    if K_inverse.shape[0] == 1:
        K_inverse = K_inverse.repeat(points.shape[0], 1, 1)
    # import ipdb
    # ipdb.set_trace()
    points = torch.einsum('bij,bkj->bki', K_inverse, points)

    # Apply perspective distortion
    y = points * distance

    return y

def estimate_translation(pose3d, pose2d, f_x, f_y, c_x, c_y, joints_conf=None):
    """Find the translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        pose3d: (B, K, 3) 3D joint locations centered around one joint!
        pose2d: (B, K, 2) 2D joint locations - in pixel space
        joints_conf: (B, K) conf (0 or 1)
    Returns:
        (B,3) translation vector
    """
    list_trans = []
    for i in range(pose3d.shape[0]):
        pose3d_i, pose2d_i = pose3d[i], pose2d[i]
        if isinstance(pose3d, torch.Tensor):
            pose3d_i = pose3d_i.numpy()
            pose2d_i = pose2d_i.numpy()
        joints_conf_i = None
        if joints_conf is not None:
            joints_conf_i = joints_conf[i]
            if isinstance(joints_conf, torch.Tensor):
                joints_conf_i = joints_conf_i.numpy()
        try:
            trans = estimate_translation_np(pose3d_i, pose2d_i, f_x, f_y, c_x, c_y, joints_conf_i)
        except:
            print('error - impossible to estimate the translation')
            trans = np.zeros((3)).astype(np.float32)
        list_trans.append(trans)
    return torch.from_numpy(np.stack(list_trans)).float()

def estimate_translation_np(pose3d, pose2d, f_x, f_y, c_x, c_y, joints_conf=None):
    """Find the translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        pose3d: (K, 3) 3D joint locations centered around one joint!
        pose2d: (K, 2) 2D joint locations - in pixel space
    Returns:
        (3,) translation vector
    """
    # import ipdb
    # ipdb.set_trace()

    if joints_conf is None:
        joints_conf = np.ones_like(pose2d[:, 0])

    num_joints = pose3d.shape[0]

    # focal length
    f = np.array([f_x, f_y])

    # optical center
    center = np.array([c_x, c_y])

    # transformations
    Z = np.reshape(np.tile(pose3d[:, 2], (2, 1)).T, -1)
    XY = np.reshape(pose3d[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
                  O - np.reshape(pose2d, -1)]).T
    c = (np.reshape(pose2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

if __name__ == "__main__":
    import ipdb
    import smplx
    import torch.nn.functional as F
    import roma
    from PIL import Image
    import time
    from tqdm import tqdm
    from posebert.constants import SMPLX_DIR

    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
    out = bm(global_orient=roma.random_rotvec().unsqueeze(0))
    with torch.no_grad():
        mano_mean = update_mano_joints(out.joints, out.vertices)
    x = mano_mean.repeat(100, 1, 1)
    N = 1

    # projection
    img_res = 512
    f_x, f_y, c_x, c_y = 600., 600., img_res/2., img_res/2.
    transl = torch.Tensor([[[0., 0., -5.]]])
    x_transl = x + transl
    x2d = perspective_projection(x, f_x, f_y, c_x, c_y)

    # numpy
    start = time.time()
    for _ in tqdm(range(N)):
        rel1, _, root1 = preprocess_skeleton(x.numpy(),center_joint=[0],  # wrist
                            xaxis=[1, 10],  # middle1-wrist
                            yaxis=[4, 0],  # index1-right1
                            iter=1, norm_x_axis=True, norm_y_axis=True, sanity_check=False)
    dur_np = (time.time() - start)/float(N)
    
    # torch
    start = time.time()
    for _ in tqdm(range(N)):
        rel1_, _, root1_ = preprocess_skeleton_torch(x,center_joint=[0],  # wrist
                        xaxis=[10, 1],  # middle1-wrist
                        yaxis=[4, 0],  # index1-right1
                        iter=1, norm_x_axis=True, norm_y_axis=True)
    dur_torch = (time.time() - start)/float(N)

    # ipdb.set_trace()
    print(f"Duration: np:{dur_np:.6f} - torch:{dur_torch:.6f}")
    print(f"Diff Rel: : {(torch.from_numpy(rel1) - rel1_).abs().sum()}")
    print(f"Diff Root: : {(torch.from_numpy(root1) - root1_).abs().sum()}")

    img1 = visu_pose3d(x[:1])
    img2 = visu_pose3d(rel1[:1])
    img3 = visu_pose3d(rel1_[:1])
    img = np.concatenate([img1, img2, img3], 1)
    Image.fromarray(img).save('img.jpg')


    print("Done")

def left_to_right_mano_params(x):
    return right_to_left_mano_params(x)

def right_to_left_mano_params(x):
    """
    Flip
    :param x: [b,16,3]
    :return: y : [b,16,3]
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
    elif isinstance(x, np.ndarray):
        y = x.copy()
    else:
        raise NameError
    y[..., 0] *= 1
    y[..., 1] *= -1
    y[..., 2] *= -1
    return y

def get_bbox(joint_img, factor=1., square=False, format='x1y1x2y2'):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    ymax = np.max(joint_img[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    if square:
        max_side = max([width, height])
        width = max_side
        height = max_side
    
    if format == 'x1y1x2y2':
        bbox[0] = (xmin + xmax)/2. - width/2*factor
        bbox[1] = (ymin + ymax)/2. - height/2*factor
        bbox[2] = (xmin + xmax)/2. + width/2*factor
        bbox[3] = (ymin + ymax)/2. + height/2*factor
    elif format == 'x1y1wh':
        bbox[0] = (xmin + xmax)/2. - width/2*factor
        bbox[1] = (ymin + ymax)/2. - height/2*factor
        bbox[2] = width*factor
        bbox[3] = height*factor
    else:
        raise NameError

    return bbox

if __name__ == "__main__":
    import smplx
    from posebert.renderer import PyTorch3DRenderer
    from pytorch3d.renderer import look_at_view_transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm_r = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
    bm_l = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=False)
    faces_r = torch.from_numpy(np.array(bm_r.faces, dtype=np.int32))
    faces_l = torch.from_numpy(np.array(bm_l.faces, dtype=np.int32))

    img_res = 512
    focal_length = 1500.
    renderer = PyTorch3DRenderer(img_res).to(device)
    dist, elev, azim = 2, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    
    global_orient = torch.Tensor([[np.pi/4., 0., 0.]])
    out_r, out_l = bm_r(global_orient=global_orient), bm_l(global_orient=global_orient)
    verts_r, verts_l = out_r.vertices[0], out_l.vertices[0]
    jts_r, jts_l = update_mano_joints(out_r.joints, out_r.vertices)[0], update_mano_joints(out_l.joints, out_l.vertices)[0]
    img_r = renderer.renderPerspective(vertices=[verts_r.to(device)], 
                                                faces=[faces_r.to(device)],
                                                rotation=rotation.to(device),
                                                camera_translation=cam.to(device),
                                                focal_length=2*focal_length/img_res,
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                ).cpu().numpy()[0]
    img_l = renderer.renderPerspective(vertices=[verts_l.to(device)], 
                                                faces=[faces_l.to(device)],
                                                rotation=rotation.to(device),
                                                camera_translation=cam.to(device),
                                                focal_length=2*focal_length/img_res,
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                ).cpu().numpy()[0]

    verts_r_ = torch.stack([-verts_l[:,0], verts_l[:,1], verts_l[:,2]], 1)
    print((verts_r_ - verts_r_).abs().sum())

    img_r_ = renderer.renderPerspective(vertices=[verts_r.to(device)], 
                                                faces=[faces_r.to(device)],
                                                rotation=rotation.to(device),
                                                camera_translation=cam.to(device),
                                                focal_length=2*focal_length/img_res,
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                ).cpu().numpy()[0]

    img_top = np.concatenate([img_l, img_r_, img_r], 1)
    
    jts_r_ = torch.stack([-jts_l[...,0], jts_l[...,1], jts_l[...,2]], -1)
    print((jts_r_ - jts_r_).abs().sum())
    img1 = visu_pose3d(jts_l.unsqueeze(0).clone(), res=img_res, bones=get_mano_skeleton())
    img2 = visu_pose3d(jts_r_.unsqueeze(0).clone(), res=img_res, bones=get_mano_skeleton())
    img3 = visu_pose3d(jts_r.unsqueeze(0).clone(), res=img_res, bones=get_mano_skeleton())
    img_bottom = np.concatenate([img1, img2, img3], 1)

    img = np.concatenate([img_top, img_bottom])

    Image.fromarray(img).save('img.jpg')

    print("done")
