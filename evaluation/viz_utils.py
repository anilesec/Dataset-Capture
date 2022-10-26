import numpy as np
import matplotlib.pylab as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import pickle
import os
import sys
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from ipdb import set_trace as bb

def project(P, X):
    """
  X: Nx3
  P: 3x4 projection matrix, ContactPose.P or K @ cTo
  returns Nx2 perspective projections
  """
    X = np.vstack((X.T, np.ones(len(X))))
    x = P @ X
    x = x[:2] / x[2]
    return x.T

def draw_hands(im, joints, colors=((255, 0, 0), (255, 0, 0)), circle_radius=3,
               line_thickness=2, offset=np.zeros(2, dtype=np.int), jts_order='CP',
               line_type=cv2.LINE_AA):
    if im is None:
        print('Invalid image')
        return im
    if im.ndim == 2:  # depth image
        im = colorcode_depth_image(im)
    for hand_idx, (js, c) in enumerate(zip(joints, colors)):
        if js is None:
            continue
        else:
            js = np.round(js - offset[np.newaxis, :]).astype(np.int)
        for j in js:
            im = cv2.circle(im, tuple(j), circle_radius, c, -1, cv2.LINE_AA)
        # bb()
        if jts_order == 'DOPE':
            # DOPE dets order
            thumb_idxs = [0, 1, 6, 7, 8]
            indxfing_idxs = [0, 2, 9, 10, 11]
            midfing_idxs = [0, 3, 12, 13, 14]
            ringfing_idxs = [0, 4, 15, 16, 17]
            litfing_idxs = [0, 5, 18, 19, 20]

            for t_idx in range(4):
                im = cv2.line(im, tuple(js[thumb_idxs[t_idx]]), tuple(js[thumb_idxs[t_idx + 1]]), (255, 0, 0),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[indxfing_idxs[t_idx]]), tuple(js[indxfing_idxs[t_idx + 1]]), (0, 128, 0),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[midfing_idxs[t_idx]]), tuple(js[midfing_idxs[t_idx + 1]]), (0, 0, 255),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[ringfing_idxs[t_idx]]), tuple(js[ringfing_idxs[t_idx + 1]]), (255, 255, 0),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[litfing_idxs[t_idx]]), tuple(js[litfing_idxs[t_idx + 1]]), (139, 0, 139),
                              line_thickness, line_type)
        elif jts_order == 'CP':
            finger_colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0), (139, 0, 139)]
            for (finger, color) in zip(range(5), finger_colors):
                base = 4 * finger + 1
                im = cv2.line(im, tuple(js[0]), tuple(js[base]), color,
                              line_thickness, line_type)
                for j in range(3):
                    im = cv2.line(im, tuple(js[base + j]), tuple(js[base + j + 1]),
                                  color, line_thickness, line_type)
        elif jts_order == 'FMOCAP':
                # FMOCAP dets order
                thumb_idxs = [0, 1, 2, 3, 4]
                indxfing_idxs = [0, 5, 6, 7, 8]
                midfing_idxs = [0, 9, 10, 11, 12]
                ringfing_idxs = [0, 13, 14, 15, 16]
                litfing_idxs = [0, 17, 18, 19, 20]

                for t_idx in range(4):
                    im = cv2.line(im, tuple(js[thumb_idxs[t_idx]]), tuple(js[thumb_idxs[t_idx + 1]]), (255, 0, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[indxfing_idxs[t_idx]]), tuple(js[indxfing_idxs[t_idx + 1]]), (0, 128, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[midfing_idxs[t_idx]]), tuple(js[midfing_idxs[t_idx + 1]]), (0, 0, 255),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[ringfing_idxs[t_idx]]), tuple(js[ringfing_idxs[t_idx + 1]]),
                                  (255, 255, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[litfing_idxs[t_idx]]), tuple(js[litfing_idxs[t_idx + 1]]), (139, 0, 139),
                                  line_thickness, line_type)
        elif jts_order == 'HO3D':
                # FMOCAP dets order
                thumb_idxs = [0, 13, 14, 15, 16]
                indxfing_idxs = [0, 1, 2, 3, 17]
                midfing_idxs = [0, 4, 5, 6, 18]
                ringfing_idxs = [0, 10, 11, 12, 19]
                litfing_idxs = [0, 7, 8, 9, 20]

                for t_idx in range(4):
                    im = cv2.line(im, tuple(js[thumb_idxs[t_idx]]), tuple(js[thumb_idxs[t_idx + 1]]), (255, 0, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[indxfing_idxs[t_idx]]), tuple(js[indxfing_idxs[t_idx + 1]]), (0, 128, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[midfing_idxs[t_idx]]), tuple(js[midfing_idxs[t_idx + 1]]), (0, 0, 255),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[ringfing_idxs[t_idx]]), tuple(js[ringfing_idxs[t_idx + 1]]),
                                  (255, 255, 0),
                                  line_thickness, line_type)
                    im = cv2.line(im, tuple(js[litfing_idxs[t_idx]]), tuple(js[litfing_idxs[t_idx + 1]]), (139, 0, 139),
                                  line_thickness, line_type)
        if jts_order == 'OURS':
            thumb_idxs = [0, 1, 6, 11, 16]
            indxfing_idxs = [0, 2, 7, 12, 17]
            midfing_idxs = [0, 3, 8, 13, 18]
            ringfing_idxs = [0, 4, 9, 14, 19]
            litfing_idxs = [0, 5, 10, 15, 20]

            for t_idx in range(4):
                im = cv2.line(im, tuple(js[thumb_idxs[t_idx]]), tuple(js[thumb_idxs[t_idx + 1]]), (0, 0, 255),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[indxfing_idxs[t_idx]]), tuple(js[indxfing_idxs[t_idx + 1]]), (0, 128, 0),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[midfing_idxs[t_idx]]), tuple(js[midfing_idxs[t_idx + 1]]), (255, 0, 0),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[ringfing_idxs[t_idx]]), tuple(js[ringfing_idxs[t_idx + 1]]), (0, 255, 255),
                              line_thickness, line_type)
                im = cv2.line(im, tuple(js[litfing_idxs[t_idx]]), tuple(js[litfing_idxs[t_idx + 1]]), (139, 0, 139),
                              line_thickness, line_type)
    # bb()
    return im


def load_pkl(pkl_file):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


def write_gt_poses(poses, intrinsics, affine, fpth=None):
    os.makedirs(os.path.dirname(fpth), exist_ok=True)
    np.savez(fpth, cam_poses=poses, cam_intrinscs=intrinsics, affine=affine)
    print(f"GT poses saved here: {fpth}")

    return None


def safe_arccos(x):
    """
        Returns the arcosine of x, clipped between -1 and 1.
        Use this when you know x is a cosine, but it might be
        slightly over 1 or below -1 due to numerical errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))


def geodesic_distance_for_rotations(R1, R2):
    """
        Returns the geodesic distance between two rotation matrices.
        It is computed as the angle of the rotation :math:`R_1^{*} R_2^{-1}``.
    """
    R = np.dot(R1, R2.T)
    axis1, angle1 = axis_angle_from_rotation(R)  # @UnusedVariable
    # print(axis1)
    return angle1


def axis_angle_from_rotation(R):
    """
        Returns the *(axis,angle)* representation of a given rotation.
        There are a couple of symmetries:
        * By convention, the angle returned is nonnegative.
        * If the angle is 0, any axis will do.
          In that case, :py:func:`default_axis` will be returned.
    """
    angle = safe_arccos((R.trace() - 1) / 2)

    if angle == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    else:
        v = np.array(
            [R[2, 1] - R[1, 2],
             R[0, 2] - R[2, 0],
             R[1, 0] - R[0, 1]]
        )

        computer_with_infinite_precision = False
        if computer_with_infinite_precision:
            axis = (1 / (2 * np.sin(angle))) * v
        else:
            # OK, the formula above gives (theoretically) the correct answer
            # but it is imprecise if angle is small (dividing by a very small
            # quantity). This is way better...
            axis = (v * np.sign(angle)) / np.linalg.norm(v)

        return axis, angle


def quat_diff(Q1, Q2):
    """
    computes the difference between two quaternion rotations in deg
    """
    return 2 * np.arccos(np.abs(np.dot(Q1, Q2))) * (180. / np.pi)


def rotmat_diff(P, Q):
    """
    computes the difference angle b/w rot mats P and Q in deg(°)
    """
    R = np.dot(P, Q.T)
    theta = (np.trace(R) - 1) * 0.5
    # theta_clip = np.clip(theta, -1., 1.)
    theta_deg = np.arccos(theta) * (180 / np.pi)
    # print(f"Angle b/w rotation mats is: {theta_deg:04f}°")

    return theta_deg


trnsfm2homat = lambda t: np.vstack((t, [0., 0., 0., 1.]))


def comb_masks(mask_lst):
    h, w, c = mask_lst[0].shape
    result = np.full((h, w, c), (0, 0, 0), dtype=np.uint8)
    for mask in mask_lst:
        result = cv2.add(result, mask)

    return result


def inv_trnsfm(T):
    """
    compute inverse transform of batch of 3x4 transforms
    :param T: Nx3x4
    :return: Nx4x4
    """
    assert T.ndim == 3, "T should be of size Nx3x4 or Nx4x4"

    if T.shape[-2] != 4:
        hom_axis = np.repeat(np.array([0, 0, 0, 1]).reshape(1, 1, -1), T.shape[0], 0)
        T = np.concatenate((T, hom_axis), 1)

    return np.linalg.inv(T)


def rotran2homat(R, T):
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = R
    hom_mat[:3, 3] = T

    return hom_mat


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def load_json(fpth):
    with open(fpth, 'rb') as fid:
        anno = json.load(fid)

    return anno


def saveas_json(dict_var, fpth=None):
    with open(fpth, 'w') as f:
        json.dump(dict_var, f)

    return None


def write_imgs2gif(imgs, savepth):
    import imageio
    with imageio.get_writer(savepth, mode='I') as f:
        for im in tqdm(imgs):
            f.append_data(im)

    return print(f" gif saved here: {savepth}")


def gif2imgs(fname):
    import imageio
    gif = imageio.get_reader(fname)
    imgs = np.array([im[:, :, :3] for im in gif])

    return imgs


def rtvec2rtmat(rvec, tvec):
    rtmat = np.eye(4)
    rtmat[:3, :3] = cv2.Rodrigues(rvec)[0]
    rtmat[:3, 3] = tvec.flatten()

    return rtmat[:3, :]


def get_projtd_handjts(jts3d_rt, r2ctrnsfm, intrinsics, affine=np.eye(3)):
    proj_mat = affine @ (intrinsics @ r2ctrnsfm)
    projtd_jts2d = project(P=proj_mat, X=jts3d_rt)

    return projtd_jts2d


def reproj_error(y_hat, y, weight=None):
    if weight is None:
        weight = y.new_ones((1, 1))
    return (((weight[:, :, None] * (y - y_hat[..., :2])) ** 2).sum(dim=-1) ** 0.5).mean(
        dim=-2
    )


def draw_projtd_handjts(im, jts2d, jts_order, line_type=cv2.LINE_AA, colors=((0, 0, 255), (0, 0, 255))):
    """
    @param im: input image
    @param jts2d: 2d joints
    @param colors: joint colors(give same two set of colors for now to work as expected)
    @return: image with projected joints
    """
    # draw on image
    img_proj_pts = draw_hands(im=im, joints=jts2d, colors=colors, jts_order=jts_order, line_type=line_type)

    return img_proj_pts


def imgs2vid_ffmpeg(imgs_dir, file_pth, frm_rate=10):
    print(f"ffmpeg creating video...")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate {frm_rate} -pattern_type glob -i '{imgs_dir}/*.jpg' -c:v " \
          f"libx264 -vf fps=30 -pix_fmt yuv420p {file_pth} -y "
    os.system(cmd)
    # os.system(f"rm {imgs_dir}/*.jpg")

    return print(f"video saved here: {file_pth}")


def create_juxt_vid(filepath, inp_imgs, jts_order, all_2d_jts=None, all_3d_jts_rt=None, all_3d_jts_cam=None,
                    all_3d_jts_prcst_algnd=None):
    seq_len = inp_imgs.shape[0]
    tmp_dir = 'out/juxt/imgs'
    os.makedirs(tmp_dir, exist_ok=True)

    imgs = []
    res = None
    for i in tqdm(range(seq_len)):
        if all_2d_jts is not None:
            img1 = draw_projtd_handjts(im=inp_imgs[i], jts2d=all_2d_jts[i].reshape(-1, 21, 2), jts_order=jts_order,
                                       line_type=cv2.LINE_AA, colors=((255, 0, 0), (255, 0, 0)))  # red color
            res = img1.shape[0] // 2
            imgs.append(img1)

        if all_3d_jts_rt is not None:
            # img2 = viz_hand_jts3d(hand_jts=all_3d_jts_rt[i], jts_order='DOPE', grid_axis='ON', line_sz=2, dot_sz=2,
            # elev=15, azim=45, xlim=[-0.2, 0.2], ylim=[-0.2, 0.2], zlim=[-0.2, 0.2], resln=img1.shape[0] // 2,
            # title='3D Joints in wrist frame') elev = 15, azim = 45,
            img2 = viz_hand_jts3d(hand_jts=all_3d_jts_rt[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[all_3d_jts_rt[:, :, 0].min() - 0.1, all_3d_jts_rt[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_rt[:, :, 1].min() - 0.1, all_3d_jts_rt[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_rt[:, :, 2].min() - 0.1, all_3d_jts_rt[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints in wrist frame')
            if res is not None:
                img2 = np.concatenate([np.zeros((img1.shape[0] // 2, *img2.shape[1:]), np.uint8), img2])
            imgs.append(img2)

        if all_3d_jts_cam is not None:
            # img3 = viz_hand_jts3d(hand_jts=all_3d_jts_cam[i], jts_order='DOPE', grid_axis='ON', line_sz=1, dot_sz=1,
            #                       elev=-90, azim=-90, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], zlim=[-1.0, 1.0],
            #                       resln=img1.shape[0] // 2, title='3D Joints in camera Frame (wrist to camera frame)')
            img3 = viz_hand_jts3d(hand_jts=all_3d_jts_cam[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=2, elev=-90, azim=-90,
                                  xlim=[all_3d_jts_cam[:, :, 0].min() - 0.1, all_3d_jts_cam[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_cam[:, :, 1].min() - 0.1, all_3d_jts_cam[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_cam[:, :, 2].min() - 0.1, all_3d_jts_cam[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints in camera Frame (wrist to camera frame)')
            if res is not None:
                img3 = np.concatenate([np.zeros((img1.shape[0] // 2, *img3.shape[1:]), np.uint8), img3])
            imgs.append(img3)

        if all_3d_jts_prcst_algnd is not None:
            img4 = viz_hand_jts3d(hand_jts=all_3d_jts_prcst_algnd[i], jts_order=jts_order, grid_axis='ON',
                                  line_sz=2, dot_sz=4, elev=-90, azim=-90,
                                  xlim=[all_3d_jts_prcst_algnd[:, :, 0].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 0].max() + 0.1],
                                  ylim=[all_3d_jts_prcst_algnd[:, :, 1].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 1].max() + 0.1],
                                  zlim=[all_3d_jts_prcst_algnd[:, :, 2].min() - 0.1,
                                        all_3d_jts_prcst_algnd[:, :, 2].max() + 0.1],
                                  resln=res, title='3D Joints after procrust alignment')
            if res is not None:
                img4 = np.concatenate([np.zeros((img1.shape[0] // 2, *img4.shape[1:]), np.uint8), img4])
            imgs.append(img4)

        disp_img = np.hstack(imgs)
        # bb()
        Image.fromarray(disp_img).save(f"{tmp_dir}/{i:04d}.jpg")
        imgs = []
    # breakpoint()
    imgs2vid_ffmpeg(imgs_dir=tmp_dir, file_pth=filepath, frm_rate=30)


def draw_hand_jts_seq(all_imgs, all_2d_jts, jts_order, save_dir):
    for idx, (J, I) in tqdm(enumerate(zip(all_2d_jts, all_imgs))):
        img_jts_2d = draw_hands(im=I, joints=J.reshape(-1, 21, 2), colors=((255, 0, 0), (255, 0, 0)),
                                       circle_radius=3, line_thickness=2, offset=np.zeros(2, dtype=np.int),
                                       jts_order=jts_order, line_type=cv2.LINE_AA)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"frame{idx:04}.jpg"), img_jts_2d[..., ::-1])


def create_handskel_motion_vid(hand_jts_3d, jts_order, savepth, line_sz=1, dot_sz=1, elev=15, azim=45,
                               xlim=None, ylim=None, zlim=None, grid_axis='ON', title=None, resln=None):
    if not os.path.exists(os.path.dirname(savepth)):
        os.makedirs(os.path.dirname(savepth))

    seq_len = hand_jts_3d.shape[0]
    tmp_dir = 'out/juxt/imgs'
    os.makedirs(tmp_dir, exist_ok=True)

    if xlim is None:
        xlim = [hand_jts_3d[:, :, 0].min() - 0.1, hand_jts_3d[:, :, 0].max() + 0.1]
    if ylim is None:
        ylim = [hand_jts_3d[:, :, 1].min() - 0.1, hand_jts_3d[:, :, 1].max() + 0.1]
    if zlim is None:
        zlim = [hand_jts_3d[:, :, 2].min() - 0.1, hand_jts_3d[:, :, 2].max() + 0.1]

    for i in tqdm(range(seq_len)):
        img = viz_hand_jts3d(hand_jts=hand_jts_3d[i], jts_order=jts_order, grid_axis=grid_axis,
                             line_sz=line_sz, dot_sz=dot_sz, elev=elev, azim=azim,
                             xlim=xlim, ylim=ylim, zlim=zlim, resln=resln, title=title)
        Image.fromarray(img).save(f"{tmp_dir}/{i:03d}.jpg")
    imgs2vid_ffmpeg(imgs_dir=tmp_dir, file_pth=savepth, frm_rate=10)

    return None


def viz_hstack_imgs(imgs_lst, title=None, resln=None):
    """
    @param imgs_lst: list of images to be stacked for display
    (should be of same size)
    """
    plt.style.use('dark_background')
    # fig = Figure((resln / 100., resln / 100.))
    fig = plt.figure()
    canvas = FigureCanvas(fig)

    plt.imshow(np.hstack(imgs_lst)[:, :, ::-1])

    if title is not None:
        plt.title(title)

    canvas.draw()
    (width, height) = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()
    plt.close('all')

    if img.shape[0] != resln and resln is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((resln, resln)))

    return img


def draw_hand_jts_on_img(hand_jts, jts_order, img, color='r', linestyle='--', line_sz=2, dot_sz=2, resln=None):
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    plt.imshow(img)
    plt.scatter(hand_jts[:, 0], hand_jts[:, 1], c=color, s=0.5)

    if jts_order == 'CP':
        # Contactpose labels order
        thumb_idxs = [0, 1, 2, 3, 4]
        indxfing_idxs = [0, 5, 6, 7, 8]
        midfing_idxs = [0, 9, 10, 11, 12]
        ringfing_idxs = [0, 13, 14, 15, 16]
        litfing_idxs = [0, 17, 18, 19, 20]
    elif jts_order == 'DOPE':
        # DOPE dets order
        thumb_idxs = [0, 1, 6, 7, 8]
        indxfing_idxs = [0, 2, 9, 10, 11]
        midfing_idxs = [0, 3, 12, 13, 14]
        ringfing_idxs = [0, 4, 15, 16, 17]
        litfing_idxs = [0, 5, 18, 19, 20]
    elif jts_order == 'OURS':
        thumb_idxs = [0, 1, 6, 11, 16]
        indxfing_idxs = [0, 2, 7, 12, 17]
        midfing_idxs = [0, 3, 8, 13, 18]
        ringfing_idxs = [0, 4, 9, 14, 19]
        litfing_idxs = [0, 5, 10, 15, 20]
    else:
        ValueError("Specify hand joints order type like ('DOPE', 'CONACTPOSE'")
    plt.scatter(hand_jts[:, 0], hand_jts[:, 1], s=dot_sz, c=color, marker='o')
    plt.plot(hand_jts[thumb_idxs][:, 0], hand_jts[thumb_idxs][:, 1],
             linewidth=line_sz, linestyle=linestyle, color='r')
    plt.plot(hand_jts[indxfing_idxs][:, 0], hand_jts[indxfing_idxs][:, 1],
             linewidth=line_sz, linestyle=linestyle, color='g')
    plt.plot(hand_jts[midfing_idxs][:, 0], hand_jts[midfing_idxs][:, 1],
             linewidth=line_sz, linestyle=linestyle, color='b')
    plt.plot(hand_jts[ringfing_idxs][:, 0], hand_jts[ringfing_idxs][:, 1],
             linewidth=line_sz, linestyle=linestyle, color='y')
    plt.plot(hand_jts[litfing_idxs][:, 0], hand_jts[litfing_idxs][:, 1],
             linewidth=line_sz, linestyle=linestyle, color='m')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()
    plt.close('all')

    if img.shape[:2] != resln and resln is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize(resln))

    return img


def viz_hand_jts3d(hand_jts, jts_order, grid_axis='ON', line_sz=2, dot_sz=2, elev=15, azim=45, xlim=1., ylim=1.,
                   zlim=1., resln=None, title=None):
    """
    @param resln: resolution
    @param hand_jts: (N, K, 3)
    @return: visualized joints img
    """
    plt.style.use('dark_background')
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(projection='3d')

    if jts_order == 'CP':
        # hand skeleton structure
        # Contactpose labels order
        thumb_idxs = [0, 1, 2, 3, 4]
        indxfing_idxs = [0, 5, 6, 7, 8]
        midfing_idxs = [0, 9, 10, 11, 12]
        ringfing_idxs = [0, 13, 14, 15, 16]
        litfing_idxs = [0, 17, 18, 19, 20]
    elif jts_order == 'DOPE':
        # DOPE dets order
        thumb_idxs = [0, 1, 6, 7, 8]
        indxfing_idxs = [0, 2, 9, 10, 11]
        midfing_idxs = [0, 3, 12, 13, 14]
        ringfing_idxs = [0, 4, 15, 16, 17]
        litfing_idxs = [0, 5, 18, 19, 20]
    elif jts_order == 'OURS':
        thumb_idxs = [0, 1, 6, 11, 16]
        indxfing_idxs = [0, 2, 7, 12, 17]
        midfing_idxs = [0, 3, 8, 13, 18]
        ringfing_idxs = [0, 4, 9, 14, 19]
        litfing_idxs = [0, 5, 10, 15, 20]
    else:
        ValueError("Specify hand joints order type like ('DOPE', 'CONACTPOSE'")

    hand_jts = hand_jts.reshape(-1, 21, 3)
    for joints in hand_jts:
        ax.scatter3D(joints[:, 0], joints[:, 1], joints[:, 2], color='r', s=dot_sz)
        plt.plot(joints[thumb_idxs][:, 0], joints[thumb_idxs][:, 1], joints[thumb_idxs][:, 2], marker='.',
                 markersize=line_sz, color='r')
        plt.plot(joints[indxfing_idxs][:, 0], joints[indxfing_idxs][:, 1], joints[indxfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='g')
        plt.plot(joints[midfing_idxs][:, 0], joints[midfing_idxs][:, 1], joints[midfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='b')
        plt.plot(joints[ringfing_idxs][:, 0], joints[ringfing_idxs][:, 1], joints[ringfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='y')
        plt.plot(joints[litfing_idxs][:, 0], joints[litfing_idxs][:, 1], joints[litfing_idxs][:, 2], marker='.',
                 markersize=line_sz, color='m')

    # legends and ticks
    ax.view_init(elev, azim)  # 0 45 90 315
    ax.dist = 8

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # ax.set_xlabel('X axis', labelpad=-12)
    # ax.set_ylabel('Y axis', labelpad=-12)
    # ax.set_zlabel('Z axis', labelpad=-12)

    if isinstance(xlim, list) and len(xlim) == 2:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([-1 * xlim, xlim])
    if isinstance(ylim, list) and len(ylim) == 2:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-1 * ylim, ylim])
    if isinstance(zlim, list) and len(zlim) == 2:
        ax.set_zlim(zlim)
    else:
        ax.set_zlim([-1 * zlim, zlim])

    plt.title(title)

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    # switch off grid lines and axis info
    if grid_axis == 'OFF':
        plt.axis('off')
        plt.grid(b=None)

    # ax.set_proj_type('ortho')

    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()
    plt.close('all')

    if img.shape[0] != resln and resln is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((resln, resln)))

    return img


def viz_barplot(bars_batch, resln=None, bar_width=0.5):
    """
    bars_batch is of shape N x M x1; N is number of different category bars

    """
    import random
    print(f"this function is not completely implemented!!!")
    (N, M) = bars_batch.shape[:2]
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot()

    canvas = FigureCanvas(fig)
    X = range(M)
    for bars in bars_batch:
        ax.bar(X, bars.flatten(), color=random.sample('rgbymc', 1)[0], width=bar_width)

    canvas.draw()
    (width, height) = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.clf()
    plt.close('all')

    if img.shape[0] != resln and resln is not None:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((resln, resln)))

    return img


def load_npz(fpth):
    """
    loads the gt npz file and returns poses, cam_intrinsics, and affine
    :param fpth: path to npz file
    """
    data = np.load(fpth)
    intr = data.f.cam_intrinscs
    affine = data.f.affine
    poses = data.f.cam_poses

    return poses, intr, affine


def add_noise_to_cam_poses(cam_poses, rot_std=0, tr_std=0):
    """
    func from vincent script(cite author credit)
    """
    # rotation_stds = [0, 0, 0, 0, 2, 5, 10, 2, 5, 10]  # std of rotation noise in degrees
    # tr_stds = [0, 1e-2, 5e-2, 1e-1, 0, 0, 0, 1e-2, 5e-2, 1e-1]  # std of translation error in m

    cam_poses_noisy = []
    for cam in range(cam_poses.shape[0]):
        print(f"Adding noise to camera {cam}")
        breakpoint()
        axis = np.random.random(3)
        rod = np.random.normal(scale=rot_std) * (np.pi / 180.) * axis / np.linalg.norm(axis)
        rot = R.from_rotvec(rod)
        direction = np.random.random(3)
        tr = np.random.normal(scale=tr_std) * direction / np.linalg.norm(direction)

        new_rot = (R.from_matrix(cam_poses[cam][:3, :3]) * rot).as_matrix()
        new_tr = cam_poses[cam][:3, -1] + tr

        # Build extrinsics matrix
        extr = np.concatenate([new_rot, new_tr[:, None]], axis=-1)

        cam_poses_noisy.append(extr)

    return np.array(cam_poses_noisy)