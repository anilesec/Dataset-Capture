import os, glob
from ipdb import set_trace as bb
import numpy as np
from evaluation.viz_utils import *
from evaluation.eval_utils import *

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    #plt.show()
    #cv2.waitKey(0)

    plt.savefig(filename)

jts3d_hoc = np.loadtxt('/scratch/1/user/aswamy/data/hand-obj/20220705173214/jts3d_hocnet/0000000360.txt') #* np.array([1,-1,-1])
jts3d_ann = np.loadtxt('/scratch/1/user/aswamy/data/hand-obj/20220705173214/jts3d/0000000360.txt')

# bb()

root_ann = jts3d_ann[0]
jts3d_hoc_align = jts3d_hoc - jts3d_hoc[0]
jts3d_hoc_tranl = jts3d_hoc_align + root_ann
T = np.eye(4)[:3, :]
T[:3, 3] = root_ann
trnsfm = CAM_INTR @ np.eye(4)[:3, :]
jts2d_hoc = project(trnsfm, jts3d_hoc_tranl)
jts2d_ann = project(trnsfm, jts3d_ann)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(jts3d_hoc[:, 0], jts3d_hoc[:, 1], jts3d_hoc[:, 2], marker='o', color='r')
# plt.scatter(jts3d_ann[:, 0], jts3d_ann[:, 1], jts3d_ann[:, 2], marker='*', color='g')
# plt.savefig('./tmp.png')

bb()

im = cv2.imread('/scratch/1/user/aswamy/data/hand-obj/20220705173214/rgb/0000000360.png')
draw_hand_jts_seq(im[:, :, ::-1][None], jts2d_ann[None], 'OURS', './')

print(jts2d_hoc)


