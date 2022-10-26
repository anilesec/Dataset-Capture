import cv2
import numpy as np
from ipdb import set_trace as bb

def rtvec2rtmat(rvec, tvec):
    rtmat = np.eye(4)
    rtmat[:3, :3] = cv2.Rodrigues(rvec)[0]
    rtmat[:3, 3] = tvec.flatten()

    return rtmat[:3, :]

def cv2_PNPSolver(obj_pts_lst, img_pts_lst, intrinsics, dist_coeffs=np.zeros(4), method=None):
    """
    # Minimize the reprojection error, default method flag set to SOLVEPNP_ITERATIVE
    @param obj_pts_lst: 3d object points
    @param img_pts_lst: 2d objec points on image
    @param intrinsics: camera intrinsic matrix
    @param dist_coeffs: distortioni coeffs, deafult is zero
    @return: returns the rotation and the translation vectors that transform a 3D point
             expressed in the object/3d points' coordinate frame to the camera coordinate frame
    """
    r2cs = []
    rvec = np.zeros(3, dtype=np.float)
    tvec = np.array([0, 0, 1], dtype=np.float)
    for obj_pts, img_pts in zip(obj_pts_lst, img_pts_lst):
        if method is None or method == 'PNP_ITERATIVE':
            _, rvec, tvec = cv2.solvePnP(objectPoints=obj_pts, imagePoints=img_pts,
                                         cameraMatrix=intrinsics, distCoeffs=dist_coeffs,
                                         rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE
                                         )           
        elif method == 'PNP_SQPNP':
            rvec = np.zeros(3, dtype=np.float)
            tvec = np.array([0, 0, 1], dtype=np.float)
            _, rvec, tvec = cv2.solvePnP(objectPoints=obj_pts, imagePoints=img_pts,
                                         cameraMatrix=intrinsics, distCoeffs=dist_coeffs,
                                         rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE
                                         )
        r2cs.append(rtvec2rtmat(rvec, tvec))
    r2cs = np.array(r2cs)

    return r2cs


def cv2_PNPRansacSolver(obj_pts, img_pts_lst, intrinsics, dist_coeffs=np.zeros(4)):
    """
    Finds an object pose from 3D-2D point correspondences
    @param obj_pts: rigid object points(reliable frame's 3d hand joints)
    @param img_pts_lst: 2d obj points of all frames
    @param intrinsics: camera intrinsic matrix
    @param dist_coeffs: distortion coeffs
    @return obj_poses: relative obj poses wrt to first frame's obj pose
    """
    obj_poses = []
    rvec = np.zeros(3, dtype=np.float)
    tvec = np.array([0, 0, 1], dtype=np.float)
    for img_pts in img_pts_lst:
        # _, rvec, tvec, __ = cv2.solvePnPRansac(obj_pts, img_pts, intrinsics, dist_coeffs)
        _, rvec, tvec, __ = cv2.solvePnPRansac(objectPoints=obj_pts, imagePoints=img_pts,
                                               cameraMatrix=intrinsics, distCoeffs=dist_coeffs,
                                               rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                               flags=cv2.SOLVEPNP_ITERATIVE
                                               )
        obj_poses.append(rtvec2rtmat(rvec, tvec))
    obj_poses = np.array(obj_poses)

    return obj_poses


if "__name__" == "__main__":
    pass
