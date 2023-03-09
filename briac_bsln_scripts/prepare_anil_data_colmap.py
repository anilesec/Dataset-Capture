#!/usr/bin/env python3

from codecs import ascii_encode
from re import S
import math
from sys import path_hooks
from evaluation.viz_utils import create_handskel_motion_vid
import numpy as np
import cv2
import xml.etree.cElementTree as ET
from xml.dom import minidom
import os
import json

print("This script prepares Anil's data.")
#dir = "/home/btoussai/Desktop/briac/docker-test/anil_d0ata"
# dir="/scratch/aerappan/data/anil_data"
# dir = "/scratch/aerappan/data/hand-obj"
# dir = "/morpheo-nas/aerappan/briac_baseline"
dir = "/home/aswamy/server_mounts/scratch1/user/aswamy/data/briac_recon_colmap_poses_res"

np.set_printoptions(suppress=True)

width = 1280
height = 720
stride  = 8 

def parseCameraPoses_dope(path):
    print('Using dope poses...')
    global nbCams
    import glob
    pths = sorted(glob.glob(os.path.join(path, 'dope_rel_poses/*.txt')))
    assert len(pths) > 0
    transforms = np.array(
        [np.linalg.inv(np.loadtxt(pth)) for pth in pths]
    )
    return transforms

def parseCameraPoses_colmap(path):
    print('Using colmap poses...')
    global nbCams
    import glob

    pths = sorted(glob.glob(os.path.join(path, 'cam_poses/*.txt')))
    print(os.path.join(path, 'cam_poses/*.txt'))
    assert len(pths) > 0
    transforms = np.array(
        [np.linalg.inv(np.loadtxt(pth)) for pth in pths]
    )

    cipths = sorted(glob.glob(os.path.join(path, 'cam_poses/*.txt')))
    assert len(pths) > 0
    intrinsics = np.array(
        [np.loadtxt(pth) for pth in cipths]
    )
    return transforms, intrinsics


def parseCameraPoses(path):
    global nbCams
    path = path + "/icp_res"

    dirs = os.listdir(path)
    list.sort(dirs)

    nbCams = min(128, len(dirs))

    transforms = []

    for filename in dirs:
        f = os.path.join(path, filename, "f_trans.txt")
        f = open(f)
        l = np.array([[float(u) for u in s.split()] for s in f.readlines()])
        transforms.append(l)
    return transforms


def writeCalib(path, transforms, intrinsics, dest):

    nbCam = min(128, len(transforms) // stride)
    
    
    K = np.eye(3)
    K[0, 0] = 899.783
    K[1, 1] = 900.019
    K[0, 2] = 653.768
    K[1, 2] = 362.143

    CalibrationResult = ET.Element("CalibrationResult")
    Information = ET.SubElement(CalibrationResult, "Information")
    Information.set('nbCamera', str(nbCam))
    Information.set('units', "mm")
    Information.set('up', "+z")

    for i in range(0, min(nbCam, 128)):
        cam_intr = intrinsics[i]
        # cam_intr[0, 0] /= 10
        # cam_intr[1, 1] /= 10
        K = cam_intr

        cameraNode = ET.SubElement(CalibrationResult, "Camera")
        cameraNode.set("id", str(i))
        cameraNode.set("width", str(width))
        cameraNode.set("height", str(height))

        P = transforms[i*stride]
        P = np.linalg.inv(P)
        P = P[:3]

        row0 = np.copy(P[0])
        row1 = np.copy(P[1])
        row2 = np.copy(P[2])

        #P[1] = -row1
        #P[2] = -row2

        R = P[:, :3]
        T = P[:, 3]

        K_Node = ET.SubElement(cameraNode, "K")
        K_Node.text = ' '.join(str(k) for k in np.ravel(K))
        Distortion_Node = ET.SubElement(cameraNode, "Distortion")
        Distortion_Node.set("model", "opencv")
        Distortion_Node.set("K1", str(0.0))
        Distortion_Node.set("K2", str(0.0))
        Distortion_Node.set("P1", str(0.0))
        Distortion_Node.set("P2", str(0.0))
        Distortion_Node.set("K3", str(0.0))

        R_Node = ET.SubElement(cameraNode, "R")
        R_Node.text = ' '.join(str(r) for r in np.ravel(R))
        T_Node = ET.SubElement(cameraNode, "T")
        T_Node.text = ' '.join(str(t) for t in np.ravel(T))


    xmlstr = minidom.parseString(ET.tostring(CalibrationResult)).toprettyxml(indent="   ")
    f = open(path + "/" + dest, "w")
    f.write(xmlstr)
    f.close()


def writeSequence(path, transforms):
    nbCam = min(128, len(transforms)//stride)

    Video = ET.Element("Video")
    Information = ET.SubElement(Video, "Information")
    Information.set('nbCamera', str(nbCam))
    Information.set('fps', "30")
    Information.set('beginId', "0")
    Information.set('endId', "0")

    for i in range(0, nbCam):
        cameraNode = ET.SubElement(Video, "Camera")
        cameraNode.set("id", str(i))
        cameraNode.set("path", path + "/rgb")
        cameraNode.set("fileNameFormat", "{:010d}.png".format(i*stride))

    xmlstr = minidom.parseString(ET.tostring(Video)).toprettyxml(indent="   ")
    f = open(path + "/sequence.xml", "w")
    f.write(xmlstr)
    f.close()

def makeBackgrounds(path, transforms):
    os.makedirs(path + "/Backgrounds", exist_ok=True)
    nbCam = min(128, len(transforms) // stride)

    for i in range(0, nbCam):
        print("Making background ", i)
        # print(path + "/slvless_img_bin/{:010d}.png".format(i*stride), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(path + "/slvless_img_bin/{:010d}.png".format(i*stride), cv2.IMREAD_UNCHANGED)
        print(path + "/slvless_img_bin/{:010d}.png".format(i*stride))
        # fg = img[:, :, 1] > 30
        fg = img[:, :] > 0

        img2 = np.full((height, width, 4), 0, dtype=np.uint8)
        img2[fg] = np.array([0, 0, 255, 255])
        p = path + "/Backgrounds/cam-{}.png".format(i)
        cv2.imwrite(p, img2)

def createSceneSettings_auto(outDir, vhm_pth):
    import trimesh

    gt_pcd = trimesh.load(vhm_pth)
    bbox = trimesh.bounds.corners(gt_pcd.bounding_box.bounds)
    centers = bbox.mean(0)
    extents_max = gt_pcd.extents.max()

    dest_path = "{}/scene_config.txt".format(outDir)
    print("Writing " + dest_path + " ...")
    dest_file = open(dest_path, 'w')

    lines = []
    lines.append("DifferentialRendering#activeCameras ivec4 -1 -1 -1 -1")
    lines.append("SceneReferenceFrame#FrustumsLength float 0.2")
    lines.append("SceneReferenceFrame#Rotation vec3 0.000000 0.000000 0.0")
    lines.append("SceneReferenceFrame#ScaleMatrixRow0 vec4 1.000000 0.000000 0.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow1 vec4 0.000000 1.000000 0.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow2 vec4 0.000000 0.000000 1.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow3 vec4 0.000000 0.000000 0.000000 1.000000")
    lines.append("SceneReferenceFrame#SceneScale float {}".format(extents_max / 1.5))
    lines.append("SceneReferenceFrame#Translation vec3 {} {} {}".format(centers[0], centers[1], centers[2]))
    lines.append("SparseGrid#TSDFTilesDim vec3 250.000000 250.000000 250.000000")
    lines.append("SparseGrid#VolumeMinCorner vec3 -1 -1 -1")
    lines.append("SparseGrid#VoxelSize float 2.000000")
    lines.append("Window#BackgroundsDir string unknown")
    lines.append("Window#CalibrationFile string unknown/calib.xml")
    lines.append("Window#ExportDir string unknown")
    lines.append("Window#SequenceFile string unknown/sequence.xml")
    lines.append("Window#imagePyramidLevels int 4")
    lines.append("Window#segmentationPyramidLevel int 1")

    for l in lines:
        l += "\n"
        if l.startswith("Window#BackgroundsDir"):
            l = "Window#BackgroundsDir string {}\n".format(outDir + "/Backgrounds")
        elif l.startswith("Window#CalibrationFile"):
            l = "Window#CalibrationFile string {}/calib.xml\n".format(outDir)
        elif l.startswith("Window#ExportDir"):
            l = "Window#ExportDir string {}\n".format(outDir)
        elif l.startswith("Window#SequenceFile"):
            l = "Window#SequenceFile string {}/sequence.xml\n".format(outDir)
        dest_file.write(l)

def createSceneSettings(outDir):
    dest_path = "{}/scene_config.txt".format(outDir)
    print("Writing " + dest_path + "...")
    dest_file = open(dest_path, 'w')

    lines = []
    lines.append("DifferentialRendering#activeCameras ivec4 -1 -1 -1 -1")
    lines.append("SceneReferenceFrame#FrustumsLength float 0.2")
    lines.append("SceneReferenceFrame#Rotation vec3 0.000000 0.000000 0.0")
    lines.append("SceneReferenceFrame#ScaleMatrixRow0 vec4 1.000000 0.000000 0.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow1 vec4 0.000000 1.000000 0.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow2 vec4 0.000000 0.000000 1.000000 0.000000")
    lines.append("SceneReferenceFrame#ScaleMatrixRow3 vec4 0.000000 0.000000 0.000000 1.000000")
    lines.append("SceneReferenceFrame#SceneScale float 0.10000")
    lines.append("SceneReferenceFrame#Translation vec3 0.000000 0.000000 0.000000")
    lines.append("SparseGrid#TSDFTilesDim vec3 300.000000 300.000000 300.000000")
    lines.append("SparseGrid#VolumeMinCorner vec3 -1 -1 -1")
    lines.append("SparseGrid#VoxelSize float 4.000000")
    lines.append("Window#BackgroundsDir string unknown")
    lines.append("Window#CalibrationFile string unknown/calib.xml")
    lines.append("Window#ExportDir string unknown")
    lines.append("Window#SequenceFile string unknown/sequence.xml")
    lines.append("Window#imagePyramidLevels int 4")
    lines.append("Window#segmentationPyramidLevel int 1")

    for l in lines:
        l += "\n"
        if l.startswith("Window#BackgroundsDir"):
            l = "Window#BackgroundsDir string {}\n".format(outDir + "/Backgrounds")
        elif l.startswith("Window#CalibrationFile"):
            l = "Window#CalibrationFile string {}/calib.xml\n".format(outDir)
        elif l.startswith("Window#ExportDir"):
            l = "Window#ExportDir string {}\n".format(outDir)
        elif l.startswith("Window#SequenceFile"):
            l = "Window#SequenceFile string {}/sequence.xml\n".format(outDir)
        dest_file.write(l)

def prep_dataset(sqn):
    base_clmp_pose_dir = '/home/aswamy/server_mounts/scratch1/user/aswamy/data/colmap-hand-obj'
    RES_DIR = '/home/aswamy/server_mounts/scratch1/user/aswamy/data/hand-obj'
    tgt_base_dir = '/home/aswamy/server_mounts/scratch1/user/aswamy/data/briac_recon_colmap_poses_res'
    
    sqn_pth = os.path.join(tgt_base_dir, sqn)
    cmd1 = f"mkdir -p {sqn_pth}" ; 
    # print(cmd1)
    # os.system(cmd1)

    cmd2 = f"cp -vr {os.path.join(RES_DIR, sqn, 'rgb')} {sqn_pth}"
    print(cmd2)
    os.system(cmd2)

    cmd3 = f"cp -vr {os.path.join(RES_DIR, sqn, 'gt_mesh')} {sqn_pth}"
    print(cmd3)
    os.system(cmd3)

    cmd4 = f"cp -vr {os.path.join(RES_DIR, sqn, 'slvless_img_bin')} {sqn_pth}"
    print(cmd4)
    os.system(cmd4)

    # cmd5 = f"rsync --progress -auvz --exclude '*.ply' {os.path.join(RES_DIR, sqn, 'icp_res')} {sqn_pth}"
    # print(cmd5)
    # os.system(cmd5)

    # cmd5_1 = f"rm {os.path.join(tgt_base_dir, sqn, 'icp_res/*/m3trans.txt')}"
    # print(cmd5_1)
    # os.system(cmd5_1)

    cmd6 = f"cp -vr {os.path.join(base_clmp_pose_dir, sqn, 'cam_poses')} {sqn_pth}"
    print(cmd6)
    os.system(cmd6)

    cmd7 = f"cp -vr {os.path.join(base_clmp_pose_dir, sqn, 'cam_intrs')} {sqn_pth}"
    print(cmd7)
    os.system(cmd7)

    return None


if __name__ == "__main__":
    import argparse
     
    # scenes = os.listdir(dir)
    import glob
    COLMAP_VH_DIR = '/home/aswamy/server_mounts/scratch1/user/vleroy/HandOBJ/VH_colmap'
    clmp_vh_mpths = sorted(glob.glob(os.path.join(COLMAP_VH_DIR, '*/VH/*.ply')))
    scenes = [clmp_vh_mpth.split('/')[-3] for clmp_vh_mpth in clmp_vh_mpths]

    colmap_scenes_dets_ratio_1_0 = ['20220907153810', '20220902154737', '20220909114359', '20220902163904', '20220811170459', '20220912155637', '20220905140306', '20220902114024', '20220902111535', '20220909111450', '20220913153520', '20220824105341', '20220909113237', '20220824181949', '20220909142411', '20220912143756', '20220905105332', '20220902170443', '20220905112733', '20220913144436', '20220902163950', '20220912164407', '20220829154032', '20220912165455', '20220811171507', '20220824155144', '20220909152911', '20220902164854', '20220907152036', '20220905141444', '20220824180652', '20220912142017', '20220912152000', '20220809170847']
    scenes = ['20220811170459']
    print(scenes)
    skipped_sqns = []
    for scene in scenes:
        if not scene in colmap_scenes_dets_ratio_1_0:
            print('Skipping:', scene)
            skipped_sqns.append(scene)
            continue
        assert len(scene) == 14
        print('Preparing dataset...')
        prep_dataset(scene)

        print("Preparing scene: " + scene)
        path = dir + "/" + scene
        # transforms = parseCameraPoses(path)
        transforms, intrinsics = parseCameraPoses_colmap(path)
        # transforms = parseCameraPoses_dope(path)

    
        writeCalib(path, transforms, intrinsics, "calib.xml")
        writeSequence(path, transforms)
        makeBackgrounds(path, transforms)
        vh_mesh_pth = os.path.join(COLMAP_VH_DIR, scene, 'VH/VHull_000000_clean.ply')
        createSceneSettings_auto(outDir=path, vhm_pth=vh_mesh_pth)

    print('skipped seqs:', skipped_sqns)
    print("Done.")
    print("You can now run the main program, ")
    print("then select a reconstruction by selecting 'file>import scene config' in the menu bar and navigating to the desired config file.")
