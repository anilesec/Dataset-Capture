#!/usr/bin/env python3

from re import S
import math
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
dir = "/home/aswamy/server_mounts/scratch1/user/aswamy/data/briac_baseline"

np.set_printoptions(suppress=True)

width = 1280
height = 720
stride  = 8

def parseCameraPoses_colmap(path):
    global nbCams
    path = path + "/cam_poses"
    

def parseCameraPoses(path):
    global nbCams
    path = path + "/icp_res"

    dirs = os.listdir(path)
    list.sort(dirs)

    nbCams = len(dirs)

    transforms = []

    for filename in dirs:
        f = os.path.join(path, filename, "f_trans.txt")
        f = open(f)
        l = np.array([[float(u) for u in s.split()] for s in f.readlines()])
        transforms.append(l)
    return transforms


def writeCalib(path, transforms, dest):

    nbCam = len(transforms) // stride

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
    nbCam = len(transforms)//stride

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
    nbCam = len(transforms) // stride

    for i in range(0, nbCam):
        print("Making background ", i)
        # print(path + "/slvless_img_bin/{:010d}.png".format(i*stride), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(path + "/slvless_img_bin/{:010d}.png".format(i*stride), cv2.IMREAD_UNCHANGED)
        # print(path + "/slvless_img_bin/{:010d}.png".format(i*stride))
        # fg = img[:, :, 1] > 30
        fg = img[:, :] > 0

        img2 = np.full((height, width, 4), 0, dtype=np.uint8)
        img2[fg] = np.array([0, 0, 255, 255])
        p = path + "/Backgrounds/cam-{}.png".format(i)
        cv2.imwrite(p, img2)

def createSceneSettings_auto(outDir):
    import trimesh

    gt_pcd = trimesh.load(os.path.join(outDir, 'gt_mesh/xyz/tgt_pcd.ply'))
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


scenes = os.listdir(dir)
print(scenes)
scenes = ["20220705173214"]
print(scenes)
for scene in scenes:
    print("Preparing scene: " + scene)
    path = dir + "/" + scene
    transforms = parseCameraPoses(path)

    writeCalib(path, transforms, "calib.xml")
    writeSequence(path, transforms)
    makeBackgrounds(path, transforms)
    createSceneSettings_auto(path)
    


print("Done.")
print("You can now run the main program, ")
print("then select a reconstruction by selecting 'file>import scene config' in the menu bar and navigating to the desired config file.")
