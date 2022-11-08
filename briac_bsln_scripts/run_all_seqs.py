#!/usr/bin/env python3
import os.path
# import cv2
import numpy as np
import math
# import tkinter
import subprocess

def eval_dtu(scan):
    scanDir = "/home/btoussai/Desktop/briac/DTU_idr/scan" + str(scan)
    groundTruthFormat = scanDir + "/image/{:06d}.png"
    rasterDir = scanDir + "/eval_rasterization/r_{}.png"
    volumeRenderDir = scanDir + "/eval_volumeRendering/r_{}.png"
    masksFormat = scanDir + "/mask/{:03d}.png"


def eval_nerfsynthetic(scene):
    scanDir = "/home/btoussai/Desktop/briac/nerf_synthetic/" + scene
    groundTruthFormat = scanDir + "/test/r_{}.png"
    rasterDir = scanDir + "/eval_rasterization/r_{}.png"
    backgroundsFormat = scanDir + "/test/r_{}.png"


if __name__ == '__main__':
    # dtu_dir = tkinter.filedialog.askdirectory()
    exec_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
    exec_file = exec_dir + "/release/KinovisReconstruction"
    print(exec_dir, exec_file)
    # exec_dir = "/scratch/aerappan/briac_sw_installation/kinovisreconstruction"
    # exec_file = "/scratch/aerappan/briac_sw_installation/kinovisreconstruction"

    scansDTU = [122, 37, 24, 65, 106, 110, 114]
    scansDTUDir =  "/home/btoussai/Desktop/briac/DTU_idr/scan{}"

    scenesNerf = ["ship", "mic", "materials", "lego", "hotdog", "ficus", "drums", "chair"]  
    scenesNerfDir = "/home/btoussai/Desktop/briac/nerf_synthetic/{}"

    import os
    hand_obj_dir = "/scratch/aerappan/data/hand-obj"
    scenesHo = os.listdir(hand_obj_dir)
    scenesHoDir = "/scratch/aerappan/data/hand-obj/{}"

    testType = "hand-obj"

    dir = None
    list = None
    if testType == "DTU":
        dir = scansDTUDir
        list = scansDTU
    elif testType == "NERF":
        dir = scenesNerfDir
        list = scenesNerf
    elif testType == "Kinovis":
        dir = "/home/btoussai/Desktop/briac/export_sequence_julien"
        list = ["."]
    elif testType == 'hand-obj':
        dir = scenesHoDir
        list = scenesHo

    os.chdir(exec_dir)
    for l in list:
        SceneConfigPath = dir.format(l) + "/scene_config.txt"

        args = [
            "Window#AutoSave bool 1",
            "Window#AutoLoad bool 1",
            "Window#SceneConfigPath string {}".format(SceneConfigPath),
            "Sequence#FullAuto bool 1",
            "Sequence#AutoEval bool 1",
            "Window#AutoQuit bool 1",
            "Window#SHBands int 1"
        ]

        output = dir.format(l) + "/output.txt"
        output = open(output, "w")

        popen = subprocess.Popen([exec_file, *args], stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line, end='') 
            output.write(stdout_line)
        output.close()
        popen.stdout.close()
        return_code = popen.wait()

        # break


    # eval_dtu(114)
    # eval_nerfsynthetic("ship")
