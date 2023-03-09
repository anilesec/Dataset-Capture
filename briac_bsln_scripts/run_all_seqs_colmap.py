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
    # hand_obj_dir = "/scratch/aerappan/data/hand-obj"
    # scenesHo = os.listdir(hand_obj_dir)
    # scenesHoDir = "/scratch/aerappan/data/hand-obj/{}"

    hand_obj_dir = "/home/aswamy/server_mounts/scratch1/user/aswamy/data/briac_recon_colmap_poses_res"
    scenesHo = os.listdir(hand_obj_dir)
    scenesHoDir = "/home/aswamy/server_mounts/scratch1/user/aswamy/data/briac_recon_colmap_poses_res/{}"

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

    sqns_lst =  "20220811154947 20220812172414 20220812174356 20220824144438 20220824180652 20220902115034 20220902151726 20220902154737 20220905112733 20220819164041 20220902111535 20220902114024 20220902163904 20220902163950 20220902164854 20220902170443 20220905105332 20220905111237 20220905112623 20220905140306 20220809170847"
    list = sqns_lst.split(' ')
    
    bb()
    os.chdir(exec_dir)
    
    from tqdm import tqdm
    for l in tqdm(list):
        print('scene', l)

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
