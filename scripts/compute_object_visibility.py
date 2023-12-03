import os, glob, pickle
import numpy as np
from ipdb import set_trace as bb
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

RES_DIR = '/scratch/1/user/aswamy/data/hand-obj'
if __name__ == "__main__":
    sqn = '20220811170459'

    sem_dir = "/scratch/2/user/aswamy/projects/hhor_evaluation/HHOR/data/DEMO_OUT/20220811170459/colmap/semantics"

    sem_pths = sorted(glob.glob(f"{sem_dir}/*.png"))

    all_frms_obj_pixs_cnt = []
    for pth in tqdm(sem_pths):
        sem = np.array(Image.open(pth))
        obj_mask = sem[:, :, 0] > 0
        all_frms_obj_pixs_cnt.append(obj_mask.sum())
    all_frms_obj_pixs_cnt = np.array(all_frms_obj_pixs_cnt) / max(all_frms_obj_pixs_cnt) * 100
    plt.plot(all_frms_obj_pixs_cnt)
    plt.savefig('./obj_vis_ratio.png')

    # load poses
    rel_rot_err = np.load('/gfs-ssd/user/aswamy/github_repos/Dataset-Capture/20220811170459_rot_err.npy')

    # df = {
    #     'rel_rot_err' : rel_rot_err,
    #     'all_frms_obj_pixs_cnt' : all_frms_obj_pixs_cnt
    # }
    # plot = sns.lineplot(df, x="Frame Number", y="Red(degree)/Green(%)")
    # fig = plot.get_figure()
    # fig.savefig('./roterr_vs_objvis.pdf') 

    plt.figure()
    plt.plot(rel_rot_err, c='r', label='RelRotErr')
    plt.plot(all_frms_obj_pixs_cnt, c='b', label='ObjVisibilityRatio')
    plt.xlabel('Frame Number', fontsize=10)
    plt.ylabel('Red(degree) / Blue(%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.title('Rotation Error vs. Object Visibility Ratio', fontsize=10)
    plt.savefig('./roterr_vs_objvis.pdf')
    print('Done!!')
