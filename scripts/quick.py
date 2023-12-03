
import os, cv2, glob

from evaluation.eval_utils import load_pkl
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb
import pandas as pd
import numpy as np
import pickle

# tex_data = pd.read_csv('/Users/aswamy/My_drive/github_repos/Dataset-Capture/dope_hhor_recon_tex_notok.csv')
# nonsmall_data = pd.read_csv('/Users/aswamy/My_drive/github_repos/Dataset-Capture/briac_recon_colmap_non-small_objs.csv')   
# bb()



# get small_obj_sqns

# get large obj sqns

# get good tex sqns

# get not good tex sqn



# # load all object meshes
# obj_pths = glob.glob('/scratch/1/user/aswamy/data/hand-obj/object_meshes/*/obj_mesh/*.obj')
# import trimesh

# meshes = dict()
# for p in tqdm(obj_pths):
#     meshes[osp.basename(p)] = trimesh.load(p)
# keys = list(meshes.keys())
# bb()

# print('Done!')


showme_tes_sqns = os.listdir('/scratch/2/user/aswamy/projects/hhor_evaluation/HHOR_v2/HHOR/NeuS/exp/showme_sra_tra/subsample_showme-sra-tra_max60_iter100k_precomp_radius_origin')
base_dir = '/scratch/2/user/aswamy/projects/hhor_evaluation/HHOR_v2/HHOR/NeuS/exp/showme_sra_tra/subsample_showme-sra-tra_max60_iter100k_precomp_radius_origin'
all_seqs_dict = []
for sqn in showme_tes_sqns:
    pklpth = os.path.join(base_dir, sqn, 'eval_meshes_256/sra_tra_hhor_recon_eval_sample20000_dth_0.005.pkl')
    with open (pklpth, 'rb') as f:
        data = pickle.load(f)
        new_dict = {
            'sqn' : sqn,
            'accu' : f"{data['accuracy_rec']*100:.4f}",
            'comp' : f"{data['completion_rec']*100:.4f}",
            'precision_ratio' : f"{data['precision_ratio_rec']*100:.4f}",
            'comp_ratio' : f"{data['completion_ratio_rec']*100:.4f}",
            'fscore' : f"{data['fscore']:.4f}",
            # 'normal_acc' : f"{data['normal_acc']:.4f}",
            # 'normal_comp' : f"{data['normal_comp']:.4f}",
            # 'normal_avg' : f"{data['normal_avg']:.4f}",
        }
        df = pd.DataFrame.from_dict([new_dict])
        all_seqs_dict.append(df)

df_comb = pd.concat(all_seqs_dict)
df_comb_save_pth = osp.join(f'{base_dir}.csv')
df_comb.to_csv(df_comb_save_pth)
print(f"Saved here: {df_comb_save_pth}")
print('Done!!')