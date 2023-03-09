
import os, cv2, glob

from evaluation.eval_utils import load_pkl
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb
import pandas as pd
import numpy as np





pths = ["/scratch/1/user/aswamy/data/briac_baseline/20220705173214/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220809161015/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220811154947/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220812172414/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220812180133/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220823115809/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220824142508/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220902170443/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220905153946/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220907155615/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220909114359/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220912142017/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220912143756/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220913145135/briac_recon_eval_sample200000_dth_0.0050.pkl",
"/scratch/1/user/aswamy/data/briac_baseline/20220913151554/briac_recon_eval_sample200000_dth_0.0050.pkl"]

all_seqs_dict = []
for pth in pths:
    # pkl_pth_5mm = osp.join(VH_RECON_RES_DIR, sqn, 'vh_recon_eval_sample200000_dth_0.0050.pkl')
    # pkl_pth = pathlib.Path(osp.join(BRIAC_RECON_RES_DIR, sqn, args.pkl_name))
    # import pathlib
    # if not pkl_pth.exists():
    #     print('missing pkl:', pkl_pth)
    #     continue
    sqn = pth.split('/')[-2]
    data = load_pkl(str(pth))
    # data.update({'sqn' : sqn})
    
    new_dict = {
        'sqn' : sqn,
        'accu' : f"{data['accuracy_rec']:.4f}",
        'comp' : f"{data['completion_rec']:.4f}",
        'precision_ratio' : f"{data['precision_ratio_rec']*100:.4f}",
        'comp_ratio' : f"{data['completion_ratio_rec']*100:.4f}",
        'fscore' : f"{data['fscore']:.4f}",
        'normal_acc' : f"{data['normal_acc']:.4f}",
        'normal_comp' : f"{data['normal_comp']:.4f}",
        'normal_avg' : f"{data['normal_avg']:.4f}",
    }

    # bb()
    df = pd.DataFrame.from_dict([new_dict])
    all_seqs_dict.append(df)

df_comb = pd.concat(all_seqs_dict)
# df_comb_save_pth = osp.join(RES_DIR, args.pkl_name.replace('pkl', 'csv'))
df_comb.to_csv('./missing_briac_evals.csv')

print('Done!!')
