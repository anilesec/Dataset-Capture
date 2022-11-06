import os, cv2, glob

from evaluation.eval_utils import load_pkl
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb
import pandas as pd
import numpy as np

RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')
VH_RECON_RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/vh_recon_res')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None, help='seq id')
    parser.add_argument('--start_ind', type=int, default=None, help='start sqn ind')
    parser.add_argument('--end_ind', type=int, default=None, help='end sqn ind')
    parser.add_argument('--pkl_name', type=str, required=True, help='pkl file name')
    args = parser.parse_args()

    # get all sqn ids
    all_sqns = os.listdir(VH_RECON_RES_DIR)
    
    if args.sqn is not None:
        assert args.sqn in all_sqns, f"{args.sqn} is not present in listed sequences!!!"
        all_sqns = [args.sqn]
    

    all_seqs_dict = []
    for sqn in tqdm(all_sqns[args.start_ind: args.end_ind]):
        print(f"seq: {sqn}")

        # pkl_pth_5mm = osp.join(VH_RECON_RES_DIR, sqn, 'vh_recon_eval_sample200000_dth_0.0050.pkl')
        pkl_pth = osp.join(VH_RECON_RES_DIR, sqn, args.pkl_name)
        data = load_pkl(pkl_pth)
        # data.update({'sqn' : sqn})
        
        new_dict = {
            'sqn' : sqn,
            'accu' : f"{data['accuracy_rec']:.4f}",
            'comp' : f"{data['completion_rec']:.4f}",
            'precision_ratio' : f"{data['precision_ratio_rec']:.4f}",
            'comp_ratio' : f"{data['completion_ratio_rec']:.4f}",
            'fscore' : f"{data['fscore']:.4f}",
            'normal_acc' : f"{data['normal_acc']:.4f}",
            'normal_avg' : f"{data['normal_avg']:.4f}",
        }

        # bb()
        df = pd.DataFrame.from_dict([new_dict])
        all_seqs_dict.append(df)

    df_comb = pd.concat(all_seqs_dict)
    df_comb_save_pth = osp.join(RES_DIR, args.pkl_name.replace('pkl', 'csv'))
    df_comb.to_csv(df_comb_save_pth)
    print(f"Saved here: {df_comb_save_pth}")
    print('Done!!')



    