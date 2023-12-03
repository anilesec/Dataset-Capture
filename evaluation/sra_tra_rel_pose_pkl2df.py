import os, cv2, glob

from evaluation.eval_utils import load_pkl
osp = os.path
import pathlib
from tqdm import tqdm
from ipdb import set_trace as bb
import pandas as pd
import numpy as np

RES_DIR = pathlib.Path('/scratch/1/user/aswamy/data/hand-obj')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('compute relative poses')
    parser.add_argument('--sqn', type=str, default=None, help='seq id')
    parser.add_argument('--start_ind', type=int, default=None, help='start sqn ind')
    parser.add_argument('--end_ind', type=int, default=None, help='end sqn ind')
    args = parser.parse_args()

    # select all the sids with .tar 
    all_seqs_tar_pths = glob.glob(f"{RES_DIR}/*.tar")
    all_sqns = []
    for spth in all_seqs_tar_pths:
        if os.path.isfile(spth):
            all_sqns.append(osp.basename(spth.split('.')[0]))
    print(f"All seq ids: \n {all_sqns}")

    if args.sqn is not None:
        all_sqns = [args.sqn]
    
    all_seqs_dict = []
    missing_pkl_seqs = []
    for sqn in tqdm(all_sqns[args.start_ind: args.end_ind]):
        print(f"seq: {sqn}")

        pkl_pth = pathlib.Path(osp.join(RES_DIR, sqn, 'eval_rel_pose_sratra_latest.pkl'))
        if not pkl_pth.exists():
            missing_pkl_seqs.append(sqn)
            print(f'eval pkl not avaialable for seq: {missing_pkl_seqs}')
            continue
        else:
            data = load_pkl(pkl_pth)
            new_dict = {
                'sqn' : data['sqn'],
                'RRE_mean' : data['RRE_valid_pairs_mean'],
                'RTE_mean' : data['RTE_valid_pairs_mean'],
                'DET_rate' : data['DET_rate'],
                'RRE_5.0'  : data['RRE@5.0'],
                'RRE_10.0' : data['RRE@10.0'],
                'RRE_20.0' : data['RRE@20.0'],
                'RRE_30.0' : data['RRE@30.0'],
                'RTE_0.05' : data['RTE@0.05'],
                'RTE_0.1' : data['RTE@0.1'],
                'RTE_0.2' : data['RTE@0.2'],
                'RTE_0.3' : data['RTE@0.3'],
                'RRE_5.0_RTE_0.05' : data['RRE@5.0_RTE@0.05'],
                'RRE_10.0_RTE_0.1' : data['RRE@10.0_RTE@0.1'],
                'RRE_15.0_RTE_0.15' : data['RRE@15.0_RTE@0.15'],
                'RRE_20.0_RTE_0.2' : data['RRE@20.0_RTE@0.2'],
                'RRE_25.0_RTE_0.25' : data['RRE@25.0_RTE@0.25'],
                'RRE_30.0_RTE_0.3' : data['RRE@30.0_RTE@0.3']
            }

            df = pd.DataFrame.from_dict([new_dict])
            all_seqs_dict.append(df)
    bb()
    df_comb = pd.concat(all_seqs_dict)

    df_comb_save_pth = osp.join(RES_DIR, 'all_valid_seqs_sra_tra_rel_pose_all_framepairs_latest.csv')
    df_comb.to_csv(df_comb_save_pth)
    print(f"Saved here: {df_comb_save_pth}")
    print('Done!!')


