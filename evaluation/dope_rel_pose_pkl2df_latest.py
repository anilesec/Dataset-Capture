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
    for sqn in tqdm(all_sqns[args.start_ind: args.end_ind]):
        print(f"seq: {sqn}")

        pkl_pth = osp.join(RES_DIR, sqn, 'eval_rel_pose_dope_latest.pkl')
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

        # ths = [[4.0, 0.02], [10.0, 0.05], [20.0, 0.10]]
        # bins = []
        # for th in ths:
        #     # bb()
        #     bin_vals = np.logical_and(data['RRE'] <= th[0], data['RTE'][:-1] <= th[1])
        #     bins.append(float(f"{np.average(bin_vals)*100:.2f}"))
        # new_dict = {
        #     'sqn' : data['sqn'],
        #     f"rre_{ths[0][0]}_rte{ths[0][1]}" : bins[0], 
        #     f"rre_{ths[1][0]}_rte{ths[1][1]}" : bins[1],
        #     f"rre_{ths[2][0]}_rte{ths[2][1]}" : bins[2]  
        # }

        # new_dict = {
        #     'sqn' : data['sqn'],
        #     'repr_mean' : f"{data['REPE_mean']:.4f}",
        #     'rre_mean' : f"{np.mean(data['RRE']):.4f}",
        #     'rre_med' : f"{np.median(data['RRE']):.4f}",
        #     'rre_max' : f"{np.max(data['RRE']):.4f}",
        #     'rre_min' : f"{np.min(data['RRE']):.4f}",
        #     'rre_std' : f"{np.std(data['RRE']):.4f}",
        #     'rre_25th' : f"{np.percentile(data['RRE'], [25, 50, 75, 90])[0]:.4f}",
        #     'rre_50th' : f"{np.percentile(data['RRE'], [25, 50, 75, 90])[1]:.4f}",
        #     'rre_75th' : f"{np.percentile(data['RRE'], [25, 50, 75, 90])[2]:.4f}",
        #     'rre_95th' : f"{np.percentile(data['RRE'], [25, 50, 75, 90])[3]:.4f}",
        #     'rte_mean' : f"{np.mean(data['RTE']):.4f}",
        #     'rte_med' : f"{np.median(data['RTE']):.4f}",
        #     'rte_max' : f"{np.max(data['RTE']):.4f}",
        #     'rte_min' : f"{np.min(data['RTE']):.4f}",
        #     'rte_std' : f"{np.std(data['RTE']):.4f}",
        #     'rte_25th' : f"{np.percentile(data['RTE'], [25, 50, 75, 90])[0]:.4f}",
        #     'rte_50th' : f"{np.percentile(data['RTE'], [25, 50, 75, 90])[1]:.4f}",
        #     'rte_75th' : f"{np.percentile(data['RTE'], [25, 50, 75, 90])[2]:.4f}",
        #     'rte_95th' : f"{np.percentile(data['RTE'], [25, 50, 75, 90])[3]:.4f}",
        #     'miss_frms_no' : f"{data['missing_det_info'][1]:.4f}",
        #     'miss_frms_ratio' : f"{data['missing_det_info'][2]:.4f}"    
        # }

        df = pd.DataFrame.from_dict([new_dict])
        all_seqs_dict.append(df)

    df_comb = pd.concat(all_seqs_dict)

    df_comb_save_pth = osp.join(RES_DIR, 'all_seqs_dope_rel_pose_eval_perframe_latest.csv')
    df_comb.to_csv(df_comb_save_pth)
    print(f"Saved here: {df_comb_save_pth}")
    print('Done!!')

    

        








        





        