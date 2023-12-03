import os
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip, clips_array
from ipdb import set_trace as bb
    

if __name__ == "__main__":
    gt_tts_dir = "/scratch/2/user/aswamy/temp/turntables/PC"   
    fdr_tts_dir = "/scratch/2/user/aswamy/temp/turntables/colmap_FDR_reorient2gt/meshes"
    hhor_tts_dir = "/scratch/2/user/aswamy/temp/turntables/colmap_hhor_reorient2gt/meshes"
    vh_tts_dir = "/scratch/2/user/aswamy/temp/turntables/colmap_VH_reorient2gt/meshes"
    save_dir = "/scratch/2/user/aswamy/temp/turntables/comb_gt-vh_reorient2gt-fdr_reorient2gt-hhor_reorient2gt"

    # all_sqns:
    all_sqns = ['20220902110304', '20220824142508', '20220829154032', '20220907155615', '20220812180133', '20220905153946', '20220912142017', '20220913151554', '20220902170443', '20220823115809', '20220913145135', '20220805164755', '20220824102636', '20220830161218', '20220902104048', '20220912155637', '20220909134639', '20220912165455', '20220905141444', '20220705173214', '20220812172414', '20220902163904', '20220809161015', '20220907152036', '20220912152000', '20220909114359', '20220909113237', '20220909142411', '20220912161700', '20220913154643', '20220912164407', '20220824154342', '20220811171507', '20220912161552', '20220824180652', '20220902151726', '20220811163525', '20220830162330', '20220819164041', '20220907153810', '20220913153520', '20220824181949', '20220809170847', '20220805165947', '20220905111237', '20220819162529', '20220823113402', '20220824150228', '20220909141430', '20220809171854', '20220830163143', '20220905112623', '20220913144436', '20220824144438', '20220811165540', '20220909121541', '20220909120614', '20220824152850', '20220811170459', '20220902153221', '20220902114024', '20220902111409', '20220905140306', '20220824104203', '20220909151546', '20220905105332', '20220811172657', '20220909111450', '20220912144751', '20220824105341', '20220819155412', '20220909152911', '20220909145039', '20220824155144', '20220905155551', '20220812170512', '20220909140016', '20220909115705', '20220905154829', '20220905151029', '20220829155218', '20220902164854', '20220912160620', '20220902154737', '20220902111535', '20220902115034', '20220809163444', '20220902163950', '20220811154947', '20220912143756', '20220823114538', '20220905112733', '20220812174356', '20220912151849', '20220824160141', '20220905142354']
    missing_gts = []
    missing_vhs = []
    missing_fdrs = []
    missing_hhors = []
    for sqn in tqdm(all_sqns):
        print('sqn:', sqn)

        # if not sqn=='20220905141444':
        #     continue

        gt_vpth = os.path.join(gt_tts_dir, sqn, 'nolight/vid.mp4')
        if os.path.exists(gt_vpth):
            v_gt = VideoFileClip(gt_vpth)
        else:
            missing_gts.append(sqn)
            v_gt = None

        vh_vpth = os.path.join(vh_tts_dir, f"{sqn}/nolight/vid.mp4")
        if os.path.exists(vh_vpth):
            v_vh = VideoFileClip(vh_vpth)
        else:
            missing_vhs.append(sqn)
            v_vh = None

        fdr_vpth = os.path.join(fdr_tts_dir, f"{sqn}/nolight/vid.mp4")
        if os.path.exists(fdr_vpth):
            v_fdr = VideoFileClip(fdr_vpth)
        else:
            missing_fdrs.append(sqn)
            v_fdr = None
        
        hhor_vpth = os.path.join(hhor_tts_dir, f"{sqn}/nolight/vid.mp4")
        if os.path.exists(hhor_vpth):
            v_hhor = VideoFileClip(hhor_vpth)
        else:
            missing_hhors.append(sqn)
            v_hhor = None

        if None in [v_gt, v_vh, v_fdr, v_hhor]:
            continue
        comb = clips_array([[v_gt, v_vh, v_fdr, v_hhor]])
    
        os.makedirs(save_dir, exist_ok=True)
        comb.write_videofile(os.path.join(save_dir, f"{sqn}.mp4"))

    print('missing_gts', missing_gts, len(missing_gts))
    print('missing_vhs', missing_vhs, len(missing_vhs))
    print('missing_fdrs', missing_fdrs, len(missing_fdrs))
    print('missing_hhors', missing_hhors, len(missing_hhors))
    print('Done')



