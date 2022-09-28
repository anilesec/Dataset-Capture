# mainly vleroy's idea 
import numpy as np
from ipdb import set_trace as bb
from PIL import Image
from matplotlib import pyplot as pl
import cv2, glob, os
osp = os.path
from tqdm import tqdm
from skimage.measure import label   
from scipy.ndimage.morphology import binary_dilation, binary_erosion

def getLargestCC(segmentation):
    "get largest connected components"
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largestCC

def  segm_sleeve(rgbs_dir, fgnd_masks_dir, save_rt_dir, slv_clr, start_ind, end_ind, norm_type,
                kmeans_max_iter=100, kmeans_eps=0.2, kmeans_num_clustrs=10):
    rgbs_pths = sorted(glob.glob(osp.join(rgbs_dir, '*.png')))
    # bb()
    for idx, imp in tqdm(enumerate(rgbs_pths[start_ind: end_ind])):
        # if idx <= 123:
        #     continue
        # bb()
        fgnd_mskp = osp.join(fgnd_masks_dir, osp.basename(imp))
        img = np.array(Image.open(imp))
        fgnd_msk = np.array(Image.open(fgnd_mskp))[:, :, 0]
        im = cv2.bitwise_and(img, img, mask=fgnd_msk)

        # normalize image
        if norm_type == 'L2':
            im_norm = 255. * im / (np.linalg.norm(im, axis=-1)[:, :, None] + 1e-6)
            iTHRESH = 75  # normalized intensity thresh (this should be adjusted for each seq/when ratio is used instead of L2 normalization)
        else:
            im_norm = 255. * im / (np.sum(im,axis=-1)[:, :, None] + 1e-6)
            iTHRESH = 110  # normalized intensity thresh (this should be adjusted for each seq/when ration is used instead of L2 normalization)

        #k-means on normalized image space
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_max_iter, kmeans_eps)
        _, labels, (centers) = cv2.kmeans(im_norm.reshape([-1, 3]).astype(np.float32),
                                          kmeans_num_clustrs, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        # Compute segmented image
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(im.shape)

        counts = []
        for ind in range(kmeans_num_clustrs):
            counts.append(np.count_nonzero(labels==ind))

        # slv_clr_ind = np.argsort(counts)[-2] #2nd maximum count pixel color/cluster color
        # slv_clr_ind = np.argsort(centers[:, 1])[-2] # 2nd max green clr value
        slv_clr_ind = np.argmax(centers[:, 1])
        slv_color_auto = centers[slv_clr_ind]
        # cv2.imwrite('./segmented_image.png', segmented_image); slv_i = np.linalg.norm(segmented_image - np.array(slv_color_auto)[None, None], axis=-1).astype(np.uint8); cv2.imwrite('./slv_i.png', slv_i)
        # cv2.imwrite('./segmented_image.png', segmented_image); slv_i = np.linalg.norm(segmented_image - np.array(slv_clr)[None, None], axis=-1).astype(np.uint8); cv2.imwrite('./slv_i.png', slv_i)
        # bb()        
        
        # slv_i = np.linalg.norm(segmented_image - np.array(slv_clr)[None, None], axis=-1).astype(np.uint8)
        # cv2.imwrite('./slv_i.png', slv_i)

        # slv = np.linalg.norm(segmented_image - np.array(slv_clr)[None, None], axis=-1) < iTHRESH
        slv = np.linalg.norm(segmented_image - np.array(slv_color_auto)[None, None], axis=-1) < iTHRESH
        # slv1 = np.linalg.norm(segmented_image - np.array(slv_clr)[None, None], axis=-1) < 0.45*255
        # slv2 = np.linalg.norm(segmented_image - np.array(slv_clr)[None, None], axis=-1) > 0.35*255
        # slv = np.logical_and(slv1, slv2)
        
        # Remove small components after small hole filling
        # slv = binary_erosion(binary_dilation((binary_dilation(binary_dilation(slv)))))
        slv = binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation((binary_dilation(binary_dilation(slv)))))))))
        # slv = binary_dilation(binary_dilation(binary_dilation((binary_dilation(binary_dilation(slv))))))
        # slv = (binary_dilation(binary_dilation(binary_dilation(slv))))
        slvLCC = getLargestCC(slv)

        # fn_slv_mask = os.path.join(save_rt_dir, f'slv_msk_{norm_type}_kmeans{kmeans_max_iter}mxiter_lcc', osp.basename(imp))
        fn_slv_mask = osp.join(save_rt_dir, 'slv_msk', osp.basename(imp))
        if not os.path.exists(os.path.dirname(fn_slv_mask)):
            os.makedirs(os.path.dirname(fn_slv_mask))
        slv_msk = slvLCC.astype(np.uint8) * 255
        # bb()
        Image.fromarray(slv_msk).save(fn_slv_mask)

        # fn_mask_wo_slv = os.path.join(args.save_rt_dir, 'slv_mskinv_kmeans_lcc_kmeans10iter', osp.basename(imp))
        fn_mask_wo_slv = osp.join(save_rt_dir, 'slvless_msk', osp.basename(imp))
        if not os.path.exists(os.path.dirname(fn_mask_wo_slv)):
            os.makedirs(os.path.dirname(fn_mask_wo_slv))
        slvless_msk = np.logical_not(slvLCC).astype(np.uint8) * 255
        Image.fromarray(slvless_msk).save(fn_mask_wo_slv)

        # fn_im_woslv = os.path.join(args.save_rt_dir, f'img_woslv_{args.norm_type}_kmeans{kmean_num_clustrs}mxiter_lcc', osp.basename(imp))
        fn_im_woslv = osp.join(save_rt_dir, 'slvless_img', osp.basename(imp))
        if not os.path.exists(os.path.dirname(fn_im_woslv)):
            os.makedirs(os.path.dirname(fn_im_woslv))
        masked = cv2.bitwise_and(im[:, :, ::-1], im[:, :, ::-1], mask=slvless_msk)
        Image.fromarray(masked[:, :, ::-1]).save(fn_im_woslv)

    return osp.dirname(fn_im_woslv)

def sho(data):
    pl.figure()
    pl.imshow(data)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Processing captured raw data")

    parser.add_argument('--rgbs_dir', type=str, required=True,
                        help='path to the rgbs')
    parser.add_argument('--fgnd_masks_dir', type=str, required=True,
                        help='path to the fgnd masks')
    parser.add_argument('--save_rt_dir', type=str, required=True,
                        help='root path to save, should be seq path ideally')
    # parser.add_argument('--save_rgb', action='store_true',
    #                     help='flag to save rgb image')    
    # parser.add_argument('--save_depth', action='store_true',
    #                     help='flag to save depth')    
    # parser.add_argument('--save_pcd', action='store_true',
    #                     help='flag to save only xyz')
    # parser.add_argument('--save_mask', action='store_true',
    #                     help='flag to save foreground mask, need save_pcd and seg_pcd flags to be set')                        
    # parser.add_argument('--pcd_normals', action='store_true',
    #                     help='flag to compute pcd normals')  
    # parser.add_argument('--pcd_color', action='store_true',
    #                     help='flag to save pcd color')                        
    # parser.add_argument('--pcd_bkgd_rm', action='store_true',
    #                     help='flag to segment foreground poincloud, needs save_pcd flag to be set')  
    # parser.add_argument('--depth_thresh', type=float, required=True,
    #                     help='depth/dist threshold in (meters) to segment the foreground(hand+obj)')
    parser.add_argument('--start_ind', type=int, default=None,
                        help='start index of the frame')
    parser.add_argument('--end_ind', type=int, default=None,
                        help='end index of the frame')
    parser.add_argument('--norm_type', type=str, default='L2',
                        help='norm coice; "L2" or "ratio"')
    parser.add_argument('--kms_num_clstrs', type=int, default=4,
                        help='k-means number of clusters')
    # parser.add_argument('--debug', type=int, default=0,
    #                     help='set to 1 for breakpoint') 
    args = parser.parse_args()
    print("args:", args)  
    
    # parameters
    SLV_COLOR =  [144, 195, 69] #[159, 185, 70] # normalized color of arm sleeve
    KMEANS_MAX_ITER = 100  # k-means max iteration criteria (increases computation time)
    KMEANS_EPSILON = 0.2 # k-means accuracy criteria
    # KMEANS_NUM_CLUSTERS = 3 # k-means number of clusters
    segm_sleeve(rgbs_dir=args.rgbs_dir, fgnd_masks_dir=args.fgnd_masks_dir, save_rt_dir=args.save_rt_dir, slv_clr=SLV_COLOR, start_ind=args.start_ind,
                end_ind=args.end_ind, norm_type=args.norm_type, kmeans_max_iter=KMEANS_MAX_ITER,
                kmeans_eps=KMEANS_EPSILON, kmeans_num_clustrs=args.kms_num_clstrs)
    print('Done!')

    """
    if args.start_ind is None:
        args.start_ind = 0
        
    if args.end_ind is None:
        args.end_ind = len(rgbs_pths)

    for idx, imp in tqdm(enumerate(rgbs_pths[args.start_ind: args.end_ind])):
        im = np.array(Image.open(imp))

        # normalize image
        if args.norm_type == 'L2':
            im_norm = 255. * im / (np.linalg.norm(im, axis=-1)[:,:,None] + 1e-6)
            iTHRESH = 50  # normalized intensity thresh (this should be adjusted for each seq/when ration is used instead of L2 normalization)
        else:
            im_norm = 255. * im / (np.sum(im,axis=-1)[:,:,None] + 1e-6)
            iTHRESH = 110  # normalized intensity thresh (this should be adjusted for each seq/when ration is used instead of L2 normalization)

        #k-means on normalized image space
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_MAX_ITER, KMEANS_EPSILON)
        _, labels, (centers) = cv2.kmeans(im_norm.reshape([-1, 3]).astype(np.float32), KMEANS_NUM_CLUSTERS, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # Compute segmented image
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(im.shape)

        arm = np.linalg.norm(segmented_image - np.array(ARM_COLOR)[None, None], axis=-1) < iTHRESH

        # Remove small components after small hole filling
        arm = binary_erosion(binary_erosion(binary_dilation(binary_dilation(binary_dilation(arm)))))
        armLCC = getLargestCC(arm)

        fn_slv_mask = os.path.join(args.save_rt_dir, f'slv_msk_{args.norm_type}_kmeans{KMEANS_MAX_ITER}mxiter_lcc', osp.basename(imp))
        if not os.path.exists(os.path.dirname(fn_slv_mask)):
            os.makedirs(os.path.dirname(fn_slv_mask))
        slv_msk = armLCC.astype(np.uint8) * 255
        Image.fromarray(slv_msk).save(fn_slv_mask)

        # fn_mask_wo_slv = os.path.join(args.save_rt_dir, 'slv_mskinv_kmeans_lcc_kmeans10iter', osp.basename(imp))
        # if not os.path.exists(os.path.dirname(fn_mask_wo_slv)):
        #     os.makedirs(os.path.dirname(fn_mask_wo_slv))
        slvless_msk = np.logical_not(armLCC).astype(np.uint8) * 255
        # Image.fromarray(slvless_msk).save(fn_mask_wo_slv)

        fn_im_woslv = os.path.join(args.save_rt_dir, f'img_woslv_{args.norm_type}_kmeans{KMEANS_MAX_ITER}mxiter_lcc', osp.basename(imp))
        if not os.path.exists(os.path.dirname(fn_im_woslv)):
            os.makedirs(os.path.dirname(fn_im_woslv))
        masked = cv2.bitwise_and(im[:, :, ::-1], im[:, :, ::-1], mask=slvless_msk)
        Image.fromarray(masked[:, :, ::-1]).save(fn_im_woslv)

    print('Done!')
    """
 
