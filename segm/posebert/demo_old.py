from copyreg import pickle
import os
import argparse
from PIL import Image
import torch
from posebert.dope import LCRNet_PPI_improved, dope_resnet50, assign_hands_and_head_to_body
import ipdb
import numpy as np
import smplx
import roma
from model import PoseBERT
from skeleton import (convert_jts, visu_pose3d, visu_pose2d, update_mano_joints, get_mano_traversal,
                      normalize_skeleton_by_bone_length, preprocess_skeleton, inverse_projection_to_3d
)
from renderer import PyTorch3DRenderer
from pytorch3d.renderer import look_at_view_transform
from tqdm import tqdm
try:
    import _pickle as pickle
except:
    import pickle

@torch.no_grad()
def main(video,
         intrinsics,
         render,
         output_jts_fn,
         t_max=-1,
         dope_dir='/tmp-network/user/fbaradel/projects/PoseBERT/models_release',
        #  posebert_ckpt='/tmp-network/user/fbaradel/projects/PoseBERT/journal/logs/posebert_jts2mano/dope_M76_mix_randomMaskBlock50_loc_j2dFrame/checkpoints/last.pt',
         posebert_ckpt='/tmp-network/user/fbaradel/projects/PoseBERT/journal/logs/posebert_jts2mano/dope_M76_mix_randomMaskBlock50_loc_j2dFrame_bis/checkpoints/last.pt',
         requested_img_size=640, 
         requested_focal_length=615,
        #  requested_focal_length=530,
        #  requested_img_size=960, requested_focal_length=530
        img_type='vincent',
        output_fn='./out.pkl',
        precomputed_dope=None
         ):
    assert os.path.isdir(video)
    
    tmp_dir = "/tmp/roar/"
    os.makedirs(tmp_dir, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print("\nLoading models...")
    # Load DOPE model
    ckpt_dope = torch.load(os.path.join(dope_dir, 'DOPErealtime_v1_0_0.pth.tgz'), map_location=device)
    ckpt_dope['half'] = True
    dope = dope_resnet50(**ckpt_dope['dope_kwargs'])
    dope.eval()
    dope.load_state_dict(ckpt_dope['state_dict'])
    dope = dope.to(device)
    
    # Load MANO
    SMPLX_DIR = '/tmp-network/SlowHA/user/fbaradel/data/SMPLX'
    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True).to(device)
    faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32))
    out = bm()
    mano_mean = update_mano_joints(out.joints.cpu(), out.vertices.cpu()).numpy()[0]
    traversal, parents = get_mano_traversal()
    
    # Load PoseBERT
    checkpoint = torch.load(posebert_ckpt)
    posebert = PoseBERT().to(device)
    posebert.eval()
    posebert.load_state_dict(checkpoint['model_state_dict'])
    # posebert_seq_len = 76
    posebert_seq_len = 76
    
    # Retrieve images
    if img_type == 'anil':
        img_fns = [os.path.join(video, x) for x in os.listdir(video)]
        img_fns.sort()
    elif img_type == 'vincent':
        ldirs = os.listdir(video)
        ldirs.sort()
        img_fns = [os.path.join(video, x, 'view_0.png') for x in ldirs]
    
    # Image size
    img = Image.open(img_fns[0]).convert('RGB')
    init_width, init_height = img.size
    if requested_img_size is not None:
        if init_width > init_height:
            basewidth = requested_img_size
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        else:
            baseheight = requested_img_size
            wpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[0])*float(wpercent)))
            img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    width, height = img.size
    print(f"Initial size={init_width}x{init_height} - New size={width}x{height}")    
    
    # Camera intrinsics
    delta = np.abs(height - width)
    img_size = max([width, height])
    if len(intrinsics) > 0 and os.path.isfile(intrinsics):
        K = torch.from_numpy(np.load(intrinsics, mmap_mode='r')['kinect2_right']).reshape(1, 3, 3)
        if width > height:
            K[0,0,-1] += delta / 2.
        else:
            K[0,1,-1] += delta / 2.
    else:
        f_x, f_y = requested_focal_length, requested_focal_length
        K = torch.zeros([1, 3, 3])
        K[:, 0, 0] = f_x
        K[:, 1, 1] = f_y
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = img_size/2. #camera_center
    print(f"Camera Intrinsic:\n{K[0].numpy()}")
    K_inverse = torch.inverse(K).float()
    
    # Rendering
    alpha=0.8
    renderer = PyTorch3DRenderer(img_size).to(device)
    dist, elev, azim = 0.0001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    
    # Run DOPE on each frame
    print("\nRunning DOPE on each frame")
    list_x = []
    list_mask = []
    list_j2d_dope, list_j3d_dope = [], []
    for t, img_fn in enumerate(tqdm(img_fns)):
        # Load image
        img = Image.open(img_fn).convert('RGB')
        if requested_img_size is not None:
            img = img.resize((width,height), Image.ANTIALIAS)
        
        # Run DOPE
        imlist = [torch.from_numpy(np.asarray(img) / 255.).permute(2, 0, 1).to(device).float()]
        resolution = imlist[0].size()[-2:]

        # preconputed dope
        if args.precomputed_dope is not None:
            fn = os.path.join(args.precomputed_dope, img_fn.split('/')[-1].split('.')[0]+'.pkl')
            with open(fn, 'rb') as f:
                dets = pickle.load(f)
            
            for j in range(len(dets['detections'])):
                j2d_up = (dets['detections'][j]['pose2d'] / np.asarray([[init_width, init_height]])) * np.asarray([[width, height]])
                dets['detections'][j]['pose2d'] = j2d_up.astype(np.float32)

            detections = {'hand': dets['detections']}
        else:
            results = dope(imlist, None)[0]
            parts = ['body', 'hand', 'face']
            res = {k: v.float().data.cpu().numpy() for k, v in results.items()}
            detections = {}
            for part in parts:
                    detections[part] = LCRNet_PPI_improved(res[part + '_scores'], res['boxes'], res[part + '_pose2d'],
                                                        res[part + '_pose3d'], resolution,
                                                        **ckpt_dope[part + '_ppi_kwargs'])
            detections, body_with_wrists, body_with_head = assign_hands_and_head_to_body(detections)
        
        x = torch.zeros(1, 108).float()
        j2d = np.zeros((21, 2)).astype(np.float32)
        mask = 0.
        if 'hand' in detections and len(detections['hand']) > 0:
            hand_isright = [z['hand_isright'] for z in detections['hand']]
            if sum(hand_isright) > 0:
                idx = hand_isright.index(True)
                hand = detections['hand'][idx]
                j2d = convert_jts(hand['pose2d'].reshape(1, 21, -1), 'dope_hand', 'mano')[0]
                j3d = convert_jts(hand['pose3d'].reshape(1, 21, -1), 'dope_hand', 'mano')[0]
                # print(j2d)
                
                # Post-processs 3D
                j3d_ = j3d.copy() - j3d[[0]]
                j3d_ = normalize_skeleton_by_bone_length(j3d_, mano_mean, traversal, parents).reshape(1, 21, 3)
                rel3d, _, root3d = preprocess_skeleton(j3d_,
                                                center_joint=[0],  # wrist
                                                xaxis=[1, 10],  # middle1-wrist
                                                yaxis=[4, 0],  # index1-right1
                                                iter=3, norm_x_axis=True, norm_y_axis=True,
                                                sanity_check=False)
                rel3d_ = torch.from_numpy(rel3d)[:, 1:].flatten(1)
                root3d_ = torch.from_numpy(root3d)[..., :2].flatten(1)
                # root3d = roma.rotmat_to_rotvec(torch.from_numpy(root3d))
                x3d = torch.cat([root3d_, rel3d_], 1)
                
                # Post-process 2D
                x2d = torch.from_numpy(j2d.copy())
                if height > width:
                    x2d[:, 0] += delta/2.
                else:
                    x2d[:, 1] += delta/2.
                x2d = x2d.reshape(1, -1)
                x2d /= img_size
            
                x = torch.cat([x3d, x2d], -1)
                mask = 1.0

                if t in [24]:
                    x = x * .0
                    j2d = j2d * 0.
                    mask = 0.0
                # compute distance with previous j2d
                # @fbaradel posebert should solve that!
                # if t > 0:
                #     ids = [0]
                #     dist2d = np.sqrt(((j2d[ids] - list_j2d_dope[-1][ids])**2).sum(-1).sum(-1))
                #     if dist2d > 50:
                #         mask = 0.
                #         print(dist2d, mask)
            else:
                j2d = list_j2d_dope[-1].copy()
                x = list_x[-1].clone()
                print("not sure it is great....")
                mask = 1.0

                # x = x * .0
                # j2d = j2d * 0.
                # mask = 0.0
            
        list_x.append(x)
        list_mask.append(mask)
        list_j2d_dope.append(j2d)
        
        if t_max == t:
            print(f"Running the method for the {t_max} first frames only..")
            break

    # Run PoseBERT in an offline manner
    print("\nRunning PoseBERT offline (and do the rendering as well)")
    list_j3d_posebert = []
    list_global_orient, list_transl = [], []
    x = torch.cat(list_x).float()
    mask = torch.from_numpy(np.stack(list_mask)).bool()
    for t in tqdm(range(len(list_x))):        
        
        if True:
            # select appropriate subseq
            if t - posebert_seq_len // 2 < 0:
                start_ = max([0, t - posebert_seq_len // 2])
                end_ = start_ + posebert_seq_len
            else:
                end_ = min([len(list_x), t + posebert_seq_len // 2])
                start_ = end_ - posebert_seq_len
            tt = np.clip(np.arange(start_, end_), 0, len(list_x) - 1).tolist()
            t_of_interest = tt.index(t)

            # PoseBERT forward
            x_ = torch.cat([list_x[t_] for t_ in tt])
            mask_ = torch.Tensor([list_mask[t_] for t_ in tt]).bool()
            y_hat, loc_hat = posebert(x_.unsqueeze(0).to(device), mask_.unsqueeze(0).to(device))
            # y_hat, loc_hat = posebert(x_.unsqueeze(0).to(device))
            y_hat = roma.rotmat_to_rotvec(y_hat)
            transl = inverse_projection_to_3d(loc_hat, img_size, K_inverse.to(device))
            global_orient = y_hat[:, t_of_interest, 0]
            hand_pose = y_hat[:, t_of_interest, 1:].flatten(1)
            transl = transl[:, t_of_interest]
        else:
            y_hat, loc_hat = posebert(x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
            y_hat, loc_hat = y_hat[:,t], loc_hat[:,t]
            y_hat = roma.rotmat_to_rotvec(y_hat)
            global_orient, hand_pose = y_hat[:, 0], y_hat[:, 1:].flatten(1)
            transl = inverse_projection_to_3d(loc_hat.unsqueeze(0), img_size, K_inverse.to(device))[0]
        
        # MANO
        out = bm(global_orient=global_orient, hand_pose=hand_pose, transl=transl)
        verts = out.vertices
        jts = update_mano_joints(out.joints, verts)
        list_j3d_posebert.append(jts.cpu())
        list_global_orient.append(global_orient.cpu())
        list_transl.append(transl.cpu())
        
        # Rendering
        if render:
            img_posebert = renderer.renderPerspective(vertices=[verts[0].to(device)], 
                                                faces=[faces.to(device)],
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(device)],
                                                rotation=rotation.to(device),
                                                camera_translation=cam.to(device),
                                                # focal_length=2 * f_x / img_size
                                                focal_length = torch.Tensor([[2 * K[0,0,0] / img_size, 
                                                                            2* K[0,1,1] / img_size
                                                                            ]]).to(device),
                                                ).cpu().numpy()[0]
            img_rgb = Image.open(img_fns[t]).convert('RGB')
            if requested_img_size is not None:
                img_rgb = img_rgb.resize((width,height), Image.ANTIALIAS)
            img_rgb = np.asarray(img_rgb) # RGB not RGBA

            # Debug
            # _j2d = x_[t_of_interest,-42:].reshape(-1, 2).cpu().numpy()
            # img_posebert = visu_pose2d(img_posebert, _j2d* img_size)
            # from PIL import ImageDraw
            # img_ = Image.fromarray(img_posebert.copy())
            # draw = ImageDraw.Draw(img_)
            # lw_dot = 10
            # p0 = loc_hat[0, t_of_interest,:2].cpu().numpy() * img_size
            # p0_ = (p0[0] - lw_dot, p0[1] - lw_dot, p0[0] + lw_dot, p0[1] + lw_dot)
            # draw.ellipse(p0_, fill='red', outline='red')
            # img_posebert = np.asarray(img_)
            #

            if height > width:
                img_posebert = img_posebert[:, delta//2:width+(delta//2)]
            else:
                img_posebert = img_posebert[delta//2:height+(delta//2)]
            
            fg_mask = (np.sum(img_posebert, axis=-1) != 0)
            fg_mask = np.concatenate((fg_mask[:,:,None], fg_mask[:,:,None], fg_mask[:,:,None]), axis=2)
            img_posebert_rgb = (fg_mask * (alpha * img_posebert + (1.0-alpha) * img_rgb) + (1-fg_mask) * img_rgb).astype(np.uint8)
            
            # # Visu DOPE 2D/3D, estimated bone-lenths normalized relative 3D pose and PoseBERT outputs
            img_dope2d = visu_pose2d(np.asarray(img_rgb), list_j2d_dope[t])
            # img_dope3d = visu_pose3d(j3d.reshape(1, 21, 3), res=img_size)
            # img_dopeRel3d = visu_pose3d(rel3d.reshape(1, 21, 3), res=img_size)
            # img_dope2d_normed = visu_pose2d(np.zeros((img_size, img_size, 3)).astype(np.uint8), img_size * x2d.reshape(21, 2).numpy())
            
            out = np.concatenate([img_rgb, img_dope2d, img_posebert_rgb], 1)
            import cv2
            out = cv2.putText(out, f"t={t:05d}", (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            Image.fromarray(out).save(os.path.join(tmp_dir, f"img_{t:05d}.jpg"))
            
            # Image.fromarray(out).save(f"img.jpg")
            # ipdb.set_trace()

            # if not mask_[t_of_interest].item():
                # Image.fromarray(out).save(f"img.jpg")
                # ipdb.set_trace()

        
    # Create video
    if render:
        print("\nCreating the output video")
        fn = "video.mp4"
        cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {fn} -y"
        os.system(cmd)
        os.system(f"rm {tmp_dir}/*.jpg")
        
    # Save the MANO joints
    j3d_posebert = torch.cat(list_j3d_posebert).numpy()
    global_orient_posebert = torch.cat(list_global_orient).numpy()
    transl_posebert = torch.cat(list_transl).numpy()
    print(f"\nSaving MANO joints of shape {j3d_posebert.shape} into: {output_jts_fn}")
    os.makedirs(os.path.dirname(output_jts_fn), exist_ok=True)
    np.save(output_jts_fn, j3d_posebert)

    # Save them all
    print(f"\nSaving joints, global_orient and transl into: {output_fn}")
    out = {'j3d': j3d_posebert, 'global_orient': global_orient_posebert, 'transl': transl_posebert}
    # ipdb.set_trace()
    with open(output_fn, 'wb') as f:
        pickle.dump(out, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Mesh Recovery from a RGB video')
    parser.add_argument('--video', type=str, default='/tmp-network/user/aswamy/cp_for_posebert/full1_use/kinect2_right/mug', help='path to the video')
    parser.add_argument('--intrinsics', type=str, default='/tmp-network/user/aswamy/cp_for_posebert/full1_use/kinect2_right/cam_intr.npz', help='path to the camera instrinsic parameters file')
    parser.add_argument('--render', type=int, default=0, choices=[0,1], help='render the mesh or not')
    parser.add_argument('--output_jts_fn', type=str, default='./joints.npy', help='path to save the joints extracted by DOPE + PoseBERT')
    parser.add_argument('--output_fn', type=str, default='./posebert.pkl', help='path to save the joints, transl amd global_orient extracted by DOPE + PoseBERT')
    parser.add_argument('--posebert_ckpt', type=str, 
    # default='/tmp-network/user/fbaradel/projects/PoseBERT/journal/logs/posebert_jts2mano/dope_M76_mix_randomMaskBlock50_loc_j2dFrame/checkpoints/last.pt', 
    default='/tmp-network/user/fbaradel/projects/PoseBERT/journal/logs/posebert_jts2mano/dope_M76_mix_randomMaskBlock50_loc_j2dFrame_decomp/checkpoints/last.pt',
    help='checkpoint weights of posebert')
    parser.add_argument('--t_max', type=int, default=-1, help='for debugging')
    parser.add_argument('--img_type', type=str, default='anil', help='for debugging')
    parser.add_argument('--precomputed_dope', type=str, default=None)
    args = parser.parse_args()
    main(args.video, args.intrinsics, args.render == 1, args.output_jts_fn, args.t_max, img_type=args.img_type, output_fn=args.output_fn, precomputed_dope=args.precomputed_dope)
