from posebert.dataset_old import ContactPoseDataset, J3dDataset, worker_init_fn
import ipdb
import argparse
import torch
import os
import smplx
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from posebert.model import PoseBERT, NUM_JOINTS_HAND
from torch.utils.data import DataLoader
from posebert.utils import get_last_checkpoint, AverageMeter
import sys
from tqdm import tqdm
from posebert.skeleton import inverse_projection_to_3d, perspective_projection, update_mano_joints
import roma
from posebert.renderer import PyTorch3DRenderer
from pytorch3d.renderer import look_at_view_transform
from PIL import Image
from posebert.skeleton import visu_pose2d, visu_pose3d
from posebert.interpenetration import DistanceFieldPenetrationLoss, get_collision_idxs, get_collision_idx_batch

SMPLX_DIR = '/tmp-network/SlowHA/user/fbaradel/data/SMPLX'

class Trainer():
    def __init__(self, *,
                 model, optimizer, device, args,
                 epoch, start_iter, seq_len=16,
                 best_val=None):
        super(Trainer, self).__init__()
        self.best_val = 1e5 if best_val is None else best_val
        self.args = args
        self.device = device
        self.optimizer = optimizer
        self.seq_len = seq_len

        self.bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True).to(device)
        self.betas = torch.zeros(1, 10).to(self.device)
        for param in self.bm.parameters():
            param.requires_grad = False

        self.faces = torch.from_numpy(np.array(self.bm.faces, dtype=np.int32)).to(device)
        self.init_logdir()
        self.current_iter = start_iter
        self.current_epoch = epoch
        self.model = model

        self.renderer = PyTorch3DRenderer(960).to(device)
        dist, elev, azim = 1e-5, 0., 180
        rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        self.rotation = rotation.to(device)
        self.cam = cam.to(device)
        
        self.penetration_loss = DistanceFieldPenetrationLoss().to(device)
        # collision_idxs = []
        # for _ in tqdm(range(100)):
        #     collision_idxs.append(get_collision_idxs(self.faces.cpu(), n_neighbor=10, type='random'))
        # self.collision_idxs = torch.stack(collision_idxs)

    def init_logdir(self):
        """ Create the log dir space and associated subdirs """
        log_dir_root = os.path.join(self.args.save_dir, self.args.name)
        os.makedirs(log_dir_root, exist_ok=True)
        self.args.log_dir = log_dir_root
        print(f"\n*** LOG_DIR = {self.args.log_dir} ***")

        self.args.ckpt_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        # tensorboard
        self.writer = SummaryWriter(self.args.log_dir)

        # save hparams
        with open(os.path.join(self.args.log_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, default_flow_style=False)

    def checkpoint(self, tag, extra_dict={}):
        save_dict = {'epoch': self.current_epoch,
                     'iter': self.current_iter,
                     'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}

        if hasattr(self.model, 'quantizer'):
            save_dict.update({'balance_stats': self.model.quantizer.get_state()})

        save_dict.update(extra_dict)
        torch.save(save_dict, os.path.join(self.args.ckpt_dir, tag + ".pt"))

    def get_vertices(self, pose, transl=None):
        out = self.bm(global_orient=pose[:, 0], hand_pose=pose[:, 1:].flatten(1), transl=transl, betas=self.betas.repeat(pose.size(0), 1))
        j3d = update_mano_joints(out.joints, out.vertices)
        return out.vertices, j3d

    def forward_one_batch(self, j2d_in, j3d_in,
                          K, image_size, mask,
                          j2d, j3d, 
                           valid, K_inverse, loss_type, training=True, mano_hand_pose=None, mano_valid=None):
        bs, tdim, *_ = j2d_in.size()

        # PoseBERT
        x = self.model.preprocessing(j2d_in, j3d_in, K, image_size)
        pose_hat, loc_hat = self.model.forward(x.to(self.device), mask.bool().to(self.device))
        points = loc_hat[..., :2] * image_size.unsqueeze(1).to(self.device)
        nearness = loc_hat[..., -1:]
        distance = 1. / torch.exp(nearness)
        points = torch.cat([points, torch.ones_like(points[..., :1])], -1).flatten(0, 1).unsqueeze(1) # [bs x tdim,1,3]
        K_inverse_ = K_inverse.unsqueeze(1).repeat(1, tdim, 1, 1).flatten(0, 1).to(self.device) # [bs x tdim,3,3]
        points = torch.einsum('bij,bkj->bki', K_inverse_, points)
        transl_hat = points * distance.flatten(0, 1).unsqueeze(-1)
        transl_hat = transl_hat.reshape(bs, tdim, 3)

        # MANO
        idx = torch.randperm(bs*tdim)[:self.args.n_mano] if training else torch.arange(0, bs*tdim)
        transl_hat_, pose_hat_ = transl_hat.flatten(0, 1)[idx], pose_hat.flatten(0, 1)[idx]
        verts_hat_, j3d_hat_ = self.get_vertices(roma.rotmat_to_rotvec(pose_hat_), transl_hat_)

        # Projecting 3D into 2D
        K_ = K.unsqueeze(1).repeat(1, tdim, 1, 1).flatten(0,1)[idx].to(self.device)
        j3d_hat_up = j3d_hat_ / (j3d_hat_[..., [-1]] + 1e-8)
        j2d_hat_ = torch.einsum('bij,bkj->bki', K_, j3d_hat_up)[..., :2]

        # GT
        j3d_, j2d_ = j3d.flatten(0, 1)[idx].to(self.device), j2d.flatten(0, 1)[idx].to(self.device)
        valid_ = valid.flatten(0, 1)[idx].float().to(self.device)

        # Loss on j3d
        l1_j3d = ((j3d_ - j3d_hat_).abs().sum(-1).sum(-1)* valid_).sum() / (valid_.sum() + 1e-3)
        l2_j3d = (((j3d_ - j3d_hat_)**2).sum(-1).sum(-1) * valid_).sum() / (valid_.sum() + 1e-3)

        # Loss on MANO hand pose params
        l1_mano, l2_mano = 0., 0.
        if mano_hand_pose is not None:
            mano_hand_pose = mano_hand_pose.unsqueeze(1).repeat(1, tdim, 1, 1)
            hand_pose = roma.rotvec_to_rotmat(mano_hand_pose).to(self.device)
            hand_pose_hat = pose_hat[:, :, 1:]
            mano_valid = mano_valid.to(self.device).repeat(1, tdim)
            l1_mano = ((hand_pose - hand_pose_hat).abs().sum([3, 4]).mean(-1) * mano_valid).sum() / (mano_valid.sum() + 1e-3)
            l2_mano = (((hand_pose - hand_pose_hat)**2).sum([3, 4]).mean(-1) * mano_valid).sum() / (mano_valid.sum() + 1e-3)

        # Reprojection loss
        l2_j2d, l1_j2d = 0., 0.
        if self.args.reprojection_loss:
            # Norm 2d
            if True:
                image_size_ = image_size.max(1).values.unsqueeze(1).repeat(1, tdim).flatten(0,1)[idx].to(self.device)
                j2d_hat_ = j2d_hat_ / image_size_.reshape(-1, 1, 1)
                j2d_ = j2d_ / image_size_.reshape(-1, 1, 1)
            # loss
            l1_j2d = ((j2d_ - j2d_hat_).abs().sum(-1).mean(-1)* valid_).sum() / (valid_.sum() + 1e-3)
            l2_j2d = (((j2d_ - j2d_hat_)**2).sum(-1).mean(-1)* valid_).sum() / (valid_.sum() + 1e-3)
        
        # Interpenetration loss
        loss_penetration = 0.
        if training and self.args.n_penetration > 0:
            triangles = verts_hat_[:self.args.n_penetration, self.faces.long()]
            collision_idxs = get_collision_idx_batch(self.faces, n_neighbor=self.args.n_neighbor, bs=triangles.shape[0]).to(self.device)
            # collision_idxs = self.collision_idxs[torch.randperm(self.collision_idxs.shape[0])[0]]
            # collision_idxs = collision_idxs.unsqueeze(0).repeat(triangles.shape[0], 1, 1).to(self.device)
            loss_penetration = self.penetration_loss(triangles, collision_idxs)
            loss_penetration = (loss_penetration * valid_[:self.args.n_penetration]).sum() / (valid_[:self.args.n_penetration].sum() + 1e-3)

        total_loss = l1_j3d + l2_j3d + \
                     l1_j2d + l2_j2d + \
                     l1_mano + l2_mano + \
                     loss_penetration
        losses = {'l1_j3d': l1_j3d, 'l2_j3d': l2_j3d, 'total': total_loss,
        'l1_j2d': l1_j2d, 'l2_j2d': l2_j2d, 'dist_field_penetration': loss_penetration,
        'l1_mano': l1_mano, 'l2_mano': l2_mano,
            }
        # print(losses)
        # ipdb.set_trace()
        
        if not training:
            verts_hat_, j3d_hat_, transl_hat_ = verts_hat_.reshape(bs, tdim, -1, 3), j3d_hat_.reshape(bs, tdim, -1, 3), transl_hat_.reshape(bs, tdim, 3)
        return total_loss, losses, verts_hat_, j3d_hat_, transl_hat_

    def visu_verts(self, verts, valid, name, seq, timesteps, j2d, j3d, j3d_hat, tmp_dir = "/tmp/roar", focal_length=2*530/960):
        tdim = verts.shape[0]
        os.makedirs(tmp_dir, exist_ok=True)
        for t in range(tdim):
                        if valid[t].item():
                            # RGB image
                            if seq is not None:
                                img_rgb = np.asarray(Image.open(os.path.join(seq, 'color', f"frame{int(timesteps[t].item()):03d}.png"))) # [H,W,3]
                            else:
                                img_rgb = np.zeros((960, 960, 3)).astype(np.uint8)
                            j2d_t_x = torch.clamp(j2d[t][:,0], 0, img_rgb.shape[1])
                            j2d_t_y = torch.clamp(j2d[t][:,1], 0, img_rgb.shape[0])
                            j2d_t = torch.stack([j2d_t_x, j2d_t_y], -1)
                            img_dope = visu_pose2d(img_rgb, j2d_t)

                            # J3D
                            img_j3d = visu_pose3d(j3d[[t]] - j3d[[t], [0]], res=img_rgb.shape[0])
                            img_j3d_hat = visu_pose3d(j3d_hat[[t]] - j3d_hat[[t], [0]], res=img_rgb.shape[0])

                            # MANO Rendering
                            img_mano = self.renderer.renderPerspective(vertices=[verts[t]], 
                                        faces=[self.faces],
                                        rotation=self.rotation,
                                        camera_translation=self.cam,
                                        # K=K.to(self.device),
                                        # principal_point=principal_point.to(self.device),
                                        focal_length=focal_length,
                                        color=[torch.Tensor([[0., 0.7, 1.]]).to(self.device)],
                                        ).cpu().numpy()[0]
                            if img_mano.shape[0] == img_rgb.shape[0] and img_mano.shape[1] != img_rgb.shape[1]:
                                delta = (img_mano.shape[1] - img_rgb.shape[1])//2
                                img_mano = img_mano[:, delta:-delta]
                            else:
                                delta = (img_mano.shape[0] - img_rgb.shape[0])//2
                                img_mano = img_mano[delta:-delta, :]

                            # Overlaid of MANO rendering on RGB image
                            alpha = 0.8
                            fg_mask = (np.sum(img_mano, axis=-1) != 0)
                            fg_mask = np.concatenate((fg_mask[:,:,None], fg_mask[:,:,None], fg_mask[:,:,None]), axis=2)
                            img_mano = (fg_mask * (alpha * img_mano + (1.0-alpha) * img_rgb) + (1-fg_mask) * img_rgb).astype(np.uint8)

                            # Cat
                            img = np.concatenate([img_dope, img_mano, img_j3d, img_j3d_hat], 1)

                            Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")
        cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name} -y"
        os.system(cmd)
        os.system(f"rm {tmp_dir}/*.jpg")

        # print(cmd)
        # ipdb.set_trace()

        return 1

    def perturbate_input(self, j2d, j3d, valid):
        j2d_noise = (self.args.noise_2d ** 0.5) * torch.randn_like(j2d)
        j2d_wrist_noise = (self.args.noise_2d_wrist ** 0.5) * torch.randn_like(j2d[:,:,[0]])
        # print(j2d_noise.max(), j2d_wrist_noise.max())
        j3d_noise = (self.args.noise_3d ** 0.5) * torch.randn_like(j3d)
        j2d = j2d + j2d_noise + j2d_wrist_noise
        j3d = j3d + j3d_noise
        mask_ = (torch.rand(valid.size()) > self.args.perc_mask).float()
        mask = (mask_ * valid.float()).long()
        return j2d, j3d, mask

    def train_n_iters(self, data, loss_type):
        self.model.train()

        for x in tqdm(data):
            # Input perturbation
            j2d_in, j3d_in, mask = self.perturbate_input(x['j2d'], x['j3d'], x['valid'])

            # Forward
            total_loss, losses, *_ = self.forward_one_batch(j2d_in, j3d_in, x['K'], x['image_size'], x['valid'], x['j2d'], x['j3d'], mask, x['K_inverse'], loss_type, True, mano_hand_pose=x['manoHandPose'], mano_valid=x['manoValid'])

            # Training step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Write into tensorboard
            if self.current_iter % (self.args.log_freq - 1) == 0 and self.current_iter > 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train_loss/{k}", v, self.current_iter)

            self.current_iter +=1

            # Visu
            # if True:
            #     with torch.no_grad():
            #         i = 0
            #         _, _, verts_hat, j3d_hat, *_ = self.forward_one_batch(j2d_in[[i]], 
            #                                                  j3d_in[[i]], 
            #                                                  x['K'][[i]], 
            #                                                  x['image_size'][[i]], 
            #                                                  x['valid'][[i]], 
            #                                                  x['j2d'][[i]], 
            #                                                  x['j3d'][[i]], 
            #                                                  mask[[i]], 
            #                                                  x['K_inverse'][[i]], 
            #                                                  loss_type, training=False)
            #         self.visu_verts(verts_hat[i], mask[i], "video.mp4", None, None, j2d_in[i], j3d_in[i], j3d_hat[i], focal_length=2* 530./960.)
            #     ipdb.set_trace()

        print("losses: ", losses)

    @torch.no_grad()
    def eval(self, data, *, loss_type, epoch):
        self.model.eval()

        mets = {'pck2d': AverageMeter('pck2d', ':6.3f'),
                'dist2d': AverageMeter('dist2d', ':6.3f')
        }
        n_visu_saved = 0
        for x in tqdm(data):
            assert x['dope2d'].shape[0] == 1 # batchsize == 1

            # Input perturbation
            j2d_in, j3d_in, valid = x['dope2d'], x['dope3d'], x['valid']
            valid = valid.to(self.device)
            mask = valid

            # Forward
            _, _, verts_hat, j3d_hat, _ = self.forward_one_batch(j2d_in, j3d_in, x['K'], x['image_size'], x['valid'], x['j2d'], x['j3d'], mask, x['K_inverse'], loss_type, training=False)

            # Project into 2D
            bs, tdim, *_ = j3d_in.shape
            K_ = x['K'].unsqueeze(1).repeat(1, tdim, 1, 1).flatten(0,1).to(self.device)
            j3d_hat_up = j3d_hat[0] / (j3d_hat[0][..., [-1]] + 1e-8)
            j2d_hat = torch.einsum('bij,bkj->bki', K_, j3d_hat_up)[..., :2]

            # PCK2D
            j2d = x['j2d'][0].numpy()
            dist = np.sqrt(((j2d - j2d_hat.cpu().numpy()) ** 2).sum(-1))

            # Bbox
            bbox_size = np.max(j2d, 1) - np.min(j2d, 1)
            bbox_size = np.max(bbox_size, 1).reshape(-1, 1)
            thresh = 0.1 * bbox_size # a percent of the bbox
            pck2d = 100. * (dist < thresh).mean(1)  # [T]
            pck2d_ = np.asarray([pck2d[j] for j in range(j2d.shape[0]) if x['valid'][0][j] == 1]).mean()
            dist_ = np.asarray([dist_[j].mean() for j in range(j2d.shape[0]) if x['valid'][0][j] == 1]).mean()
            mets['pck2d'].update(pck2d_)
            mets['dist2d'].update(dist_)

            # MPJPE
            # vsum = lambda y: ((y * valid).sum(-1) / valid.sum(-1)).mean()
            # mm = 1000.
            # mpjpe = mm * (((j3d_hat_ - x['j3d'].to(self.device))**2).sum(-1)).mean(-1)
            # mpjpe = vsum(mpjpe)
            # mets['mpjpe'].update(mpjpe)

            # Visu
            if n_visu_saved < self.args.n_visu_to_save:
                bs = verts_hat.shape[0]
                i = 0
                while i + n_visu_saved < self.args.n_visu_to_save and i < bs:
                    name = f"{self.args.log_dir}/visu/{self.current_epoch:06d}/{i:06d}.mp4"
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    K = x['K'][i].clone()
                    # K[:2, :2] = 2* K[:2, :2] / torch.max(x['image_size'][[i]])
                    # K[:2, -1] = (K[:2, -1] - x['image_size'][i] /2. )
                    # K[:2, -1] = K[:2, -1] / x['image_size'][i].max()
                    # K_start = torch.cat([K[:2], torch.zeros(2, 1).float()], 1)
                    # K_end = torch.Tensor([[0,    0,    0,   1],[0,    0,    1,   0]]).float()
                    # K_ = torch.cat([K_start, K_end]).unsqueeze(0)
                    verts = verts_hat[i]
                    seq = x['seq'][i]
                    timesteps = x['t_frames'][i]
                    valid = x['valid'][i]
                    j2d_ = j2d_in[i]
                    j3d_ = j3d_in[i]
                    self.visu_verts(verts, valid, name, seq, timesteps, j2d_, j3d_, j3d_hat[i], focal_length=2* K[0,0]/960.)
                    n_visu_saved +=1
                    i += 1

        # Log
        for k, v in mets.items():
            print(f"    - {k}: {v.avg:.4f}")
            self.writer.add_scalar(f"val/{k}", v.avg, self.current_iter)
        
        return mets['pck2d'].avg

    def fit(self, data_train, data_val, *, loss='l2', best_val=None):
        for epoch in range(1, self.args.max_epochs):
            sys.stdout.flush()

            print(f"\nEPOCH={epoch:03d}/{self.args.max_epochs} - ITER={self.current_iter}")

            # Train for n_iters
            self.train_n_iters(data_train, loss_type=loss)

            # Eval
            val = self.eval(data_val, loss_type=loss, epoch=epoch)

            # Save ckpt if good enough
            if val < self.best_val:
                print("Saving ckpt")
                self.checkpoint(tag='best_val', extra_dict={'mpjpe': val})
                self.best_val = val

            self.current_epoch += 1

        return None


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Training PoseBERT on ContactPose')
    parser.add_argument('--data_dir', type=str, default='/tmp-network/user/fbaradel/projects/ROAR/data/contactpose')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument("--save_dir", type=str, default='logs')
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-b_train", type=int, default=32)
    parser.add_argument("--val_batch_size", "-b_val", type=int, default=4)
    parser.add_argument("--cam", type=str, default='kinect2_middle', choices=['kinect2_middle', 'kinect2_right', 'kinect2_left', 'all'])
    parser.add_argument("--train_split", type=str, default='test')
    parser.add_argument("--val_split", type=str, default='test')
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_epochs", type=int, default=600)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", "-j", type=int, default=0)    
    parser.add_argument("--loss", type=str, default='l2')
    parser.add_argument("--n_mano", type=int, default=256)
    parser.add_argument("--n_neighbor", type=int, default=8)
    parser.add_argument("--n_penetration", type=int, default=128)
    parser.add_argument("--n_visu_to_save", type=int, default=1)
    parser.add_argument("--noise_2d", type=float, default=0)
    parser.add_argument("--noise_2d_wrist", type=float, default=0)
    parser.add_argument("--noise_3d", type=float, default=0)
    parser.add_argument("--perc_mask", type=float, default=0)
    parser.add_argument("--test_time_noise", type=int, default=0, choices=[0,1])
    parser.add_argument("--reprojection_loss", type=int, default=1, choices=[0,1])
    parser.add_argument("--overfit", type=int, default=0, choices=[0,1])
    parser.add_argument("--data_augment", type=int, default=1, choices=[0,1])

    parser = PoseBERT.add_specific_args(parser)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Overfitting
    n = -1
    if args.overfit:
        args.train_split = args.val_split
        n = 1
        print("\nWARNING! Overfitting on a single example!!!!!")

    # Data
    print(f"\nLoading data...")
    loader_train = DataLoader(J3dDataset(data_dir=args.data_dir, training=True, data_augment=args.data_augment, split=args.train_split, cam=args.cam, seq_len=args.seq_len, max_tilt_angle=180, max_roll_angle=180, max_vertical_angle=180, n_iter=args.train_batch_size * args.iter, sanity_check_inverse_projection=False, n=n), batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)
    loader_val = DataLoader(ContactPoseDataset(data_dir=args.data_dir, split=args.val_split, cam=args.cam, n=n), batch_size=1, num_workers=0, shuffle=False, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=False)

    print(f"Data - N_train={len(loader_train.dataset.seq)} - N_val={len(loader_val.dataset.seq)}")

    # Model
    print(f"\nBuilding the model...")
    # model = Model(in_dim=in_dim, **vars(args)).to(device)
    model = PoseBERT(n_jts_in=NUM_JOINTS_HAND+1, n_j2d_in=NUM_JOINTS_HAND, n_extra_in=9).to(device) # TODO add args
    if args.n_mano == -1:
        args.n_mano = args.train_batch_size * args.seq_len
        print(f"n_mano: {args.n_mano}")

    # Pretrained ckpt
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        ckpt_path = args.pretrained_ckpt
    else:
        checkpoint, ckpt_path = get_last_checkpoint(args.save_dir, args.name)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Ckpt model params succesfully loaded from: {ckpt_path}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Checkpoint again
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch, saved_iter = [checkpoint[k] for k in ['epoch', 'iter']]
        bv, bc = [checkpoint[k] if k in checkpoint else None for k in ['best_val', 'best_class']]
        print(f"Ckpt optimizer params succesfully loaded from: {ckpt_path}")
    else:
        epoch, saved_iter = 0, 0
        bv, bc = None, None

    # Trainer
    print(f"\nSetting up the trainer...")
    trainer = Trainer(model=model, optimizer=optimizer, device=device, args=args,
                 epoch=epoch, start_iter=saved_iter, seq_len=args.seq_len,
                 best_val=bv)

    # Training or testing
    if args.eval_only:
        val = trainer.eval(loader_val, loss_type=args.loss, epoch=0)
    else:
        trainer.fit(loader_train, loader_val, loss=args.loss)

    

if __name__ == "__main__":
    main(None)
