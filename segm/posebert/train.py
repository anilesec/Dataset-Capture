import torch
import argparse
from posebert.model import PoseBERT
from posebert.utils import get_last_checkpoint, AverageMeter, generate_masks
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from posebert.dataset import ManoDataset, worker_init_fn
import ipdb
import smplx
from posebert.constants import SMPLX_DIR, LOGS_DIR
import numpy as np
from posebert.renderer import PyTorch3DRenderer
from pytorch3d.renderer import look_at_view_transform
from posebert.skeleton import update_mano_joints, visu_pose2d, perspective_projection, get_mano_skeleton, visu_pose3d
import os
import yaml
import sys
from tqdm import tqdm
import roma
from PIL import Image

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
        for param in self.bm.parameters():
            param.requires_grad = False

        self.faces = torch.from_numpy(np.array(self.bm.faces, dtype=np.int32)).to(device)
        self.init_logdir()
        self.current_iter = start_iter
        self.current_epoch = epoch
        self.model = model

        # Camera/Rendering setup
        # self.args.width, self.args.height = 1280, 720
        # self.args.f_x, self.args.f_y = 901.5, 901.7
        # self.args.c_x, self.args.c_y = 664.1, 380.3
        self.img_res = max([self.args.width, self.args.height])
        self.ratio = torch.Tensor([[self.img_res/self.args.width, self.img_res/self.args.height]]).float()
        self.focal_length = torch.Tensor([[2*self.args.f_x/self.img_res, 2*self.args.f_y/self.img_res]])
        self.principal_point = torch.Tensor([[self.args.c_x/self.args.width, self.args.c_y/self.args.height]])
        self.principal_point = (self.principal_point - 0.5) * 2 # values should be between -1 and 1. (0,0) is the image center
        self.principal_point /= self.ratio # solving principal point issue for rendering with non-square image
        self.renderer = PyTorch3DRenderer(self.img_res).to(device)
        dist, elev, azim = 1e-5, 0., 180
        rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        self.rotation = rotation.to(device)
        self.cam = cam.to(device)

        # Masks
        if self.args.mask_block:
            self.list_masks = generate_masks(seq_len, list_perc_mask=range(int(self.args.masking//2), int(self.args.masking)), chunk_size=[1, int(self.args.masking/100. * self.args.seq_len)], n=self.args.n_masks)
            print(f"Percentage of masked timesteps: {100. * (1. - self.list_masks.mean()):.1f}%")
        
        # Random blocks
        if self.args.random_block:
            self.list_randoms = generate_masks(seq_len, list_perc_mask=range(int(self.args.random//2), int(self.args.random)), chunk_size=[1, int(self.args.masking/100. * self.args.random)], n=self.args.n_masks)

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

    def generate_noise(self, j3d, j2d):
        # Masking
        if self.args.mask_block:
            idx_mask = np.random.choice(range(len(self.list_masks)), size=j3d.shape[0])
            mask = torch.from_numpy(self.list_masks[idx_mask]).to(self.device)
        else:
            mask = (torch.rand(j3d.size()[:2]) > self.args.masking/100.).float()
            mask = mask.to(self.device)
                    
        # Random pose
        if self.args.random > 0:
            
            # Per block
            if self.args.random_block:
                ipdb.set_trace()
                # TODO
                # idx_random = np.random.choice(range(len(self.list_randoms)), size=j3d.shape[0])
                # rand_ = torch.from_numpy(self.list_randoms[idx_random]).to(self.device)
                # ipdb.set_trace()
            else:
                # At random
                def random_pose_inversion(x, random):
                    bs, tdim, n_jts = x.shape[:3]
                    x_ = x.clone().flatten(0, 1)
                    bs_ = x_.shape[0]
                    idx1 = torch.randperm(bs_)[:int(random/100. * bs_)]
                    idx2 = torch.randperm(bs_)[:int(random/100. * bs_)]
                    x_idx1 = x_[idx1].clone()
                    x_idx2 = x_[idx2].clone()
                    x_[idx2] = x_idx1
                    x_[idx1] = x_idx2
                    x_ = x_.reshape(bs, tdim, n_jts, -1)
                    # print("random pose: ", self.args.random, idx1.shape, idx2.shape, (x - x_).abs().sum())
                    return x_

                # 3d and 2d
                j3d = random_pose_inversion(j3d, self.args.random)
                j2d = random_pose_inversion(j2d, self.args.random)

        # Noise for 3d
        j3d_noise = (self.args.noise_3d ** 0.5) * torch.randn_like(j3d)
        transl_noise = (self.args.noise_transl ** 0.5) * torch.randn_like(j3d[:,:,:1])
        j3d = j3d + j3d_noise + transl_noise

        # Noise for 2d
        noise_2d1 = (self.args.noise_2d_disp ** 0.5) * torch.randn_like(j2d)
        noise_2d2 = (self.args.noise_2d_loc ** 0.5) * torch.randn_like(j2d[:,:,:1])
        j2d = j2d + noise_2d1 + noise_2d2

        return j3d, j2d, mask

    def eval(self, data, epoch, eval_anil_only=False):
        self.model.eval()
        
        with torch.no_grad():
            mets = {'pve': AverageMeter('pve', ':6.3f')}
            if not eval_anil_only:
                n_visu_saved = 0
                for x in tqdm(data):
                    sys.stdout.flush()
                    
                    # Move to device
                    x = [x_i.to(self.device) for x_i in x]
                    j3d, global_orient, transl, hand_pose = x
                    
                    # Project to 2d
                    bs, tdim = j3d.shape[:2]
                    j2d = perspective_projection(j3d.flatten(0,1), self.args.c_x, self.args.c_y, self.args.f_x, self.args.f_y).reshape(bs, tdim, 21, 2)

                    # Generate noise
                    j3d, j2d, mask = self.generate_noise(j3d, j2d)

                    # Forward
                    # torch.isnan(mask).float().sum()
                    # torch.isnan(j3d).float().sum()
                    # torch.isnan(j2d).float().sum()
                    global_orient_hat, transl_hat, hand_pose_hat = self.model(j3d=j3d, j2d=j2d, mask=mask.bool())

                    # Move to vertices
                    bs, tdim = j3d.shape[:2]
                    verts = self.bm(global_orient=global_orient.flatten(0,1), hand_pose=hand_pose.reshape(-1, 1, 15, 3).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1), transl=transl.flatten(0,1).flatten(1), betas=self.bm.betas.repeat(bs*tdim, 1)).vertices
                    verts_hat = self.bm(global_orient=roma.rotmat_to_rotvec(global_orient_hat).flatten(0,1), hand_pose=roma.rotmat_to_rotvec(hand_pose_hat.reshape(-1, 1, 15, 3, 3)).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1), transl=transl_hat.flatten(0,1).flatten(1), betas=self.bm.betas.repeat(bs*tdim, 1)).vertices
                    verts, verts_hat = verts.reshape(bs, tdim, -1, 3), verts_hat.reshape(bs, tdim, -1, 3)

                    # PVE
                    pve = 1000. * ((verts - verts_hat)**2).sum(-1).mean(-1).mean()
                    mets['pve'].update(pve)

                    # Visu
                    if n_visu_saved < self.args.n_visu_to_save:
                        bs = verts_hat.shape[0]
                        i = 0
                        while i + n_visu_saved < self.args.n_visu_to_save and i < bs:
                            name = f"{self.args.log_dir}/visu_val/{self.current_epoch:06d}/{i:06d}.mp4"
                            os.makedirs(os.path.dirname(name), exist_ok=True)
                            tmp_dir = "/scratch/1/user/fbaradel/tmp/roar_posebert"
                            os.makedirs(tmp_dir, exist_ok=True)
                            for t in range(verts.shape[1]):
                                img2 = self.renderer.renderPerspective(vertices=[verts_hat[i,t]], 
                                                faces=[self.faces],
                                                rotation=self.rotation,
                                                camera_translation=self.cam,
                                                # focal_length=2*self.focal_length/self.img_res,
                                                principal_point=self.principal_point.to(self.device),
                                                focal_length=self.focal_length.to(self.device),
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(self.device)],
                                                ).cpu().numpy()[0]
                                img3 = self.renderer.renderPerspective(vertices=[verts[i,t]], 
                                                faces=[self.faces],
                                                rotation=self.rotation,
                                                camera_translation=self.cam,
                                                # focal_length=2*self.focal_length/self.img_res,
                                                principal_point=self.principal_point.to(self.device),
                                                focal_length=self.focal_length.to(self.device),
                                                color=[torch.Tensor([[0., 0.7, 1.]]).to(self.device)],
                                                ).cpu().numpy()[0]
                                img2 = self.postprocess_img(img2)
                                img3 = self.postprocess_img(img3)

                                # project 3d input into 2d
                                j2d_ = perspective_projection(j3d[i,[t]], self.args.c_x, self.args.c_y, self.args.f_x, self.args.f_y)
                                img1 = visu_pose2d(img2.copy() * 0, j2d_[0], get_mano_skeleton(), lw_line= 1, lw_dot = 3)

                                # 2d input
                                img1 = visu_pose2d(img1, j2d[i,t], get_mano_skeleton(), lw_line= 3, lw_dot = 3)

                                # 3d pose centered
                                j3d_ = j3d[i,[t]] - j3d[i,[t],[0]] # center around wrist
                                img0 = visu_pose3d(j3d_, res=self.img_res, bones=get_mano_skeleton())
                                img0 = self.postprocess_img(img0)

                                # j2d = perspective_projection(j3d[i,[t]], self.args.c_x, self.args.c_y, self.args.f_x, self.args.f_y)
                                # # print(j2d)
                                # img1 = visu_pose2d(img2.copy() * 0, j2d[0], get_mano_skeleton())
                                # img1 = img1 * mask[i,t].int().item()

                                img = np.concatenate([img0, img1, img2, img3], 1)
                                Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")
                            cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name} -y"
                            os.system(cmd)
                            os.system(f"rm {tmp_dir}/*.jpg")
                            n_visu_saved +=1
                            i += 1
                            
                            # os.system(f"cp {name} video.mp4")
                            # ipdb.set_trace()


                # Log
                for k, v in mets.items():
                    print(f"    - {k}: {v.avg:.4f}")
                    self.writer.add_scalar(f"val/{k}", v.avg, self.current_iter)

        if self.args.eval_anil == 1:
            # Visu Anil's sequence
            from posebert.demo import visu
            for name in ['20220614171716_red_duster', '20220614171338_green_duster', '20220614171547_green_duster']:
                videoname = f"{self.args.log_dir}/visu_anil/{self.current_epoch:06d}/{name}.mp4"
                visu(self.model, seqname=name, t_max=self.args.seq_len_anil, debug=False, name = videoname, 
                     f_x=self.args.f_x, f_y=self.args.f_y,
                     c_x=self.args.c_x, c_y=self.args.c_y,
                     )
        
        return mets['pve'].avg

    def train_n_iters(self, data):
        self.model.train()

        mets = {
            'global_orient': AverageMeter('', ':6.3f'),
            'transl': AverageMeter('', ':6.3f'),
            'hand_pose': AverageMeter('', ':6.3f'),
            'verts': AverageMeter('', ':6.3f'),
            'total': AverageMeter('', ':6.3f'),
        }

        for x in tqdm(data):
            # Move to device
            x = [x_i.to(self.device) for x_i in x]
            j3d, global_orient, transl, hand_pose = x

            # Project to 2d
            bs, tdim = j3d.shape[:2]
            j2d = perspective_projection(j3d.flatten(0,1), self.args.c_x, self.args.c_y, self.args.f_x, self.args.f_y).reshape(bs, tdim, 21, 2)

            # Add some noise
            j3d, j2d, mask = self.generate_noise(j3d, j2d)

            # Forward
            global_orient_hat, transl_hat, hand_pose_hat = self.model(j3d=j3d, j2d=j2d, mask=mask.bool())

            # Losses on params
            global_orient_ = roma.rotvec_to_rotmat(global_orient)
            loss_global_orient = ((global_orient_hat - global_orient_)**2).sum([2,3]) + ((global_orient_hat - global_orient_).abs()).sum([2,3])
            loss_global_orient = loss_global_orient.mean()
            loss_transl = ((transl - transl_hat)**2).sum(-1) + ((transl - transl_hat).abs()).sum(-1)
            loss_transl = loss_transl.mean()
            hand_pose_ = roma.rotvec_to_rotmat(hand_pose.reshape(-1, 15, 3))
            loss_hand_pose = ((hand_pose_hat - hand_pose_)**2).sum([2,3]).mean(1) +  ((hand_pose_hat - hand_pose_).abs()).sum([2,3]).mean(1)
            loss_hand_pose = loss_hand_pose.mean()
            total_loss = self.args.alpha_global_orient * loss_global_orient + \
                         self.args.alpha_hand_pose * loss_hand_pose + \
                         self.args.alpha_transl * loss_transl

            # # Loss on vertices
            bs, tdim = j3d.shape[:2]
            loss_verts = 0.
            if self.args.n_verts_loss > 0 and self.args.alpha_verts > 0:
                idx = torch.randperm(bs*tdim)[:100]
                verts = self.bm(global_orient=global_orient.flatten(0,1)[idx], hand_pose=hand_pose.reshape(-1, 1, 15, 3).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1)[idx], transl=transl.flatten(0,1).flatten(1)[idx], betas=self.bm.betas.repeat(bs*tdim, 1)[idx]).vertices
                verts_hat = self.bm(global_orient=roma.rotmat_to_rotvec(global_orient_hat).flatten(0,1)[idx], hand_pose=roma.rotmat_to_rotvec(hand_pose_hat.reshape(-1, 1, 15, 3, 3)).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1)[idx], transl=transl_hat.flatten(0,1).flatten(1)[idx], betas=self.bm.betas.repeat(bs*tdim, 1)[idx]).vertices
                loss_verts = (((verts - verts_hat)**2) + (verts - verts_hat).abs()).sum(-1).sum(-1).mean()
                total_loss = total_loss + self.args.alpha_verts * loss_verts

            mets['global_orient'].update(loss_global_orient)
            mets['hand_pose'].update(loss_hand_pose)
            mets['transl'].update(loss_transl)
            mets['verts'].update(loss_verts)
            mets['total'].update(total_loss)

            # Training step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Write into tensorboard
            if self.current_iter % (self.args.log_freq - 1) == 0 and self.current_iter > 0:
                print(f"iter={self.current_iter:06d} - loss={total_loss.item():.4f}")
                sys.stdout.flush()
                for k, v in mets.items():
                    self.writer.add_scalar(f"train_loss/{k}", v.avg, self.current_iter)
                    mets[k].reset()

            self.current_iter +=1

        # Visu
        n_visu_saved = 0
        if n_visu_saved < self.args.n_visu_to_save:
            with torch.no_grad():
                verts = self.bm(global_orient=global_orient.flatten(0,1), hand_pose=hand_pose.reshape(-1, 1, 15, 3).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1), transl=transl.flatten(0,1).flatten(1), betas=self.bm.betas.repeat(bs*tdim, 1)).vertices
                verts_hat = self.bm(global_orient=roma.rotmat_to_rotvec(global_orient_hat).flatten(0,1), hand_pose=roma.rotmat_to_rotvec(hand_pose_hat.reshape(-1, 1, 15, 3, 3)).repeat(1, tdim, 1, 1).flatten(0,1).flatten(1), transl=transl_hat.flatten(0,1).flatten(1), betas=self.bm.betas.repeat(bs*tdim, 1)).vertices
                verts, verts_hat = verts.reshape(bs, tdim, -1, 3), verts_hat.reshape(bs, tdim, -1, 3)
                i = 0
                while i + n_visu_saved < self.args.n_visu_to_save and i < bs:
                    name = f"{self.args.log_dir}/visu_train/{self.current_epoch:06d}/{i:06d}.mp4"
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    tmp_dir = "/scratch/1/user/fbaradel/tmp/roar_posebert"
                    os.makedirs(tmp_dir, exist_ok=True)
                    for t in range(verts.shape[1]):
                    # for t in tqdm(range(verts.shape[1])):
                        img2 = self.renderer.renderPerspective(vertices=[verts_hat[i,t]], 
                                            faces=[self.faces],
                                            rotation=self.rotation,
                                            camera_translation=self.cam,
                                            # focal_length=2*self.focal_length/self.img_res,
                                            principal_point=self.principal_point.to(self.device),
                                            focal_length=self.focal_length.to(self.device),
                                            color=[torch.Tensor([[0., 0.7, 1.]]).to(self.device)],
                                            ).cpu().numpy()[0]
                        img3 = self.renderer.renderPerspective(vertices=[verts[i,t]], 
                                            faces=[self.faces],
                                            rotation=self.rotation,
                                            camera_translation=self.cam,
                                            # focal_length=2*self.focal_length/self.img_res,
                                            principal_point=self.principal_point.to(self.device),
                                            focal_length=self.focal_length.to(self.device),
                                            color=[torch.Tensor([[0., 0.7, 1.]]).to(self.device)],
                                            ).cpu().numpy()[0]
                        img2 = self.postprocess_img(img2)
                        img3 = self.postprocess_img(img3)

                        # project 3d input into 2d
                        j2d_ = perspective_projection(j3d[i,[t]], self.args.c_x, self.args.c_y, self.args.f_x, self.args.f_y)
                        img1 = visu_pose2d(img2.copy() * 0, j2d_[0], get_mano_skeleton(), lw_line= 1, lw_dot = 3)

                        # 2d input
                        img1 = visu_pose2d(img1, j2d[i,t], get_mano_skeleton(), lw_line= 3, lw_dot = 3)

                        # 3d pose centered
                        j3d_ = j3d[i,[t]] - j3d[i,[t],[0]] # center around wrist
                        img0 = visu_pose3d(j3d_, res=self.img_res, bones=get_mano_skeleton())
                        img0 = self.postprocess_img(img0)

                        img1 = img1 * mask[i,t].int().item()
                        img = np.concatenate([img0, img1, img2, img3], 1)
                        Image.fromarray(img).save(f"{tmp_dir}/{t:06d}.jpg")
                    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/*.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name} -y"
                    os.system(cmd)
                    os.system(f"rm {tmp_dir}/*.jpg")

                    # os.system(f"cp {name} video.mp4")
                    # ipdb.set_trace()

                    n_visu_saved +=1
                    i += 1

    def postprocess_img(self, img):
        delta = np.abs(self.args.width - self.args.height)//2
        if delta > 0:
            if self.args.width > self.args.height:
                img = img[delta:self.args.height+delta]
            else:
                img = img[:,delta:self.args.width+delta]
        return img

    def fit(self, data_train, data_val):
        for epoch in range(1, self.args.max_epochs):
            sys.stdout.flush()

            print(f"\nEPOCH={epoch:03d}/{self.args.max_epochs} - ITER={self.current_iter}")

            # Train for n_iters
            self.train_n_iters(data_train)

            # Eval
            val = self.eval(data_val, epoch=epoch)

            # Save ckpt if good enough
            self.checkpoint(tag='last', extra_dict={'pve': val})
            if val < self.best_val:
                print("Saving ckpt")
                self.checkpoint(tag='best_val', extra_dict={'pve': val})
                self.best_val = val
            
            self.current_epoch += 1

        return None

def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser(description='Training PoseBERT')
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument("--train_split", type=str, default='train')
    parser.add_argument("--val_split", type=str, default='test')
    parser.add_argument("--overfit", type=int, default=0, choices=[0,1,10])
    parser.add_argument("--input_type", type=str, default='j3dj2d', choices=['j3d', 'j2d', 'j3dj2d', 'j2dj3d'])
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-b_train", type=int, default=32)
    parser.add_argument("--val_batch_size", "-b_val", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default=LOGS_DIR)
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--num_workers", "-j", type=int, default=0)    
    parser.add_argument("--masking", type=float, default=0.)
    parser.add_argument("--n_masks", type=int, default=1000)
    parser.add_argument("--mask_block", type=int, default=0, choices=[0, 1])
    parser.add_argument("--random_block", type=int, default=0, choices=[0, 1])
    parser.add_argument("--random", type=float, default=0.)
    parser.add_argument("--noise_3d", type=float, default=0.)
    parser.add_argument("--noise_transl", type=float, default=0.)
    parser.add_argument("--noise_2d_loc", type=float, default=0.)
    parser.add_argument("--noise_2d_disp", type=float, default=0.)
    parser.add_argument("--n_visu_to_save", type=int, default=1)
    parser.add_argument("--alpha_global_orient", type=float, default=1.)
    parser.add_argument("--alpha_transl", type=float, default=100.)
    parser.add_argument("--alpha_hand_pose", type=float, default=1.)
    parser.add_argument("--alpha_verts", type=float, default=0.1)
    parser.add_argument("--n_verts_loss", type=float, default=400)
    parser.add_argument("--eval_anil_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_anil", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seq_len_anil", type=int, default=256)
    parser.add_argument("--focal_length_anil", type=int, default=900)
    parser.add_argument("--use_input_transl", type=int, default=1, choices=[0,1])
    

    # Camera params
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--f_x", type=float, default=899.783)
    parser.add_argument("--f_y", type=float, default=900.019)
    parser.add_argument("--c_x", type=float, default=653.768)
    parser.add_argument("--c_y", type=float, default=362.143)

    parser = PoseBERT.add_specific_args(parser)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Overfitting
    n = -1
    if args.overfit > 0:
        n = args.overfit
        args.train_split = args.val_split
        print("\nWARNING! Overfitting on a single example!!!!!")

    # Data
    # range_x_init=[-1.,1]
    # range_y_init=[-1.,1.]
    # range_z_init=[0.5,3.]
    range_x_init=[-0.1,0.1]
    range_y_init=[-0.1,0.1]
    range_z_init=[0.3,0.5]
    kwargs = {'range_x_init': range_x_init,'range_y_init': range_y_init, 'range_z_init': range_z_init}
    print(f"\nLoading data...")
    loader_train = DataLoader(ManoDataset(seq_len=args.seq_len, training=True, split=args.train_split, n_iter=args.train_batch_size * args.iter, n=n, **kwargs), batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=True)
    loader_val = DataLoader(ManoDataset(seq_len=args.seq_len, training=False, split=args.val_split, n=n, **kwargs), batch_size=args.val_batch_size, num_workers=args.num_workers, shuffle=False, worker_init_fn=worker_init_fn, pin_memory=False, drop_last=False)

    print(f"Data - N_train={len(loader_train.dataset.list_pose)} - N_val={len(loader_val.dataset.list_pose)}")

    # Model
    print(f"\nBuilding the model...")
    model = PoseBERT(use_input_transl=args.use_input_transl==1).to(device) # TODO add args

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
        val = trainer.eval(loader_val, epoch=0, eval_anil_only=args.eval_anil_only)
    else:
        trainer.fit(loader_train, loader_val)

    

if __name__ == "__main__":
    main(None)
