import torch
from torch import nn
import torch.nn.functional as F
import roma
from argparse import ArgumentParser
from einops import rearrange, repeat
import numpy as np
import math
import ipdb
from posebert.skeleton import perspective_projection, preprocess_skeleton_torch, visu_pose2d

NUM_JOINTS_HAND = 21
NUM_JOINTS_MANO = 16
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForwardResidual(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=24 * 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x, init, n_iter=1):
        pred_pose = init
        for _ in range(n_iter):
            xf = torch.cat([x, init], -1)
            pred_pose = pred_pose + self.net(xf)
        return pred_pose


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        Args:
            - x: [B,T,D]
            - mask: [B,T] - dytpe= torch.bool - default True everywhere, if False it means that we don't pay attention to this timestep
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,H,T,T]
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:  # always true
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            # mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')  # [B,1,T,T] - initial version - do not update masked - it is ok if we do not apply the loss on these timesteps
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, n, 1)  # mine - for updating masked with context
            dots.masked_fill_(~mask, mask_value)  # ~ do the opposite i.e. move True to False here
            del mask
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, type='learned', max_len=1024):
        super(PositionalEncoding, self).__init__()

        if 'sine' in type:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1,t,d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, d_model))  # [1,t,d]
            self.register_buffer('pe', pe)
        else:
            raise NameError

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class TransformerRegressor(nn.Module):
    """
    A version of the transformer which regress params iteratively
    """

    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.1, out=[22 * 6, 3],
                 share_regressor=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            list_modules = [
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]

            # Regressor
            if i == 0 or not share_regressor:
                # N regressor per layer
                for out_i in out:
                    list_modules.append(PreNorm(dim, FeedForwardResidual(dim, mlp_dim, dropout=dropout, out_dim=out_i)))
            else:
                # Share regressor across layers
                for j in range(2, len(self.layers[0])):
                    list_modules.append(self.layers[0][j])
            self.layers.append(nn.ModuleList(list_modules))

    def forward(self, x, init, mask=None):
        """
        Args:
            - x: [b,t,d]
            - init: a list of init tensor
            - mask: [b,t] or None, not really used here because this network is mainly a decoder
        Return:
            - pred_rotmat: [b,t,k,3,3]
            - pred_cam: [b,t,3]
        """
        batch_size, seq_len, *_ = x.size()
        y = init
        for layers_i in self.layers:
            # attention and feeforward module
            attn, ff = layers_i[0], layers_i[1]
            x = attn(x, mask=mask) + x
            x = ff(x) + x

            # N regressors
            for j, reg in enumerate(layers_i[2:]):
                y[j] = reg(x, init=y[j], n_iter=1)

        return y



class PoseBERT(nn.Module):
    def __init__(self, dim=256, pos_type='sine_frozen', add_encoder=0,
                 depth=4, heads=4, dim_head=64, mlp_dim=512, dropout=0.1, share_regressor=1,
                 emb_dropout=0.1,
                 input_type='j3dj2d',
                 width=1280, height=720,
                 init_transl=[0.,0.,0.3],
                 use_input_transl=True,
                 *args, **kwargs):
        super(PoseBERT, self).__init__()

        self.use_input_transl = use_input_transl
        self.input_type = input_type
        self.width, self.height = width, height
        self.pos = PositionalEncoding(dim, pos_type, 1024)
        # self.emb = nn.Linear(3+3*(NUM_JOINTS_HAND-1)+6, dim)
        self.emb = nn.Linear(3+3*(NUM_JOINTS_HAND-1)+6+2*NUM_JOINTS_HAND, dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder = None
        if add_encoder:
            self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            # one more positional embedding for the decoder then
            self.pos_dec = PositionalEncoding(dim, pos_type, 1024)
        list_out = [6, (NUM_JOINTS_MANO-1) * 6, 3]
        self.decoder = TransformerRegressor(dim, depth, heads, dim_head, mlp_dim, dropout,
                                            list_out,
                                            share_regressor == 1)

        self.register_buffer('init_global_orient', torch.zeros(6).float().reshape(1, 1, -1))
        self.register_buffer('init_hand_pose', torch.zeros((NUM_JOINTS_MANO-1) * 6).float().reshape(1, 1, -1))
        self.register_buffer('init_transl', torch.Tensor(init_transl).float().reshape(1, 1, -1))

    def forward(self, j3d, j2d=None, mask=None, debug=False, fixed_hand_pose=True):
        """
        Args:
            - j3d: [bs,tdim,21,3]
            - j2d: [bs,tdim,21,2] in pixel coordinates
            - mask: [bs,tdim]
        Return:
            - global_orient: [bs,tdim,3,3]
            - transl: [bs,tdim,3]
            - hand_pose: [bs,15,3,3]
        """
        # ipdb.set_trace()
        bs, tdim, *_ = j3d.size()

        # Default masks
        if mask is None:
            mask = torch.ones(bs, tdim).type_as(j3d).bool()

        # Default j2d
        if j2d is None:
            j2d = torch.zeros(bs, tdim, j3d.shape[2], 2).type_as(j3d).float()

        # Apply mask - even if it seems not to be necessary
        j3d = j3d * mask.unsqueeze(-1).unsqueeze(-1).float()
        j2d = j2d * mask.unsqueeze(-1).unsqueeze(-1).float()

        # Set to zero if not as input
        if 'j2d' not in self.input_type:
            j2d = j2d * 0.
        if 'j3d' not in self.input_type:
            j3d = j3d * 0.

        # Preprocessing j2d
        j2d_norm = torch.stack([j2d[...,0] / self.width, j2d[...,1] / self.height], 2).flatten(2) # [bs,tdim,21*2]
        # print("forward 1: ", torch.isnan(j2d_norm).float().sum().item())

        # Preprocessing of the j3d
        # print("forward 2: ", torch.isnan(j3d).float().sum().item())
        hand_pose, transl, global_orient = preprocess_skeleton_torch(j3d.reshape(bs*tdim, -1, 3), center_joint=[0], xaxis=[10, 1], yaxis=[4, 0], iter=1, norm_x_axis=True, norm_y_axis=True)
        if debug:
            from posebert.skeleton import visu_pose3d, get_mano_skeleton
            from PIL import Image
            img1 = visu_pose3d(hand_pose[:1].clone(), bones=get_mano_skeleton(), factor=5., res=self.height)
            bg = np.zeros((self.height, self.width, 3)).astype(np.uint8)
            img2 = visu_pose2d(bg, j2d[0,0], get_mano_skeleton())
            img = np.concatenate([img1, img2], 1)
            Image.fromarray(img).save('img.jpg')
        hand_pose = hand_pose[:,1:] # discard the wrist because of the center normalization
        global_orient = global_orient[..., :2].flatten(1) # 6D representation
        hand_pose, transl, global_orient = hand_pose.reshape(bs, tdim, -1), transl.reshape(bs, tdim, -1), global_orient.reshape(bs, tdim, -1)
        # print("forward 3: ", torch.isnan(hand_pose).float().sum().item())
        # print("forward 4: ", torch.isnan(transl).float().sum().item())
        # print("forward 5: ", torch.isnan(global_orient).float().sum().item())

        # Embedding for each modality
        x = torch.cat([hand_pose, global_orient, transl, j2d_norm], -1)
        x = self.emb(x)

        # Embedding for masking and temporal position
        x = x * mask.float().unsqueeze(-1) + self.mask_token * (1. - mask.float().unsqueeze(-1))  # masked token
        x = self.pos(x)  # inject position info - summing only for the moment
        x = self.emb_dropout(x)

        # Encoder-Decoder
        if self.encoder is not None:
            hid = self.encoder(x, mask)
            hid = self.pos_dec(hid)  # positional information
            mask = None
        else:
            hid = x

        init_global_orient = self.init_global_orient.repeat(bs, tdim, 1) # initial MANO pose
        init_hand_pose = self.init_hand_pose.repeat(bs, tdim, 1) # initial MANO pose
        
        if not self.use_input_transl:
            init_transl = self.init_transl.repeat(bs, tdim, 1) # initial transl
        else:
            # TODO init from the input transl seems to be a good idea but we need to deal with masked init translation
            # init_transl = 0. * transl.clone() # start from 0 vector (we do not use the initial guess of the transl as initialization)
            init_transl = []
            for i in range(bs):
                init_transl_i = []
                for t in range(tdim):
                    if mask[i,t]:
                        init_transl_i.append(transl[i,t])
                    else:
                        idx = torch.where(mask[i])[0]
                        if len(idx) > 0:
                            k = torch.argmin(torch.abs(idx - t))
                            init_transl_i.append(transl[i,k])
                        else:
                            init_transl_i.append(self.init_transl[0,0])
                init_transl.append(torch.stack(init_transl_i))
            init_transl = torch.stack(init_transl)

            # init_transl_x = torch.median(transl[...,0], 1).values
            # init_transl_y = torch.median(transl[...,1], 1).values
            # init_transl_z = torch.median(transl[...,2], 1).values
            # init_transl = torch.stack([init_transl_x, init_transl_y, init_transl_z], 1).repeat(1, tdim, 1)

        init = [init_global_orient, init_hand_pose, init_transl]
        ys = self.decoder(hid, init, mask)
        global_orient, hand_pose, transl = ys

        # global orientation
        global_orient = roma.special_gramschmidt(global_orient.reshape(bs, tdim, 3, 2))

        # # hand pose
        # w_i = torch.ones_like(hand_pose[:,:,:,0,0]) / tdim
        # hand_pose = torch.sum(w_i[...,None,None] * hand_pose, dim=1)
        # hand_pose = roma.special_procrustes(hand_pose)
        # hand_pose = hand_pose[:,0] # take the first one only
        if fixed_hand_pose:
            hand_pose = hand_pose.mean(1) # not sure it is great
            hand_pose = roma.special_gramschmidt(hand_pose.reshape(bs, -1, 3, 2))
        else:
            hand_pose = roma.special_gramschmidt(hand_pose.reshape(bs, tdim, -1, 3, 2))

        # print("forward 6: ", torch.isnan(global_orient).float().sum().item())
        # print("forward 7: ", torch.isnan(transl).float().sum().item())
        # print("forward 8: ", torch.isnan(hand_pose).float().sum().item())
        return global_orient, transl, hand_pose

    def forward_full_sequence(self, j3d, j2d=None, mask=None, debug=False, fixed_hand_pose=True, posebert_seq_len=128):
        
        l_global_orient, l_transl, l_hand_pose = [], [], []
        tdim = j3d.shape[1]
        for t in  range(tdim):
    
            # select a subseq
            if t - posebert_seq_len // 2 < 0:
                # beginning of the seq
                start_ = max([0, t - posebert_seq_len // 2])
                end_ = start_ + posebert_seq_len
            else:
                end_ = min([tdim, t + posebert_seq_len // 2])
                start_ = end_ - posebert_seq_len
            tt = np.clip(np.arange(start_, end_), 0, tdim - 1).tolist()
            t_of_interest = tt.index(t)

            # forward
            if j2d is None:
                j2d_tt = None
            else:
                j2d_tt = j2d[:,tt]
            if mask is None:
                mask_tt = None
            else:
                mask_tt = mask[:,tt]
            global_orient, transl, hand_pose = self.forward(j3d=j3d[:,tt], j2d=j2d_tt, mask=mask_tt, debug=debug, fixed_hand_pose=fixed_hand_pose)

            # append
            l_global_orient.append(global_orient[:,t_of_interest])
            l_transl.append(transl[:,t_of_interest])
            if fixed_hand_pose:
                l_hand_pose.append(hand_pose)
            else:
                l_hand_pose.append(hand_pose[:,t_of_interest])

        # Stack
        transl = torch.stack(l_transl, 1)
        global_orient = torch.stack(l_global_orient, 1)

        # Average hand pose if fixed
        hand_pose = torch.stack(l_hand_pose, 1)
        if fixed_hand_pose:
            w_i = torch.ones_like(hand_pose[:,:,:,0,0]) / hand_pose.shape[1]
            hand_pose = torch.sum(w_i[...,None,None] * hand_pose, dim=1)
            hand_pose = roma.special_procrustes(hand_pose)

        return global_orient, transl, hand_pose

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--dim", type=int, default=256)
        parser.add_argument("--pos_type", type=str, default='sine_frozen',
                            choices=['learned', 'sine_ft', 'sine_frozen', 'none'])
        parser.add_argument("--add_encoder", type=int, default=0, choices=[0, 1])
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--dim_head", type=int, default=64)
        parser.add_argument("--mlp_dim", type=int, default=512)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--emb_dropout", type=float, default=0.1)
        parser.add_argument("--share_regressor", type=int, default=1, choices=[0, 1])

        return parser


if __name__ == "__main__":
    from posebert.dataset import ManoDataset
    from posebert.renderer import PyTorch3DRenderer
    from pytorch3d.renderer import look_at_view_transform
    import smplx
    from posebert.constants import SMPLX_DIR
    from PIL import Image

    # MANO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm = smplx.create(SMPLX_DIR, 'mano', use_pca=False, is_rhand=True)
    faces = torch.from_numpy(np.array(bm.faces, dtype=np.int32)) # [1538,3]

    # Grab one seq
    x, y, z = 0. ,0. ,0.5
    dataset = ManoDataset(n=1, seq_len=32, 
                          range_x_init=[x,x+0.01], 
                          range_y_init=[y,y+0.01], 
                          range_z_init=[z,z+.01])
    j3d, global_orient, transl, hand_pose = dataset.__getitem__(0)
    print(transl[0])
    j3d = j3d.unsqueeze(0)

    # Instrinsics
    width, height = 1280, 720
    image_size=max([width, height])
    ratio = torch.Tensor([[image_size/width, image_size/height]]).float()
    f_x, f_y = 900., 900.
    c_x, c_y = 660., 380.
    j2d = perspective_projection(j3d.flatten(0,1), c_x, c_y, f_x, f_y).reshape(j3d.shape[0], j3d.shape[1], 21, 2)

    # Feed to the model
    model = PoseBERT()
    out = model.forward_full_sequence(j3d)
    out = model(j3d, debug=False)
    mask = torch.Tensor(np.random.choice([0,1],j3d.shape[0]*j3d.shape[1])).reshape(j3d.shape[0], j3d.shape[1]).bool()
    out = model(j3d, j2d, mask=mask, debug=True)
    for y in out:
        print(y.shape)

    # Visu output mesh
    renderer = PyTorch3DRenderer(image_size=image_size).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    focal_length = torch.Tensor([[2*f_x/image_size, 2*f_y/image_size]])
    principal_point = torch.Tensor([[c_x/width, c_y/height]])
    principal_point = (principal_point - 0.5) * 2 # values should be between -1 and 1. (0,0) is the image center
    principal_point /= ratio # solving principal point issue for rendering with non-square image
    # principal_point = torch.Tensor([[c_x, c_y]]) # in pixel space
    # principal_point = ((principal_point / image_size) - 0.5) * 2. # values should be between -1 and 1. (0,0) is the image center
    print("principal point: ", principal_point)
    
    if False:
        # using output of posebert
        global_orient, transl, hand_pose = out
    else:
        # using ground-truth
        global_orient = roma.rotvec_to_rotmat(global_orient).unsqueeze(0)
        transl = transl.unsqueeze(0)
        hand_pose = roma.rotvec_to_rotmat(hand_pose.reshape(1, 15, 3))

    verts = bm(global_orient=roma.rotmat_to_rotvec(global_orient[0,:1]),
               transl=transl[0], 
               hand_pose=roma.rotmat_to_rotvec(hand_pose).flatten(1)).vertices[0]
    print(verts[0])
    img = renderer.renderPerspective(vertices=[verts.to(device)], 
                                     faces=[faces.to(device)],
                                     rotation=rotation.to(device),
                                     camera_translation=cam.to(device),
                                     principal_point=principal_point.to(device),
                                     focal_length=focal_length,
                                     ).cpu().numpy()[0]
    delta = np.abs(width - height)//2
    if width > height:
        img = img[delta:-delta]
    else:
        img = img[:,delta:-delta]
    img_in = np.asarray(Image.open('img.jpg'))
    img = np.concatenate([img_in, img], 1)
    Image.fromarray(img).save('img_.jpg')

    print('done')