from tkinter import image_names
from posebert.skeleton import estimate_translation_np, estimate_translation, perspective_projection, update_mano_joints, visu_pose2d, get_mano_skeleton, convert_jts, get_bbox
import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.renderer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
# import smplx
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from PIL import Image
import ipdb
from posebert.constants import SMPLX_DIR
import os
from tqdm import tqdm
import math
import roma
from ipdb import set_trace as bb
from pytorch3d.renderer import (
    PointLights,
)
osp = os.path
MANO_DATA_DIR = "/scratch/1/user/aswamy/data/hand-obj/mano_fitting_data_texture"
RES_DIR = "/scratch/1/user/aswamy/data/hand-obj"
TMP_SAVE_DIR = "/scratch/1/user/aswamy/tmp/ho_renders"
MANO_RES_DIR = "/scratch/1/user/aswamy/data/hand-obj/mano_fitting_res_latest"

def ours2dope(**kwargs):
    return np.array([0, 1, 2, 3, 4, 5, 6, 11, 16, 7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20])

def get_cam_mesh(dist=0.05):
    """
    Args:
        - R: np.array (3,3)
        - t: np.array (3)
    Return:
        - vl: np.array (5,3)
        - fl: np.array (6,3)
    """
    vl = dist * np.array([[-0., -0., -1.],
                   [-1., -1.,  0.],
                   [ 1., -1.,  0.],
                   [ 1.,  1.,  0.],
                   [-1.,  1.,  0.]])

    fl = np.array([[0,2,1],[0,3,2],[0,4,3],[0,1,4],[1,2,3],[1,3,4]]) 
    return vl, fl

def load_mesh(path,filename):
    strFileName = os.path.join(path,filename)
    vertices = []
    texcoords = []
    normals = []
    faces = []
    texInd = []
    texture = None
    
    #first, read file and build arrays of vertices and faces
    for line in open(strFileName, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'mtllib':
            texture = loadMaterialTexture(path+"/"+values[1])
        if values[0] == 'v':
            vertices.append(list(map(float, values[1:4])))
            
        elif values[0] == 'vn':
            v = list(map(float, values[1:4]))
            normals.append(v)
        elif values[0] == 'vt':
            texcoords.append(list(map(float, values[1:3])))
            # continue
        elif values[0] in ('usemtl', 'usemat'):
            continue
        elif values[0] == 'f':
            for triNum in range(len(values)-3):  ## one line fan triangulation to triangulate polygons of size > 3
                v = values[1]
                w = v.split('/')
                faces.append(int(w[0])-1)           #vertex index (1st vertex)
                
                if(len(w)>1):
                    # print(len(w), w)
                    try:
                        texInd.append(int(w[1])-1)
                    except:
                        texInd.append(int(w[2])-1)

                for v in values[triNum+2:triNum+4]:
                    w = v.split('/')
                    faces.append(int(w[0])-1)           #vertex index (additional vertices)
                    if(len(w)>1):
                        try:
                            texInd.append(int(w[1])-1)
                        except:
                            texInd.append(int(w[2])-1)

    
    vertices = np.array(vertices).astype(np.float32)
    texcoords = np.array(texcoords).astype(np.float32)
    nb_vert = vertices.shape[0]

    # If 16 bits are not enough to write vertex indices, use 32 bits 
    if nb_vert<65536:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint16)
    else:
        faces = np.array(faces).reshape(len(faces) // 3, 3).astype(np.uint32)
    if len(texInd)>0:
        if texcoords.shape[0]<65536:
            texInd = np.array(texInd).reshape(faces.shape).astype(np.uint16)
        else:
            texInd = np.array(texInd).reshape(faces.shape).astype(np.uint32)

    return vertices, faces, texcoords, texInd, texture


def loadMaterialTexture(strFileName):

    txtFileName = ''
    for line in open(strFileName, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue

        if values[0] == "newmtl":
            continue
        elif values[0] == "Ka":
            # new_mat.ka = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "Kd":
            # new_mat.kd = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "Ks":
            # new_mat.ks = [float(values[1]),float(values[2]),float(values[3])]
            continue
        elif values[0] == "illum":
            # new_mat.illul = values[1]
            continue
        elif values[0] == "map_Ka":
            # txtFileName = getPath(strFileName)+"/"+values[1]
            txtFileName = os.path.dirname(strFileName)+"/"+values[1]
        elif values[0] == "map_Kd":
            # txtFileName = getPath(strFileName)+"/"+values[1]
            txtFileName = os.path.dirname(strFileName)+"/"+values[1]

    myIm = Image.open(txtFileName)
    npim = np.array(myIm)

    return npim


class PyTorch3DRenderer(torch.nn.Module):
    """
    Thin wrapper around pytorch3d threed.
    Only square renderings are supported.
    Remark: PyTorch3D uses a camera convention with z going out of the camera and x pointing left.
    """

    def __init__(self,
                 image_size,
                 background_color=(0, 0, 0),
                 convention='opencv',
                 blur_radius=1e-10,
                 faces_per_pixel=1,
                 bg_blending_radius=1,
                 max_faces_per_bin=200000,
                #  max_faces_per_bin=2000000,
                 # https://github.com/facebookresearch/pytorch3d/issues/448
                 # https://github.com/facebookresearch/pytorch3d/issues/348
                 # https://github.com/facebookresearch/pytorch3d/issues/316
                 ):
        super().__init__()
        self.image_size = image_size
        raster_settings_soft = pytorch3d.renderer.RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=max_faces_per_bin,
            )
        rasterizer = pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings_soft)

        materials = pytorch3d.renderer.materials.Materials(shininess=1.0)
        self.background_color = background_color
        blend_params = pytorch3d.renderer.BlendParams(background_color=background_color,
        # gamma=1e-4,
        # sigma=1e-4,
        # sigma=0.15,
        # gamma=0.14,
        )
        self.blend_params = blend_params
        print('blend_params', blend_params)

        # One need to attribute a camera to the shader, otherwise the method "to" does not work.
        dummy_cameras = pytorch3d.renderer.OrthographicCameras()
        shader = pytorch3d.renderer.SoftPhongShader(cameras=dummy_cameras,
                                                    materials=materials,
                                                    blend_params=blend_params)

        # Differentiable soft threed using per vertex RGB colors for texture
        self.renderer = pytorch3d.renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)
        # self.renderer = pytorch3d.renderer.MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

        self.convention = convention
        if convention == 'opencv':
            # Base camera rotation
            base_rotation = torch.as_tensor([[[-1, 0, 0],
                                              [0, -1, 0],
                                              [0, 0, 1]]], dtype=torch.float)
            self.register_buffer("base_rotation", base_rotation)
            self.register_buffer("base_rotation2d", base_rotation[:, 0:2, 0:2])

        # Light Color
        self.ambient_color = 0.5
        self.diffuse_color = 0.3
        self.specular_color = 0.2

        self.bg_blending_radius = bg_blending_radius
        if bg_blending_radius > 0:
            self.register_buffer("bg_blending_kernel",
                                 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (
                                         2 * bg_blending_radius + 1) ** 2)
            self.register_buffer("bg_blending_bias", -torch.ones(1))
        else:
            self.blending_kernel = None
            self.blending_bias = None

    def compose_foreground_on_background(self, fg_img, fg_masks, bg_img, alpha=1.):
        """
        Args:
            - fg_img: [B,3,W,H]
            - fg_mask: [B,W,H]
            - bg_img: [B,3,W,H]
        Copy-paste foreground on a background using the foreground masks.
        Done using a simple smoothing or by hard copy-pasting.
        """

        if self.bg_blending_radius > 0:
            # Simple smoothing of the mask
            fg_masks = torch.clamp_min(
                torch.nn.functional.conv2d(fg_masks.unsqueeze(1), weight=self.bg_blending_kernel, bias=self.bg_blending_bias,
                                           padding=self.bg_blending_radius) * fg_masks.unsqueeze(1), 0.0)[:,0].unsqueeze(-1)
        out = (alpha* fg_img + (1-alpha) * bg_img )* fg_masks+ bg_img * (1.0 - fg_masks)
        return out

    def to(self, device):
        # Transfer to device is a bit bugged in pytorch3d, one needs to do this manually
        self.renderer.shader.to(device)
        return super().to(device)

    def render(self, vertices, faces, cameras, color=None, faces_uvs=None, verts_uvs=None):
        """
        Args:
            - vertices: [B,N,V,3] OR list of shape [V,3]
            - faces: [B,F,3] OR list of shape [F,3]
            - maps: [B,N,W,H,3] in 0-1 range - if None the texture will be metallic
            - cameras: PerspectiveCamera object
            - color: [B,N,V,3]
        Return:
            - img: [B,W,H,C]
        """

        if isinstance(vertices, torch.Tensor):
            _, N, V, _ = vertices.size()
            list_faces = []
            list_vertices = []
            for i in range(N):
                list_faces.append(faces + V * i)
                list_vertices.append(vertices[:, i])
            faces = torch.cat(list_faces, 1)  # [B,N*F,3]
            vertices = torch.cat(list_vertices, 1)  # [B,N*V,3]

            # Metallic texture
            verts_rgb = torch.ones_like(vertices).reshape(-1, N, V, 3)  # [1,N,V,3]
            if color is not None:
                verts_rgb = color * verts_rgb
            verts_rgb = verts_rgb.flatten(1, 2)
            textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)
            
            # Create meshes
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
        else:
            # UV MAP
            if color is not None and len(color[0].shape) == 3:
                textures = pytorch3d.renderer.TexturesUV(maps=color, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            else:
                tex = [torch.ones_like(vertices[i]) if color is None else torch.ones_like(vertices[i]) * color[i] for i in range(len(vertices))]
                tex = torch.cat(tex)[None]
                textures = pytorch3d.renderer.Textures(verts_rgb=tex)
            
            verts = torch.cat(vertices)

            faces_up = []
            n = 0
            for i in range(len(faces)):
                faces_i = faces[i] + n
                faces_up.append(faces_i)
                n += vertices[i].shape[0]
            faces = torch.cat(faces_up)
            meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces], textures=textures)
        
        return self.add_light_and_render(meshes, cameras)

    def add_light_and_render(self, meshes, cameras):
        # Create light
        # lights = pytorch3d.renderer.DirectionalLights(
        #     ambient_color=((self.ambient_color, self.ambient_color, self.ambient_color),),
        #     diffuse_color=((self.diffuse_color, self.diffuse_color, self.diffuse_color),),
        #     specular_color=(
        #         (self.specular_color, self.specular_color, self.specular_color),),
        #     direction=((0, 0, -1.0),),
        #     # direction=((0, 1.0, 0.),),
        #     device=meshes.device)
        lights = PointLights(device=meshes.device, ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[0, 0, 0]], specular_color=[[0, 0, 0]], location=((0, 0, 0), ))
        images = self.renderer(meshes, cameras=cameras, lights=lights)
        

        rgb_images = images[..., :3]
        rgb_images = torch.clamp(rgb_images, 0., 1.)
        rgb_images = rgb_images * 255
        rgb_images = rgb_images.to(torch.uint8)

        # tmp = frags.bary_coords.squeeze(0).squeeze(2).reshape(-1, 3)

        # def ndc2cam(pts):
        #     fx = 1.4059; fy = 1.4063; px = 0.0215; py = 0.0060

        #     # X = fx * pts[0] / (pts[2] + 1e-6) + px
        #     tmp[:, 0] = fx * tmp[:, 0] / (tmp[:, 2]+ 1e-6) + px
        #     # Y = fy * pts[0] / (pts[2] + 1e-6) + py
        #     tmp[:, 1] = fy * tmp[:, 1] / (tmp[:, 2]+ 1e-6) + py
        #     # Z = 1 / (pts[2]+ 1e-6)
        #     tmp[:, 2] = 1 / (tmp[:, 2]+ 1e-6)
        # return 
            
        return rgb_images

    def renderPerspective(self, vertices, faces, camera_translation, principal_point=None, color=None, rotation=None,
                          focal_length=2 * 500. / 500., # 2 * focal_length / image_size
                          K=None,
                          faces_uvs=None,
                          verts_uvs=None,
                          render_fn='render',
                          # with cameras
                          textures=None,
                          ):
        """
        Args:
            - vertices: [B,V,3] or [B,N,V,3] where N is the number of persons OR list of tensor of shape [V,3]
            - faces: [B,13776,3] OR list of tensor of shape [V,3]
            - focal_length: float
            - principal_point: [B,2]
            - T: [B,3]
            - color: [B,N,3]
        Return:
            - img: [B,W,H,C] in range 0-1
        """

        device = vertices[0].device

        if principal_point is None:
            principal_point = torch.zeros_like(camera_translation[:, :2])

        if isinstance(vertices, torch.Tensor) and vertices.dim() == 3:
            vertices = vertices.unsqueeze(1)

        # Create cameras
        if rotation is None:
            R = self.base_rotation
        else:
            R = torch.bmm(self.base_rotation, rotation)
        camera_translation = torch.einsum('bik, bk -> bi', self.base_rotation.repeat(camera_translation.size(0), 1, 1),
                                          camera_translation)
        if self.convention == 'opencv':
            principal_point = -torch.as_tensor(principal_point)
        
        # fx = focal_length[:, 0]
        # fy = focal_length[:, 1]
        # px = principal_point[:, 0]
        # py = principal_point[:, 1]
        # K = torch.Tensor([[
        # [fx,   0,   px,   0],
        # [0,   fy,   py,   0],
        # [0,    0,    0,   1],
        # [0,    0,    1,   0],
        # ]])
        # cameras = pytorch3d.renderer.PerspectiveCameras(
        #                                                 R=R, T=camera_translation, device=device,
        #                                                 K=K,
        #                                                 # focal_length=focal_length, principal_point=principal_point,
        #                                                 )
        # ipdb.set_trace()

        # Screen space camera
        # image_size = ((128, 256),)    # (h, w)
        # fcl_screen = (76.8,)          # fcl_ndc * min(image_size) / 2
        # prp_screen = ((115.2, 48), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
        # cameras = pytorch3d.renderer.PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size)
        # ipdb.set_trace()

        cameras = pytorch3d.renderer.PerspectiveCameras(
                                                        R=R, T=camera_translation, device=device,
                                                        K=K,
                                                        focal_length=focal_length, principal_point=principal_point,
                                                        )

        if render_fn == 'render':
            rgb_images = self.render(vertices, faces, cameras, color, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
            # bb()
        else:
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
            # verts_rgb = torch.ones_like(vertices).reshape(-1, N, V, 3)  # [1,N,V,3]
            # if color is not None:
            #     verts_rgb = color * verts_rgb
            # verts_rgb = verts_rgb.flatten(1, 2)
            # textures = pytorch3d.renderer.Textures(verts_rgb=verts_rgb)
            return self.add_light_and_render(meshes, cameras)

        return rgb_images

@torch.no_grad()
def visu_all_objects(image_size = 2048, dist=.7, elev=45, azim=180, nrows=6, ncols=8, disp=0.25):
    # Loading object
    dirname = "/scratch/1/user/aswamy/data/hand-obj"
    seqnames = os.listdir(dirname)
    seqnames.sort()
    seqnames = [x for x in seqnames if os.path.exists(os.path.join(dirname, x+'.tar')) and x !=  '20220902111409' and 'DS_Store' not in x]

    ll = []
    for i, seqname in enumerate(seqnames):
        col_idx = i % ncols
        row_idx = i // ncols
        # row_idx = i // nrows
        print(i, col_idx, row_idx)
        ll.append(( col_idx, row_idx))
        if row_idx >= nrows:
            break
    # ipdb.set_trace()

    seqnames = seqnames[:nrows*ncols]

    l_verts = []
    l_faces = []
    l_color = []

    start_x = - ((ncols-1) * disp) / 2.
    start_y = 0
    # Renderer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = PyTorch3DRenderer(image_size=image_size,
                    blur_radius=0.00001,
                    faces_per_pixel=1,
                    bg_blending_radius=0,
                    #  background_color=(0, 0, 0),
                    background_color=(255, 255, 255),
                    max_faces_per_bin=2000000,
        ).to(device)
    for i, seqname in enumerate(seqnames):
        col_idx, row_idx = ll[i]
        # col_idx = i % ncols
        # row_idx = i // ncols

        seq_dir = f"/scratch/1/user/aswamy/data/hand-obj/{seqname}"
        
        obj_file = [x for x in os.listdir(f"{seq_dir}/gt_mesh") if x[0] != '.' and x[-4:] == '.obj'][0]
        vertices_, faces_, texcoords_, texInd_, texture_ = load_mesh(f"{seq_dir}/gt_mesh", obj_file)
        faces_uvs = torch.from_numpy(texInd_.astype(np.long)).long()
        verts_uvs = torch.from_numpy(texcoords_)
        faces = torch.from_numpy(np.array(faces_, dtype=np.int32))
        # verts = torch.from_numpy(vertices_).float() / 1000.
        verts = torch.from_numpy(vertices_).float()
        if (vertices_.max(0) - vertices_.min(0))[0] > 1:
            verts = verts / 1000.
        else:
            pass
        texture = (torch.from_numpy(texture_)/255.).float()

        from pytorch3d.renderer import TexturesVertex, TexturesUV
        from pytorch3d.structures import Meshes, packed_to_list
        from pytorch3d.io import load_objs_as_meshes
        # https://github.com/facebookresearch/pytorch3d/issues/854
        def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
            verts_colors_packed = torch.zeros_like(meshes.verts_packed())
            verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
            return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh())), verts_colors_packed

        obj_filename = os.path.join(f"{seq_dir}/gt_mesh", obj_file)
        mesh = load_objs_as_meshes([obj_filename])
        mesh_dash = load_objs_as_meshes([obj_filename])
        textures_vertex, verts_colors_packed = convert_to_textureVertex(mesh.textures, mesh)
        
        root_dir = f"{seq_dir}/icp_res/"
        rgb_dir = root_dir.replace('icp_res/', 'rgb')
        rgb_imgs = os.listdir(rgb_dir)
        rgb_imgs.sort()
        subdirs = os.listdir(root_dir)
        subdirs.sort()
        j3d_fns = os.listdir(os.path.join(seq_dir, 'jts3d'))
        j3d_fns.sort()
        # subdirs = subdirs[:128]
        intrinsics = np.array(
                [[899.783,   0.   , 653.768],
                [  0.   , 900.019, 362.143],
                [  0.   ,   0.   ,   1.   ]]
        )
        res = torch.Tensor([1280,720]).float()
        principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / res - 0.5) * 2).reshape(1,2)
        focal_length = torch.Tensor([[(2*intrinsics[0,0]/res[0]), (2*intrinsics[1,1]/res[0])]]) # TODO check documentation
        principal_point = None
        focal_length = torch.Tensor([[0.5, 0.5]]) # TODO check documentation

        fname = os.path.join(root_dir, subdirs[0], 'f_trans.txt')
        mat = torch.from_numpy(np.loadtxt(fname)).float()
        mat = torch.inverse(mat)
        verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
        verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
        verts_up = verts_up[:,:3,0]

        # update verts_up
        verts_up = verts_up - verts_up.mean(0, keepdims=True)

        # x_pos = start_x + disp * row_idx
        x_pos = start_x + disp * col_idx
        # y_pos = start_y + disp * col_idx
        if row_idx % 2 == 1:
            y_pos = start_y + disp * row_idx
        else:    
            y_pos = start_y  + disp * row_idx
        print(i, col_idx, row_idx)
        print(i, x_pos, y_pos)
        print(i, verts_up.min(0).values, verts_up.max(0).values)
        verts_up[:,0] = verts_up[:,0] + x_pos
        verts_up[:,2] = verts_up[:,2] + y_pos

        l_verts.append(verts_up.to(device))
        l_faces.append(faces.to(device))
        l_color.append(verts_colors_packed.to(device))

        # if i == 2:
        #     break

    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # print(rotation, cam)
    principal_point = None
    focal_length = torch.Tensor([[0.6, 0.6]]) # TODO check documentation
    img = renderer.renderPerspective(vertices=l_verts,
                                                faces=l_faces,
                                                rotation=rotation.to(device),
                                                camera_translation=cam.to(device),
                                                # principal_point=principal_point.to(device),
                                                focal_length=focal_length,
                                                color=l_color,
                                                ).cpu().numpy()[0]

        
    Image.fromarray(img).save('img.jpg')
    ipdb.set_trace()

@torch.no_grad()
def visu_cameras_anil(seqname='20220705173214', image_size = 1024, nb_timesteps=256, cam_size=0.01, dist=.7, elev=0, azim=180):
    # Loading object
    seq_dir = f"/scratch/1/user/aswamy/data/hand-obj/{seqname}"
    obj_file = [x for x in os.listdir(f"{seq_dir}/gt_mesh") if x[0] != '.' and x[-4:] == '.obj'][0]
    vertices_, faces_, texcoords_, texInd_, texture_ = load_mesh(f"{seq_dir}/gt_mesh", obj_file)
    faces_uvs = torch.from_numpy(texInd_.astype(np.long)).long()
    verts_uvs = torch.from_numpy(texcoords_)
    faces = torch.from_numpy(np.array(faces_, dtype=np.int32))
    verts = torch.from_numpy(vertices_).float()
    if (vertices_.max(0) - vertices_.min(0))[0] > 1:
        verts = verts / 1000.
    else:
        pass
    texture = (torch.from_numpy(texture_)/255.).float()

    from pytorch3d.renderer import TexturesVertex, TexturesUV
    from pytorch3d.structures import Meshes, packed_to_list
    from pytorch3d.io import load_objs_as_meshes
    # https://github.com/facebookresearch/pytorch3d/issues/854
    def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
        verts_colors_packed = torch.zeros_like(meshes.verts_packed())
        verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
        return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh())), verts_colors_packed

    obj_filename = os.path.join(f"{seq_dir}/gt_mesh", obj_file)
    mesh = load_objs_as_meshes([obj_filename])
    mesh_dash = load_objs_as_meshes([obj_filename])
    textures_vertex, verts_colors_packed = convert_to_textureVertex(mesh.textures, mesh)
    
    # Renderer
    tmp_dir = '/scratch/1/user/fbaradel/tmp/roar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                #  background_color=(0, 0, 0),
                 background_color=(255, 255, 255),
    ).to(device)
    root_dir = f"{seq_dir}/icp_res/"
    rgb_dir = root_dir.replace('icp_res/', 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(root_dir)
    subdirs.sort()
    j3d_fns = os.listdir(os.path.join(seq_dir, 'jts3d'))
    j3d_fns.sort()
    # subdirs = subdirs[:128]
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )
    res = torch.Tensor([1280,720]).float()
    ratio_render = res[0] / image_size
    delta = int(((res[0] - res[1])//2)/ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / res - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2*intrinsics[0,0]/res[0]), (2*intrinsics[1,1]/res[0])]]) # TODO check documentation

    vl, fl = get_cam_mesh(dist=cam_size)

    from webcolors import name_to_rgb
    colorname = 'blue'

    T0 = None

    from matplotlib import cm
    seq_len = len(subdirs)
    top = cm.get_cmap('Oranges_r', seq_len)
    bottom = cm.get_cmap('Blues', seq_len)
    colors = np.vstack((top(np.linspace(0, 1, seq_len//2)),
                        bottom(np.linspace(0, 1, seq_len//2))))
    colors = colors[:,:3]
    print("colors: ", colors.shape)
    # colors = np.stack([np.linspace(0, 1, len(subdirs)), 0. * np.linspace(0, 1, len(subdirs)), np.linspace(1, 0, len(subdirs))], 1)
    # colors = np.stack([np.linspace(0, 1, len(subdirs)), 0. * np.linspace(0, 1, len(subdirs)), 0. * np.linspace(1, 0, len(subdirs))], 1)


    l_verts, l_faces, l_color = [], [], []
    for t, subdir in enumerate(tqdm(subdirs)):
        if t % (len(subdirs) // nb_timesteps) == 0 or t == 0:
            # update the mesh according to the homogenous matrix
            fname = os.path.join(root_dir, subdir, 'f_trans.txt')
            mat = torch.from_numpy(np.loadtxt(fname)).float()
            mat = torch.inverse(mat)
            verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
            verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
            verts_up = verts_up[:,:3,0]

            # cam
            verts_cam = torch.from_numpy(vl).to(device).float() #+ torch.Tensor([[0., 0., 0.5]]).to(device)
            faces_cam = torch.from_numpy(fl.astype(np.int32)).long()
            try:
                color_ = colors[t] # np.array (3,)
                print('color', colors.shape, t)
            except:
                ipdb.set_trace()
            color = torch.from_numpy(color_).reshape(1,3).float()
            color = color.repeat(verts_cam.shape[0], 1)
            if t == 0:
                T0 = mat
                l_verts = [verts_up.to(device), verts_cam.to(device)]
                l_faces = [faces.to(device), faces_cam.to(device)]
                l_color = [verts_colors_packed.to(device), color.to(device)]
            else:
                T = mat @ torch.inverse(T0)
                verts_cam = torch.cat([verts_cam, torch.ones_like(verts_cam[...,-1:])], -1)
                verts_cam = T.reshape(-1,4,4).to(device) @ verts_cam.reshape(-1,4,1)
                verts_cam = verts_cam[:,:3,0]

                l_verts.append(verts_cam.to(device))
                l_faces.append(faces_cam.to(device))
                l_color.append(color.to(device))

    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # print(rotation, cam)
    img, dimg = renderer.renderPerspective(vertices=l_verts,
                                            faces=l_faces,
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            color=l_color,
                                            ).cpu().numpy()[0]
    # TODO automatic resaling of the image
    # ipdb.set_trace()
    # img > 0
    # img_segm = ((img != renderer.background_color).astype(np.float32).sum(-1, keepdims=False) > 0).astype(np.float32)
    # xx = img_segm.sum(1)

    # img_segm = (color.reshape(1,1,3) * img_segm).astype(np.uint8)

    Image.fromarray(img).save('img.jpg')

@torch.no_grad()
def teaser_anil(seqname='20220705173214', image_size = 1024, nb_timesteps=5, img_fn="img.jpg", obj_hat_fn='', rescal_factor=10.):
    # Loading object
    seq_dir = f"/scratch/1/user/aswamy/data/hand-obj/{seqname}"
    obj_file = [x for x in os.listdir(f"{seq_dir}/gt_mesh") if x[0] != '.' and x[-4:] == '.obj'][0]
    vertices_, faces_, texcoords_, texInd_, texture_ = load_mesh(f"{seq_dir}/gt_mesh", obj_file)
    faces_uvs = torch.from_numpy(texInd_.astype(np.long)).long()
    verts_uvs = torch.from_numpy(texcoords_)
    faces = torch.from_numpy(np.array(faces_, dtype=np.int32))
    verts = torch.from_numpy(vertices_).float()
    # verts = torch.from_numpy(vertices_).float() * 10. # TODO comment!
    if (vertices_.max(0) - vertices_.min(0))[0] > 1:
        print('rescale')
        verts = verts / 1000.
    else:
        pass
    texture = (torch.from_numpy(texture_)/255.).float()

    # Loading reconstructed object from anil
    from pytorch3d.io import load_ply, load_obj
    # verts, faces = load_ply('/scratch/1/user/aswamy/data/briac_baseline/20220909114359/res_colmap2/LOD0/mesh_0.ply')
    # obj = load_ply(obj_hat_fn.replace('.obj', '.ply'))
    # TODO TEASER!!! - center and rescale!!!!
    obj = load_obj(obj_hat_fn)
    vertices_hat = obj[0] #/10.
    vertices_hat = vertices_hat - vertices_hat[[0]]
    vertices_hat = vertices_hat / rescal_factor
    faces_hat = obj[1].verts_idx

    # vertices_hat, faces_hat, texcoords_hat, texInd_hat, texture_hat = load_mesh(f"", obj_hat_fn)
    # vertices_hat = torch.from_numpy(vertices_hat).float()
    # texture_hat = (torch.from_numpy(texture_hat)/255.).float()
    # ipdb.set_trace()
    # from pytorch3d.io import load_objs_as_meshes
    # mesh = load_objs_as_meshes([obj_hat_fn])

    # # renderer.add_light_and_render(mesh, cameras)
    # print('anil obj hat')
    # vertices_hat, faces__hat, texcoords_hat, texInd_hat, texture_hat = load_mesh(f"", obj_hat_fn)
    # ipdb.set_trace()
    # faces_uvs_hat = torch.from_numpy(texInd_hat.astype(np.long)).long()
    # verts_uvs_hat = torch.from_numpy(texcoords_hat)
    # faces_hat = torch.from_numpy(np.array(faces_hat, dtype=np.int32))
    
    # Renderer
    tmp_dir = '/scratch/1/user/fbaradel/tmp/roar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                #  background_color=(0, 0, 0),
                 background_color=(255, 255, 255),
    ).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    root_dir = f"{seq_dir}/icp_res/"
    rgb_dir = root_dir.replace('icp_res/', 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(root_dir)
    subdirs.sort()
    j3d_fns = os.listdir(os.path.join(seq_dir, 'jts3d'))
    j3d_fns.sort()
    # subdirs = subdirs[:128]
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )
    res = torch.Tensor([1280,720]).float()
    ratio_render = res[0] / image_size
    delta = int(((res[0] - res[1])//2)/ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / res - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2*intrinsics[0,0]/res[0]), (2*intrinsics[1,1]/res[0])]]) # TODO check documentation

    # RENDER OBJECT ONLY
    # if False:
    if True:
        rotation, cam = look_at_view_transform(dist=0.7, elev=0., azim=180)
        # verts = verts * 1000.
        # print(verts[0])
        img1 = renderer.renderPerspective(vertices=[verts.to(device)],
                                            faces=[faces.to(device)],
                                            color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            faces_uvs=[faces_uvs.to(device)],
                                            verts_uvs=[verts_uvs.to(device)],
                                            ).cpu().numpy()[0]
        # Image.fromarray(img).save('img.jpg')

        dist, elev, azim = 0.2, 10., 180
        dist, elev, azim = 0.4, 10., 180
        rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        
        import roma
        # rot = roma.rotvec_composition([torch.Tensor([[0., -0.9*np.pi, 0.]]), torch.Tensor([[-0.1*np.pi, 0., 0.]])])[0]
        rot = roma.rotvec_composition([torch.Tensor([[0., 0.7*np.pi, 0.]]), torch.Tensor([[0., 0., 0.]]), torch.Tensor([[0., 0., np.pi/8.]])])[0]
        rot = roma.rotvec_composition([torch.Tensor([[0., 0., 0.]]), torch.Tensor([[0., -0.0*np.pi, 0.]]), torch.Tensor([[0., 0., -np.pi/7.]])])[0]
        rot = roma.rotvec_composition([torch.Tensor([[0., 0., 0.]]), torch.Tensor([[0., -0.7*np.pi, 0.]]), torch.Tensor([[0., 0., 0.3*np.pi]])])[0]
        rot = roma.rotvec_to_rotmat(rot)
        vertices_hat = vertices_hat - vertices_hat.mean(0, keepdims=True)
        vertices_hat_ = (rot.unsqueeze(0) @ vertices_hat.clone().unsqueeze(-1))[:,:,0]
        img2 = renderer.renderPerspective(
            # vertices=[(vertices_hat_*10.).to(device)],
             vertices=[vertices_hat_.to(device)],
            #  vertices=[(vertices_hat_/10.).to(device)],
                                            faces=[faces_hat.to(device)],
                                            # color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            # faces_uvs=[faces_uvs.to(device)],
                                            # verts_uvs=[verts_uvs.to(device)],
                                            ).cpu().numpy()[0]
        Image.fromarray(np.concatenate([img1, img2], 1)).save('recon.jpg')
        ipdb.set_trace()

        dist, elev, azim = 0.00001, 0., 180
        rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)


    l_rgb, l_rendering, l_mesh, l_skeleton, l_mesh_hat = [], [], [], [], []
    mat0 = None
    w,h = 300, 300
    # ipdb.set_trace()
    # t_to_keep = [0, 200, 700]
    t_to_keep = [0, 300, 400]
    for t, subdir in enumerate(tqdm(subdirs)):
        # if True:
        if t in t_to_keep:
        # if t % (len(subdirs) // nb_timesteps) == 0:
            # print('render', t)
            # update the mesh according to the homogenous matrix
            fname = os.path.join(root_dir, subdir, 'f_trans.txt')
            mat = torch.from_numpy(np.loadtxt(fname)).float()

            # if t ==0:
            #     mat0 = mat.clone()
            # mat_up = mat0.numpy() @ np.linalg.inv(mat0.numpy())
            # print(mat_up)
            # mat_up = mat.clone().numpy()
            # mat_up = torch.inverse(torch.from_numpy(mat_up))
            # vertices_hat = torch.cat([vertices_hat, torch.ones_like(vertices_hat[...,-1:])], -1)
            # vertices_hat = mat_up.reshape(-1,4,4) @ vertices_hat.reshape(-1,4,1)
            # vertices_hat = vertices_hat[:,:3,0]

            # rendering the reconstructed mesh
            vertices_hat_up = vertices_hat + torch.Tensor([[0.,0.,0.7]])
            img_hat = renderer.renderPerspective(vertices=[vertices_hat_up.to(device)],
                                            faces=[faces_hat.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            ).cpu().numpy()[0][delta:-delta]

            mat = torch.inverse(mat)

            verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
            verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
            verts_up = verts_up[:,:3,0]

            # rendering the mesh with texture
            img_rendering = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            faces_uvs=[faces_uvs.to(device)],
                                            verts_uvs=[verts_uvs.to(device)],
                                            # render_fn='render_bis'
                                            ).cpu().numpy()[0][delta:-delta]

            # RGB image
            rgb = Image.open(os.path.join(rgb_dir, rgb_imgs[t]))
            width, height = rgb.size
            ratio = height / width
            new_width = img_rendering.shape[1]  
            new_height = int(ratio * new_width)
            rgb = np.asarray(rgb.resize((new_width, new_height)))

            # rendering with RGB as background
            if True:
                fg_masks = ((img_rendering != renderer.background_color).astype(np.float32).sum(-1, keepdims=True) > 0).astype(np.float32)
                fg_masks = torch.from_numpy(fg_masks.sum(-1) > 0).bool().unsqueeze(0) # [B,W,H]
                bg_img =  torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0) # [B,3,W,H]
                fg_img = torch.from_numpy(img_rendering).permute(2,0,1).unsqueeze(0)
                img_rendering = renderer.compose_foreground_on_background(fg_img, fg_masks.float(), bg_img, alpha=1.0)
                img_rendering = img_rendering[0].permute(1,2,0).numpy().astype(np.uint8)
                # Image.fromarray(img_rendering).save('img.jpg')

            # mesh_hat
            v2d = perspective_projection(vertices_hat_up.reshape(1, -1, 3), intrinsics[0,-1], intrinsics[1,-1], intrinsics[0,0], intrinsics[1,1], no_nan=True)
            v2d = v2d * new_width / width
            # bbox = get_bbox(v2d[0].numpy(), factor = 1.2, square=True).astype(np.int32)
            bbox = get_bbox(v2d[0].numpy(), factor = 1.0, square=True).astype(np.int32)
            x1, y1, x2, y2 = bbox
            try:
                l_mesh_hat.append(np.asarray(Image.fromarray(img_hat[y1:y2,x1:x2]).resize((w,h))))
            except:
                ipdb.set_trace()
            # Image.fromarray(l_mesh_hat[0]).save('img.jpg')
            # ipdb.set_trace()

            # skeleton
            v2d = perspective_projection(verts_up.reshape(1, -1, 3), intrinsics[0,-1], intrinsics[1,-1], intrinsics[0,0], intrinsics[1,1], no_nan=True)
            v2d = v2d * new_width / width
            # bbox = get_bbox(v2d[0].numpy(), factor = 1.2, square=True).astype(np.int32)
            bbox = get_bbox(v2d[0].numpy(), factor = 1.0, square=True).astype(np.int32)
            x1, y1, x2, y2 = bbox
            j3d = torch.from_numpy(np.loadtxt(os.path.join(seq_dir, 'jts3d', j3d_fns[t]))[ours2dope()].astype(np.float32)).reshape(1,-1,3)
            j3d = convert_jts(j3d, 'dope_hand', 'mano')
            j2d = perspective_projection(j3d, intrinsics[0,-1], intrinsics[1,-1], intrinsics[0,0], intrinsics[1,1], no_nan=True)
            j2d = j2d * new_width / width

            # rendering the mesh - no texture
            img_mesh = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            # color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            faces_uvs=[faces_uvs.to(device)],
                                            verts_uvs=[verts_uvs.to(device)],
                                            ).cpu().numpy()[0][delta:-delta]

            # print(verts_up.shape, verts_up[0])
            # print(intrinsics)
            # print(v2d)
            # Image.fromarray(img_mesh).save('img.jpg')
            # ipdb.set_trace()

            # rendering the segmentation on the rgb image with 2d skeleton
            color = np.asarray([137, 207, 240]).astype(np.float32)
            img_segm = ((img_mesh != renderer.background_color).astype(np.float32).sum(-1, keepdims=True) > 0).astype(np.float32)
            img_segm = (color.reshape(1,1,3) * img_segm).astype(np.uint8)
            fg_img = torch.from_numpy(img_segm).permute(2,0,1).unsqueeze(0) # [B,3,W,H]
            fg_masks = (fg_img.sum(1) > 0).bool() # [B,W,H]
            bg_img =  torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0) # [B,3,W,H]
            try:
                img_segm = renderer.compose_foreground_on_background(fg_img, fg_masks.float(), bg_img, alpha=0.7)
            except:
                ipdb.set_trace()
            img_segm = img_segm[0].permute(1,2,0).numpy().astype(np.uint8)
            img_segm = visu_pose2d(img_segm, j2d[0].numpy(), bones=get_mano_skeleton(), colors=['blue', 'green', 'cyan', 'magenta', 'yellow'], lw_line= 2, lw_dot = 1, color_dot='red')

            # Append
            try:
                l_rgb.append(np.asarray(Image.fromarray(rgb[y1:y2,x1:x2]).resize((w,h))))
                l_rendering.append(np.asarray(Image.fromarray(img_rendering[y1:y2,x1:x2]).resize((w,h))))
                l_mesh.append(np.asarray(Image.fromarray(img_mesh[y1:y2,x1:x2]).resize((w,h))))
                l_skeleton.append(np.asarray(Image.fromarray(img_segm[y1:y2,x1:x2]).resize((w,h))))
                # Image.fromarray(rgb).save('img.jpg')
                # Image.fromarray(img_rendering).save('img_rendering.jpg')
                # ipdb.set_trace()
            except:
                # Image.fromarray(rgb).save('img.jpg')
                ipdb.set_trace()
        else:
            # print('no render', t)
            pass

        if len(l_rgb) == nb_timesteps:
            break

    # Save into a single image
    rgb = np.concatenate(l_rgb, 1)
    rendering = np.concatenate(l_rendering, 1)
    mesh = np.concatenate(l_mesh, 1)
    skeleton = np.concatenate(l_skeleton, 1)
    mesh_hat = np.concatenate(l_mesh_hat, 1)
    img = np.concatenate([rgb, rendering, mesh, mesh_hat])
    # img = np.concatenate([rgb, rendering, mesh, skeleton])
    Image.fromarray(img).save(img_fn)

@torch.no_grad()
def show_video(seqname='20220705173214', video_dir="/scratch/1/user/fbaradel/showme/gt_mesh", image_size = 1280, t_start=0, t_end=1000000):
    # Loading object
    seq_dir = f"/scratch/1/user/aswamy/data/hand-obj/{seqname}"
    obj_file = [x for x in os.listdir(f"{seq_dir}/gt_mesh") if x[0] != '.' and x[-4:] == '.obj'][0]
    vertices_, faces_, texcoords_, texInd_, texture_ = load_mesh(f"{seq_dir}/gt_mesh", obj_file)
    faces_uvs = torch.from_numpy(texInd_.astype(np.long)).long()
    verts_uvs = torch.from_numpy(texcoords_)
    faces = torch.from_numpy(np.array(faces_, dtype=np.int32))
    verts = torch.from_numpy(vertices_).float()
    # verts = torch.from_numpy(vertices_).float() * 10. # TODO comment!
    if (vertices_.max(0) - vertices_.min(0))[0] > 1:
        print('rescale')
        verts = verts / 1000.
    else:
        pass

    # @fbaradel in case you need to load your obj file please do that
    if False:
        from pytorch3d.io import load_obj
        obj_fn = 'my_obj_file.obj'
        obj = load_obj(obj_fn)
        verts = obj[0] #/10.
        verts = verts - verts[[0]]
        rescal_factor = 1.
        verts = verts / rescal_factor
        faces = obj[1].verts_idx
    
    # Renderer
    os.makedirs(video_dir, exist_ok=True)
    tmp_dir = f"/scratch/1/user/aswamy/tmp/ho_renders/{seqname}"
    os.makedirs(tmp_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                #  background_color=(0, 0, 0),
                 background_color=(255, 255, 255),
    ).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # bb()
    root_dir = f"{seq_dir}/icp_res/"
    rgb_dir = root_dir.replace('icp_res/', 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(root_dir)
    subdirs.sort()
    j3d_fns = os.listdir(os.path.join(seq_dir, 'jts3d'))
    j3d_fns.sort()
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )
    # bb()
    res = torch.Tensor([1280, 720]).float()
    ratio_render = res[0] / image_size
    delta = int(((res[0] - res[1])//2)/ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / res - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2*intrinsics[0,0]/res[0]), (2*intrinsics[1,1]/res[0])]]) # TODO check documentation

    subdirs = subdirs[t_start:t_end]
    for t, subdir in enumerate(tqdm(subdirs)):
        # update the mesh according to the homogenous matrix
        fname = os.path.join(root_dir, subdir, 'f_trans.txt')
        mat = torch.from_numpy(np.loadtxt(fname)).float()
        mat = torch.inverse(mat)
        verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
        verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
        verts_up = verts_up[:,:3,0]

        # rendering the mesh - no texture
        img_mesh, dimg_mesh = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            # color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            faces_uvs=[faces_uvs.to(device)],
                                            verts_uvs=[verts_uvs.to(device)],
                                            )
        # ipdb.set_trace()
        img_mesh = img_mesh.cpu().numpy()[0][delta:-delta]
        dimg_mesh = dimg_mesh.cpu().numpy()[delta:-delta]
        # dimg_mesh = ((dimg_mesh - dimg_mesh.min()) / (dimg_mesh.max() - dimg_mesh.min())*255).astype(np.uint8)
        dimg_mesh = (dimg_mesh * 255).astype(np.uint8)
        Image.fromarray(img_mesh).save(os.path.join(tmp_dir, f"{t:06d}.jpg"))
        Image.fromarray(dimg_mesh[:, :, 0]).save(os.path.join(tmp_dir, f"{t:06d}_depth.jpg"))      
        
    # Video
    video_fn = os.path.join(video_dir, f"{seqname}.mp4")
    print(f"creating rgb video: {video_fn}")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/??????.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_fn} -y"
    os.system(cmd)
    # os.system(f"rm {tmp_dir}/*.jpg")

    video_depth_fn = os.path.join(video_dir, f"{seqname}_depth.mp4")
    print(f"creating rgb video: {video_depth_fn}")
    cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/??????_depth.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_fn} -y"
    os.system(cmd)
    # os.system(f"rm {tmp_dir}/*.jpg")

@torch.no_grad()
def render_mesh_to_all_frames(mesh_pth, save_base_dir, sqn_res_dir, image_size = 1280, t_start=0, t_end=1000000):
    # bb()
    vertices_, faces_, texcoords_, texInd_, texture_ = load_mesh(osp.dirname(mesh_pth), osp.basename(mesh_pth))
    # faces_uvs = torch.from_numpy(texInd_.astype(np.long)).long()
    # verts_uvs = torch.from_numpy(texcoords_)
    faces = torch.from_numpy(np.array(faces_, dtype=np.int32))
    verts = torch.from_numpy(vertices_).float()
    # verts = torch.from_numpy(vertices_).float() * 10. # TODO comment!
    if (vertices_.max(0) - vertices_.min(0))[0] > 1:
        print('rescale')
        verts = verts / 1000.
    else:
        pass

    os.makedirs(save_base_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                #  background_color=(0, 0, 0),
                 background_color=(255, 255, 255),
    ).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    pose_dir = osp.join(sqn_res_dir, 'icp_res')
    rgb_dir = osp.join(sqn_res_dir, 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(pose_dir)
    subdirs.sort()
    
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )

    resln = torch.Tensor([1280, 720]).float()
    ratio_render = resln[0] / image_size
    delta = int(((resln[0] - resln[1]) // 2) / ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / resln - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2 * intrinsics[0,0] / resln[0]), (2 * intrinsics[1,1] / resln[0])]]) # TODO check documentation
    # bb()
    subdirs = subdirs[t_start:t_end]

    for t, subdir in enumerate(tqdm(subdirs)):
        # update the mesh according to the homogenous matrix
        fname = os.path.join(pose_dir, subdir, 'f_trans.txt')
        mat = torch.from_numpy(np.loadtxt(fname)).float()
        mat = torch.inverse(mat)
        verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
        verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
        verts_up = verts_up[:,:3,0]
        # bb()
        # rendering the mesh - no texture
        
        img_mesh, dimg_mesh = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            # faces_uvs=[faces_uvs.to(device)],
                                            # verts_uvs=[verts_uvs.to(device)],
                                            )
        
        img_mesh = img_mesh.cpu().numpy()[0][delta : -delta]
        dimg_mesh = dimg_mesh.cpu().numpy()[delta : -delta]
        # dimg_mesh = ((dimg_mesh - dimg_mesh.min()) / (dimg_mesh.max() - dimg_mesh.min())*255).astype(np.uint8)
        dimg_mesh = (dimg_mesh * 255).astype(np.uint8)

        fn_img = os.path.join(save_base_dir, 'images', f"{t:06d}.jpg")
        os.makedirs(osp.dirname(fn_img), exist_ok=True)
        Image.fromarray(img_mesh).save(fn_img)
        fn_dimg = os.path.join(save_base_dir, 'depth', f"{t:06d}_depth.jpg")
        os.makedirs(osp.dirname(fn_dimg), exist_ok=True)
        Image.fromarray(dimg_mesh[:, :, 0]).save(fn_dimg)   

@torch.no_grad()
def render_comb_mesh(comb_mesh, save_base_dir, sqn_res_dir, image_size = 1280, t_start=0, t_end=1000000):
    verts = comb_mesh.verts_list()[0]
    faces = comb_mesh.faces_list()[0]

    # rescale to meters
    if (verts.max() - verts.min()) > 1:
        print('rescale')
        verts = verts / 1000.
    else:
        pass

    os.makedirs(save_base_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    renderer = PyTorch3DRenderer(image_size=image_size,
                 blur_radius=0.00001,
                 faces_per_pixel=1,
                 bg_blending_radius=0,
                 background_color=(0, 0, 0),
                #  background_color=(255, 255, 255),
    ).to(device)
    dist, elev, azim = 0.00001, 0., 180
    rotation, cam = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    pose_dir = osp.join(sqn_res_dir, 'icp_res')
    rgb_dir = osp.join(sqn_res_dir, 'rgb')
    rgb_imgs = os.listdir(rgb_dir)
    rgb_imgs.sort()
    subdirs = os.listdir(pose_dir)
    subdirs.sort()
    
    intrinsics = np.array(
            [[899.783,   0.   , 653.768],
            [  0.   , 900.019, 362.143],
            [  0.   ,   0.   ,   1.   ]]
    )

    resln = torch.Tensor([1280, 720]).float()
    ratio_render = resln[0] / image_size
    delta = int(((resln[0] - resln[1]) // 2) / ratio_render)
    principal_point = ((torch.from_numpy(intrinsics[:2,-1]) / resln - 0.5) * 2).reshape(1,2)
    focal_length = torch.Tensor([[(2 * intrinsics[0,0] / resln[0]), (2 * intrinsics[1,1] / resln[0])]]) # TODO check documentation
    # bb()
    subdirs = subdirs[t_start:t_end]

    for t, subdir in enumerate(tqdm(subdirs)):
        # update the mesh according to the homogenous matrix
        fname = os.path.join(pose_dir, subdir, 'f_trans.txt')
        mat = torch.from_numpy(np.loadtxt(fname)).float()
        mat = torch.inverse(mat)
        verts_up = torch.cat([verts, torch.ones_like(verts[...,-1:])], -1)
        verts_up = mat.reshape(-1,4,4) @ verts_up.reshape(-1,4,1)
        verts_up = verts_up[:,:3,0]
        # bb()
        # rendering the mesh - no texture
        texture = comb_mesh.textures.verts_features_list()[0]
        img_mesh = renderer.renderPerspective(vertices=[verts_up.to(device)],
                                            faces=[faces.to(device)],
                                            color=[texture.to(device)],
                                            rotation=rotation.to(device),
                                            camera_translation=cam.to(device),
                                            principal_point=principal_point.to(device),
                                            focal_length=focal_length,
                                            # faces_uvs=[faces_uvs.to(device)],
                                            # verts_uvs=[verts_uvs.to(device)],
                                            )
        
        img_mesh = img_mesh.cpu().numpy()[0][delta : -delta]
        # dimg_mesh = dimg_mesh.cpu().numpy()[delta : -delta]
        # # dimg_mesh = ((dimg_mesh - dimg_mesh.min()) / (dimg_mesh.max() - dimg_mesh.min())*255).astype(np.uint8)
        # dimg_mesh = (dimg_mesh * 255).astype(np.uint8)

        fn_img = os.path.join(save_base_dir, 'images', f"{subdir}.png")
        os.makedirs(osp.dirname(fn_img), exist_ok=True)
        Image.fromarray(img_mesh).save(fn_img)
        # fn_dimg = os.path.join(save_base_dir, 'depth', f"{subdir}_depth.png")
        # os.makedirs(osp.dirname(fn_dimg), exist_ok=True)
        # Image.fromarray(dimg_mesh[:, :, 0]).save(fn_dimg)   




@torch.no_grad()
def render_ho_frames(sqn='20220705173214', create_vid=False):
    sqn_res_dir = osp.join(RES_DIR, sqn)

    # get mesh paths    
    # obj_mesh_pth = [osp.join(MANO_DATA_DIR, sqn, 'obj_mesh', x)  for x in os.listdir(osp.join(MANO_DATA_DIR, sqn, 'obj_mesh')) if x[0] != '.' and x[-4:] == '.obj'][0]
    # hand_mesh_pth = [osp.join(MANO_DATA_DIR, sqn, 'ho_mesh', x)  for x in os.listdir(osp.join(MANO_DATA_DIR, sqn, 'ho_mesh')) if x[0] != '.' and x[-4:] == '.obj'][0]
    obj_mesh_pth = osp.join(MANO_RES_DIR, sqn, 'object.obj')
    hand_mesh_pth = osp.join(MANO_RES_DIR, sqn, 'mano.obj')

    # obj_renders_save_dir = osp.join(TMP_SAVE_DIR, sqn, 'obj')
    # hand_renders_save_dir = osp.join(TMP_SAVE_DIR, sqn, 'hand')
    renders_save_dir = osp.join(TMP_SAVE_DIR, sqn, 'ho')
    
    obj_vertices_, obj_faces_, obj_texcoords_, obj_texInd_, obj_texture_ = load_mesh(osp.dirname(obj_mesh_pth), osp.basename(obj_mesh_pth))
    obj_faces = torch.from_numpy(np.array(obj_faces_, dtype=np.int32))
    obj_verts = torch.from_numpy(obj_vertices_).float()
    obj_texture = pytorch3d.renderer.TexturesVertex(verts_features=torch.repeat_interleave(torch.tensor([1, 0, 0]).unsqueeze(0), obj_vertices_.shape[0], 0).unsqueeze(0))
    obj_mesh = pytorch3d.structures.Meshes(verts=[obj_verts], faces=[obj_faces], textures=obj_texture)
    
    hand_vertices_, hand_faces_, hand_texcoords_, hand_texInd_, hand_texture_ = load_mesh(osp.dirname(hand_mesh_pth), osp.basename(hand_mesh_pth))
    hand_faces = torch.from_numpy(np.array(hand_faces_, dtype=np.int32))
    hand_verts = torch.from_numpy(hand_vertices_).float()
    hand_texture = pytorch3d.renderer.TexturesVertex(verts_features=torch.repeat_interleave(torch.tensor([0, 0, 1]).unsqueeze(0), hand_vertices_.shape[0], 0).unsqueeze(0))
    hand_mesh = pytorch3d.structures.Meshes(verts=[hand_verts], faces=[hand_faces], textures=hand_texture)
    
    comb_mesh = pytorch3d.structures.join_meshes_as_scene([hand_mesh, obj_mesh])
    # bb()
    render_comb_mesh(comb_mesh, renders_save_dir, sqn_res_dir, image_size = 1280, t_start=0, t_end=1000000)
    # render
    # render_mesh_to_all_frames(obj_mesh_pth, obj_renders_save_dir, sqn_res_dir, image_size = 1280, t_start=0, t_end=1000000)
    # render_mesh_to_all_frames(hand_mesh_pth, hand_renders_save_dir, sqn_res_dir, image_size = 1280, t_start=0, t_end=1000000)

    # create video
    if create_vid:
        video_fn = os.path.join(TMP_SAVE_DIR, f"{sqn}.mp4")
        print(f"creating rgb video: {video_fn}")
        cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/??????.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_fn} -y"
        os.system(cmd)
        # os.system(f"rm {tmp_dir}/*.jpg")

        video_depth_fn = os.path.join(TMP_SAVE_DIR, f"{sqn}_depth.mp4")
        print(f"creating rgb video: {video_depth_fn}")
        cmd = f"ffmpeg -hide_banner -loglevel error -framerate 10 -pattern_type glob -i '{tmp_dir}/??????_depth.jpg' -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_fn} -y"
        os.system(cmd)
        # os.system(f"rm {tmp_dir}/*.jpg")

    print('Done!')    


if __name__ == "__main__":
    # import sys
    # exec(sys.argv[1])
    # os._exit(0)

    import argparse
    parser = argparse.ArgumentParser(description='rendering')
    parser.add_argument("--seqname", type=str, default='20220705173214')
    parser.add_argument("--obj_fn", type=str, default='todo')
    parser.add_argument("--visu_obj", type=int, default=1, choices=[0,1])
    args = parser.parse_args()

    render_ho_frames(sqn=args.seqname)
    # show_video(seqname=args.seqname)
    os._exit(0)
