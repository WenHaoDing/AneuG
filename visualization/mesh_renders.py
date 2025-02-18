import torch
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex
)
import pytorch3d
import torchvision.utils as vutils
from pytorch3d.transforms import RotateAxisAngle
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trimesh
import pyvista as pv
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import random


class MeshRender(object):
    def __init__(self, aneurysm_verts_idx=None,
                 background_color=None, aneurysm_color=None, vessel_color=None, device=torch.device('cuda:0'),
                 out_dir=None):
        
        self.device = device
        self.aneurysm_verts_idx = aneurysm_verts_idx
        # self.background_color = torch.Tensor([[30, 206, 255]]) / 255 if background_color is None else background_color
        # self.aneurysm_color = (torch.Tensor([[12, 56, 100]]) / 255).to(device) if aneurysm_color is None else aneurysm_color
        self.aneurysm_color = (torch.Tensor([[12, 56, 100]]) / 255).to(device) if aneurysm_color is None else aneurysm_color
        self.vessel_color = (torch.Tensor([[255, 255, 255]]) / 255).to(device) if vessel_color is None else vessel_color
        self.out_dir = out_dir
    
    def render_rotational_gif(self, meshes: Meshes, dual_texture=False, 
                              n_row=3, fps=24,
                              out_dir=None, label=None):
        """ render the mesh in a rotational gif """
        mesh_render = self.mesh_render
        B = len(meshes.verts_list())
        meshes = Meshes(verts=meshes.verts_padded().detach().requires_grad_(False), 
                        faces=meshes.faces_padded().detach().requires_grad_(False)).to(self.device)
        meshes = self.assign_texture(meshes, dual_texture=dual_texture)
        gif_name = label + '_dual.gif' if dual_texture else label + '.gif'
        out_dir = self.out_dir if out_dir is None else out_dir
        out_name = f'{out_dir}/{gif_name}'
        os.makedirs(out_dir, exist_ok=True)
        save_mesh_as_gif(mesh_render, meshes, nrow=n_row, fps=fps, out_name=out_name)

    def init_mesh_render(self, image_size=1024, dist=1.5, elev=90, azim=90):
        """ initialize the mesh renderer """
        self.mesh_render = init_mesh_renderer(image_size=image_size, dist=dist, elev=elev, azim=azim, device=self.device)
    
    def assign_texture(self, meshes: Meshes, dual_texture=False):
        B = len(meshes.verts_list())
        verts = meshes.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
            verts_rgb_i = torch.ones_like(verts[i]).to(self.device) + self.vessel_color
            if dual_texture:
                verts_rgb_i[self.aneurysm_verts_idx, :] = 0 * verts_rgb_i[self.aneurysm_verts_idx, :] + self.aneurysm_color
            else:
                verts_rgb_i = torch.ones_like(verts[i]).to(self.device) + self.aneurysm_color
            verts_rgb_list.append(verts_rgb_i.detach().to(self.device))
        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list).detach()
        meshes.textures = texture
        return meshes

    def save_objs(self, Meshes_: Meshes, label, save_dir=None, add_random_string=False):
        if save_dir is None:
            save_dir = f'{self.out_dir}/meshes/{label}'
        else:
            pass
        os.makedirs(save_dir, exist_ok=True)
        B = len(Meshes_.verts_list())
        # defalut transformation
        R = torch.Tensor([[- 0.5 * np.pi, 0, 0]])
        R_matrix = axis_angle_to_matrix(R).to(Meshes_.device)
        Meshes_.update_padded((Meshes_.verts_padded() @ R_matrix.transpose(-1, -2)).float())

        for i in range(B):
            verts = Meshes_.verts_list()[i].detach()
            faces = Meshes_.faces_list()[i].detach()
            usable_verts = verts[faces.flatten().unique(), :]
            # normals = Meshes_.verts_normals_list()[i].detach()
            verts -= usable_verts.mean(dim=0, keepdim=True)
            verts[:, -2] -= verts[:, -2].min()

            random_string = generate_random_string()
            if add_random_string:
                save_path = f'{save_dir}/{i}_{random_string}.obj'
            else:
                save_path = f'{save_dir}/{i}.obj'
            # mesh_to_show = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
            # pv.set_jupyter_backend('html')
            # pv.start_xvfb()
            # p = pv.Plotter()
            # p.add_mesh(mesh_to_show, color='blue', opacity=0.5)
            # p.add_points(np.array([0, 0, 0]), color='red')
            # p.show()
            # break
            pytorch3d.io.save_obj(save_path, verts=verts, faces=faces)
        print(f'Saved {B} meshes to {save_dir}')



def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, fps=24, out_name='1.gif'):
    """ save batch of mesh into gif """
    # autosdf: 36 frames, 0.08 duration
    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)    
    # rotate
    rot_comb = rotate_mesh_360(mesh_renderer, mesh) # save the first one
    
    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])
            
        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)
            
        rot_comb_img.append(img_grid_i)
    
    frames = [Image.fromarray(rot) for rot in rot_comb_img]
    
    # Save as an animated GIF
    duration = 1 / fps
    frames[0].save(out_name, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0, transparency=0, disposal=2)


def render_mesh(renderer, mesh, color=None, norm=True):
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
        # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    materials = Materials(
    device=mesh.device,
    specular_color=[[12/255, 56/255, 100/255]],
    shininess=0.0)
    images = renderer(mesh)
    return images.permute(0, 3, 1, 2)

def init_mesh_renderer(image_size=1024, dist=1.5, elev=90, azim=90, camera='0', device='cuda:0'):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 

    if camera == '0':
        # for vox orientation
        # dist, elev, azim = 1.7, 20, 20 # shapenet
        # dist, elev, azim = 3.5, 90, 90 # front view

        # dist, elev, azim = 3.5, 0, 135 # front view
        camera_cls = FoVPerspectiveCameras
    else:
        # dist, elev, azim = 5, 45, 135 # shapenet
        camera_cls = FoVOrthographicCameras
    
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = camera_cls(device=device, R=R, T=T)
    # print(f'[*] renderer device: {device}')

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=((100, 100, 0),))
    lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))
    lights = DirectionalLights(device=device, direction=((1, 1, 0),), ambient_color=((12/255, 56/255, 100/255),))

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device, 
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    # renderer = MeshRenderer(
    #         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    #         shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    #     )
    renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )
    return renderer

def rotate_mesh(mesh, axis='Y', angle=5, device='cuda'):
    rot_func = RotateAxisAngle(angle, axis, device=device)

    verts = mesh.verts_list()
    faces = mesh.faces_list()
    textures = mesh.textures
    
    B = len(verts)

    rot_verts = []
    for i in range(B):
        v = rot_func.transform_points(verts[i])
        rot_verts.append(v)
    new_mesh = Meshes(verts=rot_verts, faces=faces, textures=textures).to(device)
    return new_mesh

def rotate_mesh_360(mesh_renderer, mesh, n_frames=72):
    device = mesh.device
    cur_mesh = mesh

    B = len(mesh.verts_list())
    ret = [ [] for i in range(B)]

    angle = (360 // n_frames)

    # for i in range(36):
    for i in range(n_frames):
        cur_mesh = rotate_mesh(cur_mesh, angle=angle, device=device)
        img = render_mesh(mesh_renderer, cur_mesh, norm=False) # b c h w
        img = img.permute(0, 2, 3, 1) # b h w c
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        for j in range(B):
            ret[j].append(img[j])

    return ret

def generate_random_string(length=4):
    return ''.join(random.choice('ABCDE12345') for _ in range(length))