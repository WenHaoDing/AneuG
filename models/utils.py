import matplotlib.pyplot as plt
import torch
import os.path as osp
from models.ghd_reconstruct import GHD_Reconstruct
import wandb
from typing import List
from torch_geometric.data import Data, Batch
import random
import string
import math
from models.mesh_plugins import MeshPlugins
import numpy as np
from pytorch3d.structures import Meshes
from random import randint


def save_models(generator, optimizer_G, epoch, log_path, cond_keys=None, cond_loss_style=None, apply_norm=None):
    save_dict = {
        'generator': generator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'epoch': epoch
    }
    if cond_keys is not None:
        save_dict['cond_keys'] = cond_keys
    if cond_loss_style is not None:
        save_dict['cond_loss_style'] = cond_loss_style
    if apply_norm is not None:
        save_dict['apply_norm'] = apply_norm
    torch.save(save_dict, osp.join(log_path, 'models_epoch_{}.pth'.format(epoch)))

def load_models(generator, optimizer_G, log_path, epoch):
    checkpoint = torch.load(osp.join(log_path, 'models_epoch_{}.pth'.format(epoch)))
    generator.load_state_dict(checkpoint['generator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    epoch = checkpoint['epoch']
    print(f"Learning rate in optimizer_G: {optimizer_G.param_groups[0]['lr']}")
    return generator, optimizer_G, epoch

def plot_meshes(ghd_reconstruct: GHD_Reconstruct, *data):
    """
    return fig obj of two meshes
    data.pos contains the vertices
    assuming data only contains 1 pair of data
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    triangles = ghd_reconstruct.canonical_Meshes.faces_packed().detach().cpu().numpy()
    for data_, color in zip(data, ['yellow', 'dodgerblue', 'red']):
        verts = data_.pos.detach().cpu().numpy()
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles, edgecolor=[[0, 0, 0]], linewidth=0.01, alpha=0.2)
    ax.view_init(elev=45, azim=0)
    return fig

def plot_wandb(ghd_reconstruct: GHD_Reconstruct, dataset, generator, latent_dim, epoch, device, use_norm=True, withscale=False):
    mean, std = dataset.get_mean_std()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    ghd_real = next(iter(dataloader)).to(device)  # [2, 3*M]
    if withscale:
        ghd_real, scale_real = ghd_real[:, :-1], ghd_real[:, -1:]
    
    if withscale:
        ghd_recon, _, _, _ = generator(ghd_real, scale_real)  # [2, 3*M]
    else:
        ghd_recon, _, _ = generator(ghd_real)
    ghd_real, ghd_fake, ghd_recon = ghd_real[:1], ghd_fake[:1], ghd_recon[:1]
    z = torch.randn(16, latent_dim).to(device)
    ghd_fake = generator.decode(z, True).detach()  # [2, 3*M]
    # forward
    data_real = ghd_reconstruct.forward(ghd_real, mean, std, use_norm)
    data_fake = ghd_reconstruct.forward(ghd_fake, mean, std, use_norm)
    data_recon = ghd_reconstruct.forward(ghd_recon, mean, std, use_norm)
    # plot 
    fig_recon = plot_meshes(ghd_reconstruct, data_real, data_recon)
    fig_fake = plot_meshes(ghd_reconstruct, data_fake)
    wandb.log({"recon": [wandb.Image(fig_recon)], "uncon-gen": [wandb.Image(fig_fake)]}, step=epoch)
    plt.close(fig_recon)
    plt.close(fig_fake)

def plot_wandb_conditional(ghd_reconstruct: GHD_Reconstruct, dataset, generator, latent_dim, epoch, device, use_norm=True, dmm_calculator=None, condist_approx=None, cond_loss_style=None):
    mean, std = dataset.get_mean_std()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    ghd_real = next(iter(dataloader)).to(device)  # [2, 3*M]

    data_real = ghd_reconstruct.forward(ghd_real, mean, std, use_norm)
    cond_real = dmm_calculator.forward(data_real).to(device)
    cond_min, cond_max = cond_real.min(dim=0, keepdim=True)[0], cond_real.max(dim=0, keepdim=True)[0]
    z = torch.randn(16, latent_dim).to(device)
    B = z.shape[0]
    z = torch.randn(B, z.shape[1]).to(device)
    if not cond_loss_style == 'dist_approx':
        cond_fake = torch.rand(B, cond_real.shape[1]).to(device) * (cond_max - cond_min) + cond_min
    else:
        cond_fake = condist_approx.get_batch(normalize=True)[:B, :]

    ghd_fake = generator.decode(z, cond_fake).detach()  # [2, 3*M]
    ghd_recon, _, _ = generator(ghd_real, cond_real)  # [2, 3*M]
    ghd_real, ghd_fake, ghd_recon = ghd_real[:1], ghd_fake[:1], ghd_recon[:1]

    # forward
    data_fake = ghd_reconstruct.forward(ghd_fake, mean, std, use_norm)
    data_recon = ghd_reconstruct.forward(ghd_recon, mean, std, use_norm)
    # plot 
    fig_recon = plot_meshes(ghd_reconstruct, data_real, data_recon)
    fig_fake = plot_meshes(ghd_reconstruct, data_fake)
    wandb.log({"recon": [wandb.Image(fig_recon)], "uncon-gen": [wandb.Image(fig_fake)]},
              step=epoch)
    plt.close(fig_recon)
    plt.close(fig_fake)

def plot_wandb_centerline(ghd_reconstruct: GHD_Reconstruct, mesh_plugin: MeshPlugins,
                          ghd, cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd,
                          cpcd_glo_true, cpcd_glo_pred,
                          epoch,
                          scale=None):
    """
    ghd needs to be already de-normalized
    """
    Meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, scale=scale)
    verts = Meshes.verts_list()[0].detach().cpu().numpy()
    # triangles = mesh_plugin.return_decapped_faces().detach().cpu().numpy()
    triangles = mesh_plugin.trimmed_faces.detach().cpu().numpy()
    # we take the first shape, as uncon
    cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd = cpcd_glo_gen[0].detach().cpu(), cpcd_tangent_glo_gen[0].detach().cpu(), tangent_from_ghd[0].detach().cpu()

    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles, edgecolor='yellow', linewidth=0.01, alpha=0.2, color='dodgerblue')
    color_list = ['red', 'black', 'blue']

    num_branch = len(cpcd_glo_gen)
    for i, color in zip(range(num_branch), color_list):
        pts_gen = cpcd_glo_gen[i].numpy()
        ax.scatter(pts_gen[:, 0], pts_gen[:, 1], pts_gen[:, 2], color=color, s=1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=45, azim=0)
    wandb.log({"uncon-gen": [wandb.Image(fig)]}, step=epoch)
    plt.close(fig)

    fig_recon = plt.figure(dpi=400)
    ax = fig_recon.add_subplot(111, projection='3d')
    id = randint(0, len(cpcd_glo_true)-1)
    cpcd_glo_true, cpcd_glo_pred = cpcd_glo_true[id].detach().cpu(), cpcd_glo_pred[id].detach().cpu()
    for i in range(num_branch):
        pts_true = cpcd_glo_true[i].numpy()
        ax.scatter(pts_true[::20, 0], pts_true[::20, 1], pts_true[::20, 2], color='yellow', s=2)
        pts_pred = cpcd_glo_pred[i].numpy()
        ax.scatter(pts_pred[:, 0], pts_pred[:, 1], pts_pred[:, 2], color=color_list[i], s=0.2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=45, azim=0)
    wandb.log({"recon": [wandb.Image(fig_recon)]}, step=epoch)
    plt.close(fig)


def generate_random_name():
    # Generate 5 random characters (letters)
    letters = ''.join(random.choices(string.ascii_letters, k=5))
    # Generate 5 random digits (numbers)
    digits = ''.join(random.choices(string.digits, k=5))
    return letters + digits





