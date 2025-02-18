import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import safe_load_mesh
from models.vae_models import VAE, KL_divergence
from models.ghd_reconstruct import GHD_Reconstruct
from models.losses import wgan_gradient_penalty
from torch_geometric.data import Data
import torch.nn.functional as F
import wandb
from models.utils import save_models, load_models, plot_wandb
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from losses.mesh_loss import Rigid_Loss
from models.mesh_plugins import MeshRegulizer, MeshPlugins
from models.vae_datasets import GHDDataset


if __name__ == "__main__":
    device = torch.device('cuda:0')
    # data conf
    """
    data structure for ghd checkpoints:
    - root
        - checkpoints
            - fit_mca_stable
                - case1
                    - vanilla
                        - ghd_fitting_checkpoint.pkl
    """
    root = 'your/root/dir'
    ghd_chk_root = os.path.join(root, 'checkpoints/fit_mca_stable')
    ghd_run = 'vanilla'
    ghd_chk_name = 'ghb_fitting_checkpoint.pkl'
    eigen_chk = os.path.join(root, "checkpoints/canonical_typeB_144.pkl")
    alignment_root = os.path.join(root, 'checkpoints/align_mca_female')
    canonical_name = 'canonical_typeB'
    canonical_Meshes = safe_load_mesh(os.path.join(alignment_root, canonical_name, 'part_aligned_updated.obj'))
    cases = [case for case in os.listdir(ghd_chk_root) if os.path.isdir(os.path.join(ghd_chk_root, case)) and case != canonical_name]

    # mesh plugins
    cep_chk = "./data/excision_registration/combined_stable/canonical_typeB/diff_centreline_checkpoint.pkl"
    trimmed_mesh_path = os.path.join(os.getcwd(), "data/excision_registration/combined_stable/canonical_typeB/part_trimmed_short.obj")
    neck_chk_path = os.path.join(os.getcwd(), "data/excision_registration/combined_stable/canonical_typeB/neck_size_checkpoint.pkl")
    mesh_plugin = MeshPlugins(canonical_Meshes, cep_chk, trimmed_mesh_path=trimmed_mesh_path, neck_chk_path=neck_chk_path)
    rigidloss = Rigid_Loss(canonical_Meshes.to(device))
    mesh_regulizer = MeshRegulizer(mesh_plugin=mesh_plugin, device=device, rigidloss=rigidloss)

    # model conf
    epochs = 10000
    hidden_dim = 256
    latent_dim = 108
    batch_size = 128
    mode = 'train'
    use_norm = True  # if True, reconstruction loss includes normals
    use_reg = True  # if True, use regularization losses (MEA, trumpet, ...)
    withscale = True  # if True, consider the scale of the mesh
    MEA = True  # if True, use MEA, otherwise alignment performed on Gaussian distribution (reverse-regularization)
    overreg = False  # if True, directly descend morphing energies, (over-regularization)
    huber_loss = nn.HuberLoss(delta=2.0)
    reload_epoch = None
    # wandb conf
    log_wandb = True
    meta = 'your_meta'
    log_path = os.path.join("./checkpoints/first_stage_unconditional", meta)
    os.makedirs(log_path, exist_ok=True)
    if log_wandb:
        wandb.login()
        run = wandb.init(project="your_project",
                         name=meta)
        
    # dataset
    ghd_reconstruct = GHD_Reconstruct(canonical_Meshes, eigen_chk, num_Basis=12**2, device=device)
    dataset = GHDDataset(ghd_chk_root, ghd_run, ghd_chk_name, ghd_reconstruct, cases, withscale=withscale, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    mean, std = dataset.get_mean_std()
    # model
    generator = VAE(dataset.get_dim(), hidden_dim, latent_dim, withscale=withscale).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=2000, gamma=0.5)
    # reload model if required
    if reload_epoch is not None or mode == 'eval':
        generator, optimizer_G, epoch_ = load_models(generator, optimizer_G, log_path, reload_epoch)
        print("Reloaded model from epoch: ", reload_epoch)
    else:
        epoch_ = 0

    for epoch in range(epoch_, epochs+1):
        for i, ghd in enumerate(dataloader):
            ghd = ghd.to(device)
            # split scale ratio if usescale
            if withscale:
                ghd, scale = ghd[:, :-1], ghd[:, -1:]
            else:
                scale = None
            real_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=mean, std=std)
            B = ghd.shape[0]
            
            optimizer_G.zero_grad()
            if withscale:
                ghd_recon, scale_recon, mu, logvar = generator(ghd, scale)
                scale_loss = huber_loss(scale_recon, scale)
            else:
                ghd_recon, mu, logvar = generator(ghd, scale)
                scale_loss = torch.tensor(0).to(device)
            data_recon = ghd_reconstruct.forward(ghd_recon, mean, std, use_norm)
            data_real = ghd_reconstruct.forward(ghd, mean, std, use_norm)
            # vert_loss = torch.norm(data_recon.pos - data_real.pos, dim=-1).mean()
            vert_loss = F.mse_loss(data_recon.pos, data_real.pos)
            if use_norm:
                norm_loss = F.mse_loss(data_recon.x[:, 3:], data_real.x[:, 3:])
            else:
                norm_loss = torch.tensor(0)
            mse_loss = F.mse_loss(ghd, ghd_recon)
            kl_loss = KL_divergence(mu, logvar)

            # reg loss
            if use_reg:
                B_reg = 128
                if not MEA:
                    z_reg = torch.randn(B_reg, latent_dim).to(device)
                else:
                    z_reg = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) * 1.0
                ghd_fake = generator.decode(z_reg, True)
                fake_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd_fake, mean=mean, std=std)
                loss_rigid, loss_lap, loss_consistency, trumpet_loss = mesh_regulizer.KL_regulization(fake_meshes, real_meshes, MEA=MEA, overreg=overreg)
            else:
                loss_lap = torch.tensor(0)
                loss_consistency = torch.tensor(0)
                loss_rigid = torch.tensor(0)
                trumpet_loss = torch.tensor(0)

            loss = 0.00025 * kl_loss + 1 * mse_loss + 0.25 * scale_loss + 100 * vert_loss + 10 * norm_loss \
                    + 1 * loss_lap + 1 * loss_consistency + 1 * loss_rigid + 2000.0 * trumpet_loss

            loss.backward()
            optimizer_G.step()
            
        if epoch % 10 == 0:
            log_dict = {"epoch": epoch, "kl_loss": kl_loss.item(), "mse_loss": mse_loss.item(), "vert_loss": vert_loss.item(), 'norm_loss': norm_loss.item(),
                        "loss_lap": loss_lap.item(), "loss_consistency": loss_consistency.item(), "loss_rigid": loss_rigid.item(), "trumpet_loss": trumpet_loss.item(), "scale_loss": scale_loss.item()}
            print(log_dict)
            if log_wandb:
                log_dict.pop("epoch")
                wandb.log(log_dict, step=epoch)
        
        if epoch % 1000 == 0:
            save_models(generator, optimizer_G, epoch, log_path)

        if epoch % 500 == 0:
            if log_wandb:
                plot_wandb(ghd_reconstruct, dataset, generator, latent_dim, epoch, device, use_norm, withscale)
        scheduler_G.step()
    wandb.finish
