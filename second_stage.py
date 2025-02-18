import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import safe_load_mesh
from models.vae_models import VAE, VAE2, KL_divergence
from models.discriminators import PointNet2, GHD_Reconstruct
from models.losses import wgan_gradient_penalty
from torch_geometric.data import Data
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from models.utils import save_models, load_models, plot_wandb
from sklearn.decomposition import PCA
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from losses.mesh_loss import Rigid_Loss
from models.utils import get_fig, get_fig_advanced
from models.utils import plot_wandb_centerline
from models.mesh_plugins import MeshPlugins
from models.vae_datasets import GHDDataset
withscale = True

device = torch.device("cuda:0")
root = 'your/root/dir'
ghd_chk_root = os.path.join(root, 'checkpoints/ghd_fitting')
ghd_run = 'vanilla'
ghd_chk_name = 'ghb_fitting_checkpoint_5.pkl'
eigen_chk = os.path.join(root, "checkpoints/canonical_typeB_144.pkl")
alignment_root = os.path.join(root, 'checkpoints/align_mca_female')
canonical_name = 'canonical_typeB'
canonical_Meshes = safe_load_mesh(os.path.join(alignment_root, canonical_name, 'part_aligned_updated.obj'))
# mesh plugins & dmm calculator
cep_chk = "./data/excision_registration/combined_stable/canonical_typeB/diff_centreline_checkpoint.pkl"
trimmed_mesh_path = os.path.join(os.getcwd(), "data/excision_registration/combined_stable/canonical_typeB/part_trimmed_short.obj")
neck_chk_path = os.path.join(os.getcwd(), "data/excision_registration/combined_stable/canonical_typeB/neck_size_checkpoint.pkl")
mesh_plugin = MeshPlugins(canonical_Meshes, cep_chk, trimmed_mesh_path=trimmed_mesh_path, neck_chk_path=neck_chk_path)
ghd_reconstruct = GHD_Reconstruct(canonical_Meshes, eigen_chk, num_Basis=12**2, device=device)
cases = [case for case in os.listdir(ghd_chk_root) if os.path.isdir(os.path.join(ghd_chk_root, case)) and case != canonical_name]
ghd_dataset = GHDDataset(ghd_chk_root, ghd_run, ghd_chk_name, ghd_reconstruct, cases, withscale=withscale, normalize=True)
mean_ghd, std_ghd = ghd_dataset.get_mean_std()
mean_scale, std_scale = ghd_dataset.get_scale_mean_std()
hidden_dim_ghd = 256
latent_dim_ghd = 108

# load GHD VAE
ghd_vae = VAE(ghd_dataset.get_dim(), hidden_dim_ghd, latent_dim_ghd, withscale=withscale).to(device)
ghd_vae_chk = os.path.join("./checkpoints/paper/VAE", 'stable_108_medereg_strongtrumpet_withscale', 'models_epoch_{}.pth'.format(9000))
ghd_vae.load_state_dict(torch.load(ghd_vae_chk)['generator'])
ghd_vae.eval()

# %%
from models.vae_datasets import CenterlineDataset
from models.vae_models import CPCDReconstruct

# centerline dataset
cl_chk_root = "./checkpoints/center_fit/stable"
toss_threshold = 0.001
centerline_dataset = CenterlineDataset(cl_chk_root, normalize=True, toss_threshold=toss_threshold, device=device)
num_branch, num_fourier, fourier_per_branch = centerline_dataset.num_branch, centerline_dataset.num_fourier, centerline_dataset.fourier_per_branch
data_loader = DataLoader(centerline_dataset, batch_size=128, shuffle=False)

# cpcd reconstructer
cpcd_reconstruct = CPCDReconstruct(num_branch, num_fourier, fourier_per_branch, device=device)

from models.vae_models import ConditionalVAE4Fouriers

# model conf
epochs = 5001
reload_epoch = None
hidden_dim = 256
latent_dim = 3
basis_include = 12**2
mode = 'train'
reload_epoch = None
log_wandb = True
meta = 'h256_l3_noreg_withscale'
log_path = os.path.join("./checkpoints/CVAE", meta)
os.makedirs(log_path, exist_ok=True)
w_cpcd = 200
w_cpcd_tangent = 50
w_tangent_reg = 5
if log_wandb:
    wandb.login()
    run = wandb.init(project="co_generation",
                     name=meta,
                     config={"hidden_dim": hidden_dim,
                             "latent_dim": latent_dim,
                             "w_cpcd": w_cpcd,
                             "w_cpcd_tangent": w_cpcd_tangent,
                             "w_tangent_reg": w_tangent_reg,})

cvae = ConditionalVAE4Fouriers(num_branch, num_fourier, fourier_per_branch, num_basis=basis_include,
                               hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=0.02,
                               tangent_encoding=True, ghd_reconstruct=ghd_reconstruct, mesh_plugin=mesh_plugin,
                               cpcd_reconstruct=cpcd_reconstruct, 
                               norm_dict=centerline_dataset.return_norm_dict(device), normalize=True,
                               ghd_encoding=True,
                               withscale=withscale).to(device)

optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
mse = torch.nn.MSELoss()

if reload_epoch is not None:
    chk = torch.load(os.path.join(log_path, 'models_epoch_{}.pth'.format(reload_epoch)))
    cvae.load_state_dict(chk['generator'])
    optimizer.load_state_dict(chk['optimizer'])
    scheduler.load_state_dict(chk['scheduler'])
    epoch_ = chk['epoch']
    print("Reloaded from epoch: ", epoch_)
else:
    epoch_ = 0

for epoch in range(epoch_, epochs):
    for i, data_dict in enumerate(data_loader):
        cvae.train()
        optimizer.zero_grad()
        x, recon, mu, logvar, cpcd_glo_true, cpcd_glo_pred, cpcd_tangent_glo_true, cpcd_tangent_glo_pred, accurate_tangent = cvae(data_dict)
        B = x.shape[0]

        # sample latent
        ghd_fake = ghd_vae.decode(torch.randn(B, latent_dim_ghd).to(device))
        if withscale:
            ghd_fake, scale_fake = ghd_fake[0], ghd_fake[1]
            scale_fake = scale_fake * std_scale.to(scale_fake.device) + mean_scale.to(scale_fake.device)
        else:
            ghd_fake, scale_fake = ghd_fake, None
        if mean_ghd is not None and std_ghd is not None:
            ghd_fake = ghd_fake * std_ghd.to(ghd_fake.device) + mean_ghd.to(ghd_fake.device)
        cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd = cvae.generate(ghd_fake, scale_fake)

        # loss
        recon_loss = mse(x, recon)
        kl_loss = KL_divergence(mu, logvar)
        cpcd_loss = mse(cpcd_glo_true, cpcd_glo_pred)
        cpcd_tangent_loss = mse(cpcd_tangent_glo_true, cpcd_tangent_glo_pred) 
        tangent_reg_loss = mse(tangent_from_ghd, cpcd_tangent_glo_gen[:, :, 0, :])  # we only apply reg loss for the start point.

        loss = recon_loss + kl_loss + w_cpcd * cpcd_loss + w_cpcd_tangent * cpcd_tangent_loss + w_tangent_reg * tangent_reg_loss
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        log_dict = {"epoch": epoch,
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "cpcd_loss": cpcd_loss.item(),
                    "cpcd_tangent_loss": cpcd_tangent_loss.item(),
                    "tangent_reg_loss": tangent_reg_loss.item()}
    print(log_dict)
    if log_wandb:
        log_dict.pop("epoch")
        wandb.log(log_dict, step=epoch)
    scheduler.step()

    if epoch % 200 == 0 and log_wandb:
        plot_wandb_centerline(ghd_reconstruct, mesh_plugin, 
                              ghd_fake, cpcd_glo_gen, cpcd_tangent_glo_gen, tangent_from_ghd,
                              cpcd_glo_true, cpcd_glo_pred,
                              epoch,
                              scale=scale_fake)
        
    if epoch % 1000 == 0:
        chk_path = os.path.join(log_path, 'models_epoch_{}.pth'.format(epoch))
        torch.save({'generator': cvae.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}, chk_path)
        
wandb.finish



