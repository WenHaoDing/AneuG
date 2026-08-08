import torch
import os
from torch.utils.data import DataLoader
from utils.utils import safe_load_mesh
from models.vae_models import VAE, KL_divergence, ConditionalGHDVAE
from models.discriminators import PointNet2, GHD_Reconstruct
from models.losses import wgan_gradient_penalty
from torch_geometric.data import Data
import torch.nn.functional as F
import wandb
from models.utils import save_models, load_models, plot_wandb_conditional
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from losses.mesh_loss import Rigid_Loss
from models.dmm_calculator import DiffMorphoMarkerCalculator
from models.mesh_plugins import MeshPlugins, MeshRegulizer, CondDistributionApprox
from torch.distributions.multivariate_normal import MultivariateNormal
from copy import deepcopy
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
    mesh_plugin = MeshPlugins(canonical_Meshes, cep_chk, trimmed_mesh_path=trimmed_mesh_path, neck_chk_path=neck_chk_path, loop_start=[8, 6, 6])
    rigidloss = Rigid_Loss(canonical_Meshes.to(device))
    mesh_regulizer = MeshRegulizer(mesh_plugin=mesh_plugin, device=device, rigidloss=rigidloss)

    # morphological marker calculator (MMC)
    # currently support: AR, SR, LI, MC, NGCI, NW, V, Hmax, TA1, TA2, TA3, ImpD
    cond_keys=['AR', 'NW'] 
    dmm_calculator = DiffMorphoMarkerCalculator(mesh_plugin, cond_keys=cond_keys, device=device)
    cond_loss_style = 'distapprox'  # 'distapprox' or 'simple' or 'gaussian'

    # model conf
    epochs = 10000
    hidden_dim = 256
    latent_dim = 108
    batch_size = 128
    mode = 'train'
    use_norm = True  # if True, reconstruction loss includes normals
    use_reg = True  # if True, use regularization losses (MEA, trumpet, ...)
    MEA = True  # if True, use MEA, otherwise alignment performed on Gaussian distribution (reverse-regularization)
    drop_dmm = False  # if True, no condition loss
    reload_epoch = None

    # wandb conf
    log_wandb = False
    meta = 'your_meta'
    print("***************** Meta: ", meta)
    w_kl, w_reg, w_trumpet, w_cond = 0.00025, 0.5, 500.0, 0
    apply_norm=False  # if True, apply normalization to the condition when computing condition loss
    log_path = os.path.join("./checkpoints/paper/CVAE", meta)
    os.makedirs(log_path, exist_ok=True)
    if log_wandb:
        wandb.login()
        run = wandb.init(project="ghd_cvae",
                         name=meta,
                         config={"latent_dim": latent_dim, "batch_size": batch_size, "cond_keys": cond_keys, "cond_loss_style": cond_loss_style,
                                 "w_kl": w_kl, "w_reg": w_reg, "w_trumpet": w_trumpet, "w_cond": w_cond, "apply_norm": apply_norm, "MEA": MEA})
    
    
    # dataset
    ghd_reconstruct = GHD_Reconstruct(canonical_Meshes, eigen_chk, num_Basis=12**2, device=device)
    dataset = GHDDataset(ghd_chk_root, ghd_run, ghd_chk_name, ghd_reconstruct, cases, add_align=False, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    mean, std = dataset.get_mean_std()

    # model
    generator = ConditionalGHDVAE(dataset.get_dim(), hidden_dim, latent_dim, cond_dim=dmm_calculator.cond_dim).to(device)
    # discriminator = PointNet2(use_norm=use_norm).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=2000, gamma=0.5)

    # condist_approx
    uncond_net_path = os.path.join(root, "checkpoints/VAE", "unconditional_meta")
    uncond_net = VAE(dataset.get_dim(), hidden_dim=256, latent_dim=64).to(device)
    uncond_chk = torch.load(os.path.join(uncond_net_path, 'models_epoch_{}.pth'.format(10000)))
    uncond_net.load_state_dict(uncond_chk['generator'])
    uncond_net.eval()
    condist_approx = CondDistributionApprox(deepcopy(dataset),
                                            ghd_reconstruct, dmm_calculator,
                                            in_memory_size=1280,
                                            batch_size=128 if MEA else 64,
                                            uncond_net=uncond_net,
                                            device=device,
                                            apply_norm=apply_norm)

    # reload model if required
    if reload_epoch is not None or mode == 'eval':
        generator, optimizer_G, epoch_ = load_models(generator, optimizer_G, log_path, reload_epoch)
        print("Reloaded model from epoch: ", reload_epoch)
    else:
        epoch_ = 0

    for epoch in range(epoch_, epochs+1):
        for i, ghd in enumerate(dataloader):
            ghd = ghd.to(device)
            real_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=mean, std=std)
            B = ghd.shape[0]
            
            optimizer_G.zero_grad()
            data_real = ghd_reconstruct.forward(ghd, mean, std, use_norm)
            cond = dmm_calculator.forward(data_real)
            ghd_recon, mu, logvar = generator(ghd, condist_approx.normalize(cond, cond_keys))
            data_recon = ghd_reconstruct.forward(ghd_recon, mean, std, use_norm)
            
            # vert_loss = torch.norm(data_recon.pos - data_real.pos, dim=-1).mean()
            vert_loss = F.mse_loss(data_recon.pos, data_real.pos)
            if use_norm:
                norm_loss = F.mse_loss(data_recon.x[:, 3:], data_real.x[:, 3:])
            else:
                norm_loss = torch.tensor(0)
            mse_loss = F.mse_loss(ghd, ghd_recon)
            kl_loss = KL_divergence(mu, logvar)

            # reg loss
            B_reg = condist_approx.batch_size
            if not MEA:
                z_reg = torch.randn(B_reg, latent_dim).to(device)
            else:
                z_reg = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) * 1.0

            # create random condition & compute recalled morphological markers
            if cond_loss_style == 'simple':
                cond_min, cond_max = cond.min(dim=0, keepdim=True)[0], cond.max(dim=0, keepdim=True)[0]
                cond_fake = torch.rand(B_reg, cond.shape[1]).to(device) * (cond_max - cond_min) + cond_min
            elif cond_loss_style == 'gaussian':
                mu = cond.detach().mean(dim=0)
                covariance = torch.cov(cond.detach().T)
                dist = MultivariateNormal(mu, covariance_matrix=covariance)
                cond_fake = dist.sample((B_reg,))
            elif cond_loss_style == 'distapprox':
                cond_fake = condist_approx.get_batch(normalize=False)
                if DiffMorphoMarkerCalculator:
                    cond_fake = cond_fake[:z_reg.shape[0], :]
            cond_fake = condist_approx.normalize(cond_fake, cond_keys)
            ghd_fake = generator.decode(z_reg, cond_fake)
            data_fake = ghd_reconstruct.forward(ghd_fake, mean, std, use_norm)
            cond_fake_recall = dmm_calculator.forward(data_fake)
            loss_cond = condist_approx.cond_loss_forward(cond_fake, cond_fake_recall, cond_keys, 
                                                         normalize_recall=True, normalize_input=False,
                                                         generator=generator, drop_dmm=drop_dmm)
            
            #  reg loss
            fake_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd_fake, mean=mean, std=std)
            if use_reg:
                loss_rigid, loss_lap, loss_consistency, trumpet_loss = mesh_regulizer.KL_regulization(fake_meshes, real_meshes, MEA=MEA)
            else:
                loss_lap, loss_consistency, loss_rigid, trumpet_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            
            loss = w_kl * kl_loss + 1 * mse_loss + 100 * vert_loss + 10 * norm_loss + w_reg * loss_lap + \
                w_reg * loss_consistency + 1 * loss_rigid + w_trumpet * trumpet_loss + w_cond * loss_cond
            loss.backward()
            optimizer_G.step()
            
        if epoch % 10 == 0:
            log_dict = {"epoch": epoch, "kl_loss": kl_loss.item(), "mse_loss": mse_loss.item(), "vert_loss": vert_loss.item(),
                        'norm_loss': norm_loss.item(), "loss_lap": loss_lap.item(), "loss_consistency": loss_consistency.item(),
                        "loss_rigid": loss_rigid.item(), "loss_cond": loss_cond.item(), "loss_trumpet": trumpet_loss.item(), 
                        "total_loss": loss.item(), "lr": scheduler_G.optimizer.param_groups[0]['lr']}
            print(log_dict)
            if log_wandb:
                log_dict.pop("epoch")
                wandb.log(log_dict, step=epoch)
        
        if (epoch % 1000 == 0) or (epoch == epochs - 1):
            if reload_epoch is not None and epoch == reload_epoch:
                pass
            else:
                save_models(generator, optimizer_G, epoch, log_path, cond_keys, cond_loss_style, apply_norm)

        if epoch % 250 == 0:
            if log_wandb:
                plot_wandb_conditional(ghd_reconstruct, dataset, generator, epoch, device, use_norm, dmm_calculator, condist_approx, cond_loss_style)
        
        if epoch % round(10 * condist_approx.in_memory_size / batch_size) == 0:
            if cond_loss_style == 'distapprox':
                condist_approx.fill()
                print("Refilling condist_approx")

        scheduler_G.step()
    wandb.finish