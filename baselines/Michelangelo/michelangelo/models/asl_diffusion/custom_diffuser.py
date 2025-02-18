# -*- coding: utf-8 -*-

from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from einops import rearrange

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

from michelangelo.utils import instantiate_from_config
# from michelangelo.models.tsal.tsal_base import ShapeAsLatentPLModule
from michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentPLModule
from michelangelo.models.asl_diffusion.inference_utils import ddim_sample, custom_ddim_sample

from typing import Optional
from diffusers.models.embeddings import Timesteps
import math

from michelangelo.models.modules.transformer_blocks import MLP
from michelangelo.models.modules.diffusion_transformer import UNetDiffusionTransformer





SchedulerType = Union[DDIMScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler]

class CUSDiffuser(pl.LightningModule):
    def __init__(self,
                 occupancy_vae: nn.Module,
                 diffusion_model: nn.Module,
                 device: torch.device,
                 sample_posterior=True):
        
        super().__init__()
        self.occupancy_vae = occupancy_vae.to(device)
        for param in self.occupancy_vae.parameters():
            param.requires_grad = False
        self.model = diffusion_model.to(device)
        self.sample_posterior = sample_posterior
        self.num_train_timesteps = 1000
        beta_start = 0.00085
        beta_end = 0.012
        beta_schedule = "scaled_linear"

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.num_train_timesteps,
                                             beta_start=beta_start, beta_end=beta_end,
                                             beta_schedule=beta_schedule, variance_type="fixed_small",
                                             clip_sample=False)
        self.denoise_scheduler = DDIMScheduler(num_train_timesteps=self.num_train_timesteps,
                                               beta_start=beta_start, beta_end=beta_end,
                                               beta_schedule=beta_schedule, 
                                               clip_sample=False,
                                               set_alpha_to_one=False,
                                               steps_offset=1)
                                                                                    
        
        self.loss_cfg = DictConfig({"loss_type": "rl2"})

    def encode_first_stage(self, batch):
        surface_points = batch["surface_points"]
        surface_normals = batch["surface_normals"]
        latents, _, _ = self.occupancy_vae.encode(surface_points, surface_normals, sample_posterior=self.sample_posterior)
        return latents
    
    def decode_first_stage(self, latents):
        return self.occupancy_vae.decode(latents)

    def forward(self, batch):
        latents = self.encode_first_stage(batch)
        # Sample noise that we"ll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # diffusion model forward
        noise_pred = self.model(noisy_z, timesteps)

        diffusion_outputs = {
            "x_0": noisy_z,
            "noise": noise,
            "pred": noise_pred
        }

        return diffusion_outputs

    def training_step(self, batch):
        diffusion_outputs = self(batch)
        loss, loss_dict = self.compute_loss(diffusion_outputs, "train")
        return loss

    def compute_loss(self, model_outputs, split):
        """
        Args:
            model_outputs (dict):
                - x_0:
                - noise:
                - noise_prior:
                - noise_pred:
                - noise_pred_prior:

            split (str):
        Returns:
        """

        pred = model_outputs["pred"]

        if self.noise_scheduler.prediction_type == "epsilon":
            target = model_outputs["noise"]
        elif self.noise_scheduler.prediction_type == "sample":
            target = model_outputs["x_0"]
        else:
            raise NotImplementedError(f"Prediction Type: {self.noise_scheduler.prediction_type} not yet supported.")

        if self.loss_cfg.loss_type == "l1":
            simple = F.l1_loss(pred, target, reduction="mean")
        elif self.loss_cfg.loss_type in ["mse", "l2"]:
            simple = F.mse_loss(pred, target, reduction="mean")
        elif self.loss_cfg.loss_type == "rl2":
            simple = F.mse_loss(pred, target, reduction="mean") / F.mse_loss(target, torch.zeros_like(target), reduction="mean")
        else:
            raise NotImplementedError(f"Loss Type: {self.loss_cfg.loss_type} not yet supported.")

        total_loss = simple

        loss_dict = {
            f"{split}/total_loss": total_loss.clone().detach(),
            f"{split}/simple": simple.detach(),
        }

        return total_loss, loss_dict
    
    def sample(self, 
               bsz,
               steps=50,
               eta=0.0):
        
        outputs = []
        latents = None
        
        sample_loop = custom_ddim_sample(
            self.denoise_scheduler,
            self.model,
            bsz=bsz,
            shape=self.occupancy_vae.latent_shape_,
            steps=steps,
            device=self.device,
            eta=eta,
            disable_prog=True
        )
        for sample, t in sample_loop:
            latents = sample
        outputs.append(self.decode_first_stage(latents))
        return outputs


class MMConditionalDenoiser(nn.Module):
    """
    n_ctx is not used.
    """
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 input_channels: int=64,
                 output_channels: int=64,
                 n_ctx: int=256,
                 width: int=768,
                 layers: int=6,
                 heads: int=12,
                 context_dim: int=3,
                 context_ln: bool = True,
                 skip_ln: bool = False,
                 init_scale: float = 0.25,
                 flip_sin_to_cos: bool = False,
                 use_checkpoint: bool = False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.backbone = UNetDiffusionTransformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            skip_ln=skip_ln,
            init_scale=init_scale,
            use_checkpoint=use_checkpoint
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)

        # timestep embedding
        self.time_embed = Timesteps(width, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=0)
        self.time_proj = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale
        )

        self.context_embed = nn.Sequential(
            nn.LayerNorm(context_dim, device=device, dtype=dtype),
            nn.Linear(context_dim, width, device=device, dtype=dtype),
        )

        if context_ln:
            self.context_embed = nn.Sequential(
                nn.LayerNorm(context_dim, device=device, dtype=dtype),
                nn.Linear(context_dim, width, device=device, dtype=dtype),
            )
        else:
            self.context_embed = nn.Linear(context_dim, width, device=device, dtype=dtype)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor=None):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

            Can be unconditional by setting context to None.
        """

        _, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_proj(self.time_embed(timestep)).unsqueeze(dim=1)

        if context is not None:
            # 2. conditions projector
            context = self.context_embed(context)

        # 3. denoiser
        x = self.input_proj(model_input)
        x = torch.cat([t_emb, context, x], dim=1) if context is not None else torch.cat([t_emb, x], dim=1)
        x = self.backbone(x)
        x = self.ln_post(x)
        x = x[:, -n_data:]
        sample = self.output_proj(x)
        return sample