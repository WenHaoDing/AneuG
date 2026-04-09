from baselines.Michelangelo.michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
import torch

class OccupancyVAE(ShapeAsLatentPerceiver):
    def __init__(self, device, num_latents=256, point_feats=3, embed_dim=64, num_freqs=8, include_pi=False,
                 width=768, heads=12, num_encoder_layers=8, num_decoder_layers=16, init_scale=0.25, qkv_bias=False,
                 flash=False, use_ln_post=False, use_checkpoint=False):
        super().__init__(device=None, dtype=None, num_latents=num_latents, point_feats=point_feats, embed_dim=embed_dim,
                         num_freqs=num_freqs, include_pi=include_pi, width=width, heads=heads,
                         num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                         init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, use_ln_post=use_ln_post,
                         use_checkpoint=use_checkpoint)
        self.latent_shape_ = [num_latents, embed_dim]

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """
        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior

