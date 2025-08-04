# --------------------------------------------------------
# Code adapted from https://github.com/1zb/3DShape2VecSet/
# Licensed under The MIT License
# --------------------------------------------------------


from einops import rearrange, repeat
from functools import wraps
import numpy as np
from timm.models.layers import DropPath
import torch
from torch_cluster import fps

from unifi3d.models.autoencoders.shape2vecset_ae import *


class Shape2VecSetVAE(Shape2VecSetAE):
    def __init__(
        self,
        *,
        ckpt_path=None,
        latent_dim=64,
        **kwargs,
    ):
        super().__init__(ckpt_path=None, **kwargs)

        self.logvar_min_threshold = np.log(1e-8)
        self.logvar_max_threshold = np.log(3e4)

        # Projection layer from latent space back to model dimension
        self.proj = torch.nn.Linear(latent_dim, self.dim)

        # VAE-specific layers
        self.encoder_mu_fc = torch.nn.Linear(self.dim, latent_dim)
        self.encoder_var_fc = torch.nn.Linear(self.dim, latent_dim)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, log_var):
        return (
            -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1).mean()
        )

    def encode(self, x):
        encoder_out = super().encode(x)  # x shape: [B, num_latents, dim]
        mu = self.encoder_mu_fc(encoder_out)  # [B, num_latents, latent_dim]
        log_var = self.encoder_var_fc(encoder_out)  # [B, num_latents, latent_dim]

        # clamp to avoid infs with log_var.exp(), especially with mixed precision
        log_var = torch.clamp(
            log_var, self.logvar_min_threshold, self.logvar_max_threshold
        )

        return mu, log_var

    def decode(self, x, queries):
        # Project x back to the dimension expected by decoder
        x = self.proj(x)  # x shape: [B, num_latents, dim]

        # Continue with the decoding as in the parent class
        return super().decode(x, queries)

    def encode_wrapper(self, batch):
        # use mean as latent
        mean, _ = self.encode(batch["pcl"])
        return mean

    def decode_wrapper(self, encoded_rep, query_points=None):
        # encoded_rep = encoded_rep / self.norm_factor
        # Decode at grid query points
        if query_points is None:
            query_points = self.grid_query_points  # Shape [1, (self.density + 1)^3, 3]

        bs = encoded_rep.size()[0]
        decoded = (
            self.decode(encoded_rep, query_points)
            .view(bs, self.density + 1, self.density + 1, self.density + 1)
            .permute(0, 3, 1, 2)
        )
        return decoded

    def forward(self, batch):
        mean, log_var = self.encode(batch["pcl"])
        z = self.reparameterize(mean, log_var)
        kl_divergence = self.kl_divergence(mean, log_var)
        o = self.decode(z, batch["query_points"]).squeeze(-1)
        return {"logits": o, "kl": kl_divergence}
