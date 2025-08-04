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
from unifi3d.models.modules.quantizer import VectorQuantizer


class Shape2VecSetVQVAE(Shape2VecSetAE):
    def __init__(
        self,
        *,
        ckpt_path=None,
        latent_dim=64,
        codebook_size=512,
        **kwargs,
    ):

        super().__init__(ckpt_path=None, **kwargs)

        # Projection layers
        self.encoder_to_latent = torch.nn.Linear(self.dim, latent_dim)
        self.latent_to_decoder = torch.nn.Linear(latent_dim, self.dim)

        # Initialize the quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            beta=1.0,
        )

        # Network weights initialization
        init_weights(self.encoder_to_latent, "normal", 0.02)
        init_weights(self.latent_to_decoder, "normal", 0.02)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def encode(self, pc):
        # Use the parent class encode method to get x
        x = super().encode(pc)  # x shape: [B, num_latents, dim]

        # Project x to latent_dim
        x_latent = self.encoder_to_latent(x)  # [B, num_latents, latent_dim]

        # Quantize
        quantized_x, quantization_loss, _ = self.quantizer(x_latent, is_flat=True)

        # Check for NaNs and Infs in mu and log_var
        if torch.isnan(quantized_x).any() or torch.isinf(quantized_x).any():
            raise RuntimeError("mu contains NaNs or Infs")

        return quantization_loss, quantized_x

    def decode(self, quantized_x, queries):
        # Project quantized_x back to decoder input dimension
        x = self.latent_to_decoder(quantized_x)  # [B, num_latents, dim]

        # Continue with the decoding as in the parent class
        return super().decode(x, queries)

    def encode_no_quant(self, pc):
        # Use the parent class encode method to get x
        x = super().encode(pc)  # x shape: [B, num_latents, dim]

        # Project x to latent_dim
        x_latent = self.encoder_to_latent(x)  # [B, num_latents, latent_dim]

        return x_latent

    def decode_no_quant(self, h, queries, force_not_quantize=False):
        # If quantization is not forced to be skipped, pass 'h' through the quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantizer(h, is_flat=True)
        else:
            # If bypassing quantization, use 'h' directly
            quant = h

        # Apply post-quantization convolution
        x = self.latent_to_decoder(quant)  # [B, num_latents, dim]

        # Decode the quantized tensor
        dec = super().decode(x, queries)

        return dec

    def encode_wrapper(self, data):
        return self.encode_no_quant(data["pcl"]) * self.norm_factor

    def decode_wrapper(self, encoded_rep, query_points=None):
        encoded_rep = encoded_rep / self.norm_factor
        # Decode at grid query points
        if query_points is None:
            query_points = self.grid_query_points  # [1, (density + 1)^3, 3]

        bs = encoded_rep.size()[0]
        decoded = (
            self.decode_no_quant(encoded_rep, query_points, force_not_quantize=True)
            .view(bs, self.density + 1, self.density + 1, self.density + 1)
            .permute(0, 3, 1, 2)
        )
        return decoded

    def forward(self, batch):
        quantization_loss, quantized_x = self.encode(batch["pcl"])
        o = self.decode(quantized_x, batch["query_points"]).squeeze(-1)
        return {"logits": o, "quantize_loss": quantization_loss}
