# --------------------------------------------------------
# Adapted FROM https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt
# Licsense: Attribution-noncommercial 4.0 International (CC BY-NC 4.0)
# --------------------------------------------------------

import numpy as np
import math
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from unifi3d.models.modules.patch_embed import Tokenizer
from unifi3d.models.modules.diffusion.conditioning import (
    TimestepEmbedder,
    LabelEmbedder,
    ClipTextEncoder,
    ClipImageEncoder,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_shape):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_shape, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 1,
        hidden_size: int = 512,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        patch_embed=Tokenizer(),
        conditioning_modes=[],
        batchnorm: bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.conditioning_modes = conditioning_modes

        # double out channels if learning mu and sigma
        self.out_channels = 2 * in_channels if learn_sigma else in_channels
        # initialize x embeder
        self.x_embedder = patch_embed
        # final layer
        self.final_layer = FinalLayer(
            hidden_size, self.x_embedder.patch_volume * self.out_channels
        )

        # Init embedders of context (for conditional generation)
        self.y_embedder = None
        if "category" in conditioning_modes:
            self.y_embedder = LabelEmbedder(
                num_classes, hidden_size, class_dropout_prob
            )
        if "text" in conditioning_modes:
            self.text_encoder = ClipTextEncoder(hidden_size)
        if "image" in conditioning_modes:
            self.img_encoder = ClipImageEncoder(hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.normalize = nn.BatchNorm1d(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        self.pos_embed = self.x_embedder.get_pos_embed(self.hidden_size)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, context={}):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if self.batchnorm:
            x = self.normalize(x)
        # embed time step of diffusion
        context_emb = self.t_embedder(t)  # (N, D)

        # tokenize x # (N, T, D), where T = H * W / patch_size ** 2
        x = self.x_embedder(x) + self.pos_embed.to(x.device)

        # add other conditioning to context # TODO: wouldn't it be better to concatenate?
        if "text" in self.conditioning_modes:
            context_emb += self.text_encoder(context["text"]).to(x.device)
        if "category" in self.conditioning_modes:
            # add class condition to context
            context_emb += self.y_embedder(context["category"])
        if "image" in self.conditioning_modes:
            context_emb += self.img_encoder(context["image"]).to(x.device)

        for block in self.blocks:
            x = block(x, context_emb)  # (N, T, D)
        # final layer: divide into variance and mean
        x = self.final_layer(x, context_emb)  # (N, T, patch_size ** 2 * out_channels)

        # unpatchify
        x = self.x_embedder.unpatchify(x, self.out_channels)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
