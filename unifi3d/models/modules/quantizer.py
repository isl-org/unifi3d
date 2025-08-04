# --------------------------------------------------------
# Code adapted from https://github.com/CompVis/taming-transformers/
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
# --------------------------------------------------------


import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from typing import Any
import pudb


class VectorQuantizer(torch.nn.Module):
    """
    Reference: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - codebook_size : number of embeddings
    - latent_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        codebook_size,
        latent_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = torch.nn.Embedding(self.codebook_size, self.latent_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.codebook_size} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = codebook_size

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds) -> Any:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds) -> torch.Tensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(
        self,
        z,
        temp=None,
        rescale_logits=False,
        return_logits=False,
        is_voxel=False,
        is_flat=False,
    ):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        if not is_voxel:
            if is_flat:
                z = rearrange(z, "b c d -> b d c").contiguous()
            else:
                z = rearrange(z, "b c h w -> b h w c").contiguous()
        else:
            z = rearrange(z, "b c d h w -> b d h w c").contiguous()
        z_flattened = z.view(-1, self.latent_dim)
        # Distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        if not is_voxel:
            if is_flat:
                z_q = rearrange(z_q, "b d c -> b c d").contiguous()
            else:
                z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        else:
            z_q = rearrange(z_q, "b d h w c -> b c d h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # Add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            if not is_voxel:
                if is_flat:
                    min_encoding_indices = min_encoding_indices.reshape(
                        z_q.shape[0], z_q.shape[2]
                    )
                else:
                    min_encoding_indices = min_encoding_indices.reshape(
                        z_q.shape[0], z_q.shape[2], z_q.shape[3]
                    )
            else:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4]
                )

        return z_q, loss, min_encoding_indices

    def quantize_indices(self, indices, shape) -> Any:
        # Shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # Add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # Flatten again

        # Get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # Reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class SimpleQuantizer(nn.Module):
    """
    Reference: https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/model/vqvae.py
    """

    def __init__(
        self,
        codebook_size,
        latent_dim,
        beta,
    ):
        super(SimpleQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)

    def forward(self, x):
        B, N, C = x.shape
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat(B, 1, 1))
        min_encoding_indices = torch.argmin(dist, dim=-1)

        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )
        x = x.reshape((-1, x.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_loss = codebook_loss + self.beta * commitment_loss

        quant_out = x + (quant_out - x).detach()
        quant_out = quant_out.reshape(B, N, C)
        return quant_out, quantize_loss, min_encoding_indices

    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, "b n c, n d -> b d c")
