# --------------------------------------------------------
# Code adapted from https://github.com/openai/shap-e/
# Licensed under The MIT License
# --------------------------------------------------------

from typing import Any, Optional

import torch
import torch.nn as nn

from .shap_e_ae import Transmitter
from unifi3d.models.autoencoders.shape2vecset_ae import DiagonalGaussianDistribution
from unifi3d.utils.model.shap_e_utils import AttrDict
from unifi3d.utils.rendering.camera import sample_data_cameras


class ShapEVAE(Transmitter):
    def __init__(
        self,
        config,
        ckpt_path=None,
        output_mode="nerf",
        output_dim=64,
        views_per_obj=2,
        latent_dim=1024,
    ):
        super().__init__(
            config=config,
            ckpt_path=ckpt_path,
            output_mode=output_mode,
            output_dim=output_dim,
            views_per_obj=2,
        )

        self.latent_dim = latent_dim
        self.mu_fn = nn.Linear(latent_dim, latent_dim)
        self.log_var_fn = nn.Linear(latent_dim, latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)

    def encode(self, data: dict, options: Optional[AttrDict] = None) -> Any:
        latents = self.encoder.encode_to_bottleneck(AttrDict(data), options=options)

        B = latents.size(dim=0)
        latents = torch.reshape(latents, (B, self.latent_dim, -1))
        mu = self.mu_fn(latents)
        log_var = self.log_var_fn(latents)

        # Create distribution
        posterior = DiagonalGaussianDistribution(mu, log_var)
        z = posterior.sample()
        kl = posterior.kl()

        z = torch.reshape(z, (B, -1))
        return z, kl

    def encode_wrapper(self, batch: dict) -> Any:
        z, _ = self.encode(batch)

        return z

    def decode(self, latent: torch.Tensor) -> Any:
        B = latent.size(dim=0)
        latent = torch.reshape(latent, (B, self.latent_dim, -1))
        latent = self.proj(latent)
        latent = torch.reshape(latent, (B, -1))
        params = self.encoder.bottleneck_to_params(latent)

        return params

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        # Encode
        latents, kls = self.encode(batch)

        # Decode and compute renders for each latent
        output = {}
        batch_camera_idxs = []
        batch_fine_renders = []
        batch_coarse_renders = []
        batch_fine_transmittance = []
        batch_coarse_transmittance = []
        for i, latent in enumerate(latents):
            params = self.decode(torch.unsqueeze(latent, dim=0))
            idxs, cameras = sample_data_cameras(
                batch["cameras"][i], self.output_dim, self.views_per_obj, self.device
            )
            outputs = self.renderer(
                AttrDict(cameras=cameras),
                params=params,
                options=AttrDict(
                    rendering_mode=self.output_mode,
                    render_with_direction=self.training,
                ),
            )

            batch_camera_idxs.append(idxs)
            batch_fine_renders.append(outputs.channels.clamp(0, 255) / 255.0)
            batch_coarse_renders.append(outputs.channels_coarse.clamp(0, 255) / 255.0)
            batch_fine_transmittance.append(outputs.transmittance.clamp(0, 255) / 255.0)
            batch_coarse_transmittance.append(
                outputs.transmittance_coarse.clamp(0, 255) / 255.0
            )

        output["loss_kl"] = kls

        output["batch_camera_idxs"] = batch_camera_idxs
        output["batch_fine_renders"] = torch.cat(batch_fine_renders)
        output["batch_coarse_renders"] = torch.cat(batch_coarse_renders)
        output["batch_fine_transmittance"] = torch.cat(batch_fine_transmittance)
        output["batch_coarse_transmittance"] = torch.cat(batch_coarse_transmittance)

        return output
