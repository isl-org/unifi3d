# --------------------------------------------------------
# Code adapted from https://github.com/openai/shap-e/
# Licensed under The MIT License
# --------------------------------------------------------

from typing import Any, Optional

import torch
import torch.nn as nn

from .shap_e_ae import Transmitter
from unifi3d.models.modules.quantizer import VectorQuantizer
from unifi3d.utils.model.shap_e_utils import AttrDict
from unifi3d.utils.rendering.camera import sample_data_cameras


class ShapEVQVAE(Transmitter):
    def __init__(
        self,
        config,
        ckpt_path=None,
        output_mode="nerf",
        output_dim=64,
        views_per_obj=2,
        latent_dim=1024,
        codebook_size=1024,
    ):
        super().__init__(
            config=config,
            ckpt_path=ckpt_path,
            output_mode=output_mode,
            output_dim=output_dim,
            views_per_obj=2,
        )

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        # Projection layers
        self.encoder_to_latent = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.latent_to_decoder = torch.nn.Linear(self.latent_dim, self.latent_dim)

        # Initialize the quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=self.codebook_size,
            latent_dim=self.latent_dim,
            beta=1.0,
        )

    def encode(self, data: dict, options: Optional[AttrDict] = None) -> Any:
        latents = self.encoder.encode_to_bottleneck(AttrDict(data), options=options)

        # Project x to latent_dim
        B = latents.size(dim=0)
        latents = torch.reshape(latents, (B, self.latent_dim, -1))
        x = self.encoder_to_latent(latents)

        # Quantize
        quantized_x, quantization_loss, _ = self.quantizer(x, is_flat=True)
        quantized_x = torch.reshape(quantized_x, (B, -1))
        quantization_loss = torch.unsqueeze(quantization_loss, dim=0)

        return quantized_x, quantization_loss

    def encode_no_quant(self, data: dict, options: Optional[AttrDict] = None) -> Any:
        latents = self.encoder.encode_to_bottleneck(AttrDict(data), options=options)

        # Project x to latent_dim
        B = latents.size(dim=0)
        latents = torch.reshape(latents, (B, self.latent_dim, -1))
        x = self.encoder_to_latent(latents)
        x = torch.reshape(x, (B, -1))

        return x

    def decode(self, quantized_x: torch.Tensor) -> Any:
        B = quantized_x.size(dim=0)
        quantized_x = torch.reshape(quantized_x, (B, self.latent_dim, -1))
        latent = self.latent_to_decoder(quantized_x)
        latent = torch.reshape(latent, (B, -1))
        params = self.encoder.bottleneck_to_params(latent)

        return params

    def decode_no_quant(self, h: torch.Tensor, force_not_quantize=False) -> Any:
        B = h.size(dim=0)
        h = torch.reshape(h, (B, self.latent_dim, -1))

        # If quantization is not forced to be skipped, pass 'h' through the quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantizer(h, is_flat=True)
        else:
            # If bypassing quantization, use 'h' directly
            quant = h

        # Apply post-quantization convolution
        latent = self.latent_to_decoder(quant)
        latent = torch.reshape(latent, (B, -1))
        params = self.encoder.bottleneck_to_params(latent)

        return params

    def encode_wrapper(self, batch: dict) -> Any:
        latents = self.encode_no_quant(batch)

        return latents

    def decode_wrapper(self, batch_latents: torch.Tensor) -> Any:
        batch_params = []
        for i, latent in enumerate(batch_latents):
            params = self.decode_no_quant(torch.unsqueeze(latent, dim=0))
            batch_params.append(params)

        return batch_params

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        # Encode
        latents, qls = self.encode(batch)

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

        output["loss_quantized"] = qls

        output["batch_camera_idxs"] = batch_camera_idxs
        output["batch_fine_renders"] = torch.cat(batch_fine_renders)
        output["batch_coarse_renders"] = torch.cat(batch_coarse_renders)
        output["batch_fine_transmittance"] = torch.cat(batch_fine_transmittance)
        output["batch_coarse_transmittance"] = torch.cat(batch_coarse_transmittance)

        return output
