"""
This file contains code for the auto-encoder implemented in Shap-E, called the Transmitter.
It takes as input a colored point cloud and images and outputs the parameters of a MLP, corresponding to a NeRF.
Code is taken from https://github.com/openai/shap-e/tree/main (MIT License), with minor modifications.
"""

from functools import partial
import os
import yaml
from typing import Any, Dict, Optional, Union
import torch
import numpy as np
import warnings

warnings.warn("ATTENTION: Shap-E was only tested with PyTorch 2.1.0 and Blender 3.6.2")

from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder
from unifi3d.models.modules.shap_e.nerstf import MLPNeRSTFModel, VoidNeRFModel
from unifi3d.models.modules.shap_e.nn import batch_meta_state_dict, subdict
from unifi3d.models.modules.shap_e.transmitter import (
    PointCloudPerceiverChannelsEncoder,
)
from unifi3d.utils.model.model_utils import load_net_from_safetensors
from unifi3d.utils.model.shap_e_utils import AttrDict
from unifi3d.utils.rendering.camera import sample_data_cameras
from unifi3d.utils.rendering.renderer import (
    BoundingBoxVolume,
    NeRSTFRenderer,
    nerf_to_mesh,
)


def model_from_config(
    config: Union[str, Dict[str, Any]], device: torch.device
) -> torch.nn.Module:
    config = config.copy()
    name = config.pop("name")

    if name == "NeRSTFRenderer":
        config = config.copy()
        for field in ["nerstf", "void", "volume"]:
            if field not in config:
                continue
            config[field] = model_from_config(config.pop(field), device)
        config.setdefault("sdf", None)
        config.setdefault("tf", None)
        config.setdefault("nerstf", None)
        return NeRSTFRenderer(device=device, **config)
    elif name == "MLPNeRSTFModel":
        return MLPNeRSTFModel(device=device, **config)
    elif name == "BoundingBoxVolume":
        return BoundingBoxVolume(device=device, **config)
    elif name == "VoidNeRFModel":
        return VoidNeRFModel(device=device, **config)
    elif name == "PointCloudPerceiverChannelsEncoder":
        return PointCloudPerceiverChannelsEncoder(
            device=device, dtype=torch.float32, **config
        )
    else:
        raise NotImplementedError

    return NeRSTFRenderer(device=device, **config)


class Transmitter(BaseEncoderDecoder):
    def __init__(
        self, config, ckpt_path=None, output_mode="nerf", output_dim=64, views_per_obj=2
    ):
        super().__init__()

        self.is_transmitter = True  # helper flag to identify this model
        self.output_mode = output_mode  # Only nerf implemented here
        self.output_dim = output_dim  # Dimension of rendered images
        self.views_per_obj = views_per_obj
        self.renderer_config = config.pop("renderer")
        self.encoder_config = config.pop("encoder")
        self.ckpt_path = ckpt_path

    def build(self, device):
        self.device = device
        self.renderer = model_from_config(self.renderer_config, device)
        param_shapes = {
            k: v.shape[1:]
            for k, v in batch_meta_state_dict(self.renderer, batch_size=1).items()
        }
        self.encoder_config["param_shapes"] = param_shapes
        self.encoder = model_from_config(self.encoder_config, device)

        if self.ckpt_path is not None:
            if self.ckpt_path.endswith("pt"):
                state_dict = torch.load(
                    self.ckpt_path, map_location=lambda storage, loc: storage
                )
                self.load_state_dict(state_dict, strict=False)
            elif self.ckpt_path.endswith("safetensors"):
                state_dict = load_net_from_safetensors(self.ckpt_path)
                self.load_state_dict(state_dict)
            print("Loaded checkpoint from", self.ckpt_path)

    def encode(self, data: dict, options: Optional[AttrDict] = None) -> Any:
        latents = self.encoder.encode_to_bottleneck(AttrDict(data), options=options)

        return latents

    def encode_wrapper(self, batch: dict) -> Any:
        return self.encode(batch)

    def decode(self, latent: torch.Tensor) -> Any:
        params = self.encoder.bottleneck_to_params(latent)

        return params

    def decode_wrapper(self, batch_latents: torch.Tensor) -> Any:
        batch_params = []
        for i, latent in enumerate(batch_latents):
            params = self.decode(torch.unsqueeze(latent, dim=0))
            batch_params.append(params)

        return batch_params

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        # Encode
        latents = self.encode(batch)

        # Decode and compute renders for each latent
        output = {}
        batch_camera_idxs = []
        batch_fine_renders = []
        batch_coarse_renders = []
        batch_fine_transmittance = []
        batch_coarse_transmittance = []
        for i, latent in enumerate(latents):
            params = self.encoder.bottleneck_to_params(torch.unsqueeze(latent, dim=0))
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
        output["batch_camera_idxs"] = batch_camera_idxs
        output["batch_fine_renders"] = torch.cat(batch_fine_renders)
        output["batch_coarse_renders"] = torch.cat(batch_coarse_renders)
        output["batch_fine_transmittance"] = torch.cat(batch_fine_transmittance)
        output["batch_coarse_transmittance"] = torch.cat(batch_coarse_transmittance)

        return output

    @torch.no_grad()
    def representation_to_mesh(
        self, params: AttrDict, options: Optional[AttrDict] = None
    ):
        """
        The Shap-E mesh generation assumes that the representation is a STF, not NeRF,
        so we have to find another implementation of marching cubes.
        Based on the render_views function in NeRSTFRenderer.

        Args:
            params -- Parameters for the NeRF's MLP
        Outputs:
            Open3D TriangleMesh computed using params and other information in the Shap-E renderer.
        """

        params = self.renderer.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        nerstf_fn = None
        if self.renderer.nerstf is not None:
            nerstf_fn = partial(
                self.renderer.nerstf.forward_batched,
                params=subdict(params, "nerstf"),
                options=options,
            )
        else:
            raise NotImplementedError
        output = nerf_to_mesh(
            options,
            nerstf_fn=nerstf_fn,
            volume=self.renderer.volume,
            grid_size=self.renderer.grid_size,
            device=self.device,
        )

        return output
