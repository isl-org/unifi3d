"""
MIT License

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from unifi3d.models.modules.shap_e.generation import SimplePerceiver, Transformer
from unifi3d.models.modules.shap_e.nn import (
    MultiviewPointCloudEmbedding,
    PosEmbLinear,
    PointSetEmbedding,
)
from unifi3d.utils.model.shap_e_utils import AttrDict
from unifi3d.utils.rendering.camera import (
    ProjectiveCamera,
    DifferentiableProjectiveCamera,
)


def flatten_param_shapes(param_shapes: Dict[str, Tuple[int]]):
    flat_shapes = OrderedDict(
        (name, (int(np.prod(shape)) // shape[-1], shape[-1]))
        for name, shape in param_shapes.items()
    )
    return flat_shapes


class ParamsProj(nn.Module, ABC):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
    ):
        super().__init__()
        self.device = device
        self.param_shapes = param_shapes
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class ChannelsProj(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        vectors: int,
        channels: int,
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(d_latent, vectors * channels, device=device)
        self.use_ln = use_ln
        self.learned_scale = learned_scale
        if use_ln:
            self.norm = nn.LayerNorm(normalized_shape=(channels,), device=device)
            if learned_scale is not None:
                self.norm.weight.data.fill_(learned_scale)
            scale = init_scale / math.sqrt(d_latent)
        elif learned_scale is not None:
            gain = torch.ones((channels,), device=device) * learned_scale
            self.register_parameter("gain", nn.Parameter(gain))
            scale = init_scale / math.sqrt(d_latent)
        else:
            scale = init_scale / math.sqrt(d_latent * channels)
        nn.init.normal_(self.proj.weight, std=scale)
        nn.init.zeros_(self.proj.bias)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.view(self.vectors, self.channels, self.d_latent)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        if self.use_ln:
            h = self.norm(h)
        elif self.learned_scale is not None:
            h = h * self.gain.view(1, 1, -1)
        h = h + b_vc
        return h


class ChannelsParamsProj(ParamsProj):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        d_latent: int,
        init_scale: float = 1.0,
        learned_scale: Optional[float] = None,
        use_ln: bool = False,
    ):
        super().__init__(device=device, param_shapes=param_shapes, d_latent=d_latent)
        self.param_shapes = param_shapes
        self.projections = nn.ModuleDict({})
        self.flat_shapes = flatten_param_shapes(param_shapes)
        self.learned_scale = learned_scale
        self.use_ln = use_ln
        for k, (vectors, channels) in self.flat_shapes.items():
            self.projections[_sanitize_name(k)] = ChannelsProj(
                device=device,
                vectors=vectors,
                channels=channels,
                d_latent=d_latent,
                init_scale=init_scale,
                learned_scale=learned_scale,
                use_ln=use_ln,
            )

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        out = AttrDict()
        start = 0
        for k, shape in self.param_shapes.items():
            vectors, _ = self.flat_shapes[k]
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
            start = end
        return out


def params_proj_from_config(
    config: Dict[str, Any],
    device: torch.device,
    param_shapes: Dict[str, Tuple[int]],
    d_latent: int,
):
    name = config.pop("name")
    if name == "linear":
        return LinearParamsProj(
            **config, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
    elif name == "mlp":
        return MLPParamsProj(
            **config, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
    elif name == "channels":
        return ChannelsParamsProj(
            **config, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
    else:
        raise ValueError(f"unknown params proj: {name}")


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")


class LatentBottleneck(nn.Module, ABC):
    def __init__(self, *, device: torch.device, d_latent: int):
        super().__init__()
        self.device = device
        self.d_latent = d_latent

    @abstractmethod
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class IdentityLatentBottleneck(LatentBottleneck):
    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x


class ClampBottleneck(LatentBottleneck):
    def __init__(self, *, device: torch.device, d_latent: int):
        super().__init__(device=device, d_latent=d_latent)

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        x = x.tanh()
        return x


class LayerNormBottleneck(LatentBottleneck):
    def __init__(self, *, device: torch.device, d_latent: int):
        super().__init__(device=device, d_latent=d_latent)
        self.ln = nn.LayerNorm(d_latent**2).cuda()

    def forward(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        x = self.ln(x)
        return x


def latent_bottleneck_from_config(
    config: Dict[str, Any], device: torch.device, d_latent: int
):
    name = config.pop("name")
    if name == "clamp":
        return ClampBottleneck(**config, device=device, d_latent=d_latent)
    elif name == "identity":
        return IdentityLatentBottleneck(**config, device=device, d_latent=d_latent)
    elif name == "layer_norm":
        return LayerNormBottleneck(**config, device=device, d_latent=d_latent)
    else:
        raise ValueError(f"unknown latent bottleneck: {name}")


class LatentWarp(nn.Module, ABC):
    def __init__(self, *, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def warp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass

    @abstractmethod
    def unwarp(self, x: torch.Tensor, options: Optional[AttrDict] = None) -> AttrDict:
        pass


class IdentityLatentWarp(LatentWarp):
    def warp(self, x, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x

    def unwarp(self, x, options: Optional[AttrDict] = None) -> AttrDict:
        _ = options
        return x


def latent_warp_from_config(config: Dict[str, Any], device: torch.device):
    name = config.pop("name")
    if name == "identity":
        return IdentityLatentWarp(**config, device=device)
    else:
        raise ValueError(f"unknown latent warping function: {name}")


class Encoder(nn.Module, ABC):
    def __init__(self, *, device: torch.device, param_shapes: Dict[str, Tuple[int]]):
        """
        Instantiate the encoder with information about the renderer's input
        parameters. This information can be used to create output layers to
        generate the necessary latents.
        """
        super().__init__()
        self.param_shapes = param_shapes
        self.device = device

    @abstractmethod
    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        """
        Encode a batch of data into a batch of latent information.
        """


class VectorEncoder(Encoder):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        d_latent: int,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(device=device, param_shapes=param_shapes)
        if latent_bottleneck is None:
            latent_bottleneck = dict(name="identity")
        if latent_warp is None:
            latent_warp = dict(name="identity")
        self.d_latent = d_latent
        self.params_proj = params_proj_from_config(
            params_proj, device=device, param_shapes=param_shapes, d_latent=d_latent
        )
        self.latent_bottleneck = latent_bottleneck_from_config(
            latent_bottleneck, device=device, d_latent=d_latent
        )
        self.latent_warp = latent_warp_from_config(latent_warp, device=device)

    def forward(self, batch: AttrDict, options: Optional[AttrDict] = None) -> AttrDict:
        h = self.encode_to_bottleneck(batch, options=options)
        return self.bottleneck_to_params(h, options=options)

    def encode_to_bottleneck(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        return self.latent_warp.warp(
            self.latent_bottleneck(
                self.encode_to_vector(batch, options=options), options=options
            ),
            options=options,
        )

    @abstractmethod
    def encode_to_vector(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        """
        Encode the batch into a single latent vector.
        """

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(
            self.latent_warp.unwarp(vector, options=options), options=options
        )


class ChannelsEncoder(VectorEncoder):
    def __init__(
        self,
        *,
        device: torch.device,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        d_latent: int,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            d_latent=d_latent,
            latent_bottleneck=latent_bottleneck,
            latent_warp=latent_warp,
        )
        self.flat_shapes = flatten_param_shapes(param_shapes)
        self.latent_ctx = sum(flat[0] for flat in self.flat_shapes.values())

    @abstractmethod
    def encode_to_channels(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        """
        Encode the batch into a per-data-point set of latents.
        :return: [batch_size, latent_ctx, latent_width]
        """

    def encode_to_vector(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        return self.encode_to_channels(batch, options=options).flatten(1)

    def bottleneck_to_channels(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        _ = options
        return vector.view(vector.shape[0], self.latent_ctx, -1)

    def bottleneck_to_params(
        self, vector: torch.Tensor, options: Optional[AttrDict] = None
    ) -> AttrDict:
        _ = options
        return self.params_proj(
            self.bottleneck_to_channels(self.latent_warp.unwarp(vector)),
            options=options,
        )


@dataclass
class DatasetIterator:
    embs: torch.Tensor  # [batch_size, dataset_size, *shape]
    batch_size: int

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        _outer_batch_size, dataset_size, *_shape = self.embs.shape

        while True:
            start = self.idx
            self.idx += self.batch_size
            end = self.idx
            if end <= dataset_size:
                break
            self._reset()

        return self.embs[:, start:end]

    def _reset(self):
        self._shuffle()
        self.idx = 0  # pylint: disable=attribute-defined-outside-init

    def _shuffle(self):
        outer_batch_size, dataset_size, *shape = self.embs.shape
        idx = torch.stack(
            [
                torch.randperm(dataset_size, device=self.embs.device)
                for _ in range(outer_batch_size)
            ],
            dim=0,
        )
        idx = idx.view(outer_batch_size, dataset_size, *([1] * len(shape)))
        idx = torch.broadcast_to(idx, self.embs.shape)
        self.embs = torch.gather(self.embs, 1, idx)


class PerceiverChannelsEncoder(ChannelsEncoder, ABC):
    """
    Encode point clouds using a perceiver model with an extra output
    token used to extract a latent vector.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        param_shapes: Dict[str, Tuple[int]],
        params_proj: Dict[str, Any],
        min_unrolls: int,
        max_unrolls: int,
        d_latent: int = 512,
        latent_bottleneck: Optional[Dict[str, Any]] = None,
        latent_warp: Optional[Dict[str, Any]] = None,
        width: int = 512,
        layers: int = 12,
        xattn_layers: int = 1,
        heads: int = 8,
        init_scale: float = 0.25,
        # Training hparams
        inner_batch_size: Union[int, List[int]] = 1,
        data_ctx: int = 1,
    ):
        super().__init__(
            device=device,
            param_shapes=param_shapes,
            params_proj=params_proj,
            d_latent=d_latent,
            latent_bottleneck=latent_bottleneck,
            latent_warp=latent_warp,
        )
        self.width = width
        self.device = device
        self.dtype = dtype

        if isinstance(inner_batch_size, int):
            inner_batch_size = [inner_batch_size]
        self.inner_batch_size = inner_batch_size
        self.data_ctx = data_ctx
        self.min_unrolls = min_unrolls
        self.max_unrolls = max_unrolls

        encoder_fn = lambda inner_batch_size: SimplePerceiver(
            device=device,
            dtype=dtype,
            n_ctx=self.data_ctx + self.latent_ctx,
            n_data=inner_batch_size,
            width=width,
            layers=xattn_layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.encoder = (
            encoder_fn(self.inner_batch_size[0])
            if len(self.inner_batch_size) == 1
            else nn.ModuleList(
                [encoder_fn(inner_bsz) for inner_bsz in self.inner_batch_size]
            )
        )
        self.processor = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=self.data_ctx + self.latent_ctx,
            layers=layers - xattn_layers,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.register_parameter(
            "output_tokens",
            nn.Parameter(
                torch.randn(self.latent_ctx, width, device=device, dtype=dtype)
            ),
        )
        self.output_proj = nn.Linear(width, d_latent, device=device, dtype=dtype)

    @abstractmethod
    def get_h_and_iterator(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Tuple[torch.Tensor, Iterable[Union[torch.Tensor, Tuple]]]:
        """
        :return: a tuple of (
            the initial output tokens of size [batch_size, data_ctx + latent_ctx, width],
            an iterator over the given data
        )
        """

    def encode_to_channels(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> torch.Tensor:
        h, it = self.get_h_and_iterator(batch, options=options)
        n_unrolls = self.get_n_unrolls()

        for _ in range(n_unrolls):
            data = next(it)
            if isinstance(data, tuple):
                for data_i, encoder_i in zip(data, self.encoder):
                    h = encoder_i(h, data_i)
            else:
                h = self.encoder(h, data)
            h = self.processor(h)

        h = self.output_proj(self.ln_post(h[:, -self.latent_ctx :]))
        return h

    def get_n_unrolls(self):
        if self.training:
            n_unrolls = torch.randint(
                self.min_unrolls, self.max_unrolls + 1, size=(), device=self.device
            )
            # dist.broadcast(n_unrolls, 0)
            n_unrolls = n_unrolls.item()
        else:
            n_unrolls = self.max_unrolls
        return n_unrolls


class PointCloudPerceiverChannelsEncoder(PerceiverChannelsEncoder):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(
        self,
        *,
        cross_attention_dataset: str = "pcl",
        fps_method: str = "fps",
        # point cloud hyperparameters
        input_channels: int = 6,
        pos_emb: Optional[str] = None,
        # multiview hyperparameters
        image_size: int = 256,
        patch_size: int = 32,
        pose_dropout: float = 0.0,
        use_depth: bool = False,
        max_depth: float = 5.0,
        # point conv hyperparameters
        pointconv_radius: float = 0.5,
        pointconv_samples: int = 32,
        pointconv_hidden: Optional[List[int]] = None,
        pointconv_patch_size: int = 1,
        pointconv_stride: int = 1,
        pointconv_padding_mode: str = "zeros",
        use_pointconv: bool = False,
        # other hyperparameters
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert cross_attention_dataset in ("pcl_and_multiview_pcl",)
        assert fps_method in ("fps", "first")
        self.cross_attention_dataset = cross_attention_dataset
        self.fps_method = fps_method
        self.input_channels = input_channels
        self.input_proj = PosEmbLinear(
            pos_emb,
            input_channels,
            self.width,
            device=self.device,
            dtype=self.dtype,
        )
        self.use_pointconv = use_pointconv
        if use_pointconv:
            if pointconv_hidden is None:
                pointconv_hidden = [self.width]
            self.point_conv = PointSetEmbedding(
                n_point=self.data_ctx,
                radius=pointconv_radius,
                n_sample=pointconv_samples,
                d_input=self.input_proj.weight.shape[0],
                d_hidden=pointconv_hidden,
                patch_size=pointconv_patch_size,
                stride=pointconv_stride,
                padding_mode=pointconv_padding_mode,
                fps_method=fps_method,
                device=self.device,
                dtype=self.dtype,
            )
        if self.cross_attention_dataset == "pcl_and_multiview_pcl":
            self.view_pose_width = self.width // 2
            self.image_size = image_size
            self.patch_size = patch_size
            self.max_depth = max_depth
            assert use_depth
            self.mv_pcl_embed = MultiviewPointCloudEmbedding(
                posemb_version="nerf",
                n_channels=3,
                out_features=self.view_pose_width,
                device=self.device,
                dtype=self.dtype,
            )
            self.patch_emb = nn.Conv2d(
                in_channels=self.view_pose_width,
                out_channels=self.width,
                kernel_size=patch_size,
                stride=patch_size,
                device=self.device,
                dtype=self.dtype,
            )

    def get_h_and_iterator(
        self, batch: AttrDict, options: Optional[AttrDict] = None
    ) -> Tuple[torch.Tensor, Iterable]:
        """
        :return: a tuple of (
            the initial output tokens of size [batch_size, data_ctx + latent_ctx, width],
            an iterator over the given data
        )
        """
        options = AttrDict() if options is None else options

        # Build the initial query embeddings
        points = batch.points.permute(0, 2, 1)  # NCL -> NLC
        if self.use_pointconv:
            points = self.input_proj(points).permute(0, 2, 1)  # NLC -> NCL
            xyz = batch.points[:, :3]
            data_tokens = self.point_conv(xyz, points).permute(0, 2, 1)  # NCL -> NLC
        else:
            fps_samples = self.sample_pcl_fps(points)
            data_tokens = self.input_proj(fps_samples)
        batch_size = points.shape[0]
        latent_tokens = self.output_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.data_ctx + self.latent_ctx, self.width)

        # Build the dataset embedding iterator
        dataset_fn = {
            "pcl_and_multiview_pcl": self.get_pcl_and_multiview_pcl_dataset,
        }[self.cross_attention_dataset]
        it = dataset_fn(batch, options=options)

        return h, it

    def get_pcl_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict[str, Any]] = None,
        inner_batch_size: Optional[int] = None,
    ) -> Iterable:
        _ = options
        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size[0]
        points = batch.points.permute(0, 2, 1)  # NCL -> NLC
        dataset_emb = self.input_proj(points)
        assert dataset_emb.shape[1] >= inner_batch_size
        return iter(DatasetIterator(dataset_emb, batch_size=inner_batch_size))

    def get_pcl_and_multiview_pcl_dataset(
        self,
        batch: AttrDict,
        options: Optional[AttrDict] = None,
        use_distance: bool = True,
    ) -> Iterable:
        _ = options

        pcl_it = self.get_pcl_dataset(
            batch, options=options, inner_batch_size=self.inner_batch_size[0]
        )
        multiview_pcl_emb = self.encode_multiview_pcl(batch, use_distance=use_distance)
        batch_size, num_views, n_patches, width = multiview_pcl_emb.shape

        assert num_views >= self.inner_batch_size[1]

        multiview_pcl_it = iter(
            DatasetIterator(multiview_pcl_emb, batch_size=self.inner_batch_size[1])
        )

        def gen():
            while True:
                pcl = next(pcl_it)
                multiview_pcl = next(multiview_pcl_it)
                assert multiview_pcl.shape == (
                    batch_size,
                    self.inner_batch_size[1],
                    n_patches,
                    self.width,
                )
                yield pcl, multiview_pcl.reshape(batch_size, -1, width)

        return gen()

    def encode_multiview_pcl(
        self, batch: AttrDict, use_distance: bool = True
    ) -> torch.Tensor:
        """
        :return: [batch_size, num_views, n_patches, width]
        """
        all_views = self.views_to_tensor(batch.views).to(self.device)
        depths = self.raw_depths_to_tensor(batch.depths)
        all_view_alphas = self.view_alphas_to_tensor(batch.view_alphas).to(self.device)
        mask = all_view_alphas >= 0.999

        dense_poses, camera_z = self.dense_pose_cameras_to_tensor(batch.cameras)
        dense_poses = dense_poses.permute(0, 1, 4, 5, 2, 3)

        origin, direction = dense_poses[:, :, 0], dense_poses[:, :, 1]
        if use_distance:
            ray_depth_factor = torch.sum(
                direction * camera_z[..., None, None], dim=2, keepdim=True
            )
            depths = depths / ray_depth_factor
        position = origin + depths * direction
        all_view_poses = self.mv_pcl_embed(all_views, origin, position, mask)

        batch_size, num_views, _, _, _ = all_view_poses.shape

        views_proj = self.patch_emb(
            all_view_poses.reshape([batch_size * num_views, *all_view_poses.shape[2:]])
        )
        views_proj = (
            views_proj.reshape([batch_size, num_views, self.width, -1])
            .permute(0, 1, 3, 2)
            .contiguous()
        )  # [batch_size x num_views x n_patches x width]

        return views_proj

    def views_to_tensor(
        self, views: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 3 x size x size] tensor in the range [-1, 1].
        """
        if isinstance(views, torch.Tensor):
            return views

        tensor_batch = []
        num_views = len(views[0])
        for inner_list in views:
            assert len(inner_list) == num_views
            inner_batch = []
            for img in inner_list:
                img = img.resize((self.image_size,) * 2).convert("RGB")
                inner_batch.append(
                    torch.from_numpy(np.array(img)).to(
                        device=self.device, dtype=torch.float32
                    )
                    / 127.5
                    - 1
                )
            tensor_batch.append(torch.stack(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0).permute(0, 1, 4, 2, 3)

    def view_alphas_to_tensor(
        self, view_alphas: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 1 x size x size] tensor in the range [0, 1].
        """
        if isinstance(view_alphas, torch.Tensor):
            return view_alphas

        tensor_batch = []
        num_views = len(view_alphas[0])
        for inner_list in view_alphas:
            assert len(inner_list) == num_views
            inner_batch = []
            for img in inner_list:
                tensor = (
                    torch.from_numpy(np.array(img)).to(
                        device=self.device, dtype=torch.float32
                    )
                    / 255.0
                )
                tensor = F.interpolate(
                    tensor[None, None],
                    (self.image_size,) * 2,
                    mode="nearest",
                )
                inner_batch.append(tensor)
            tensor_batch.append(torch.cat(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0)

    def raw_depths_to_tensor(
        self, depths: Union[torch.Tensor, List[List[Image.Image]]]
    ) -> torch.Tensor:
        """
        Returns a [batch x num_views x 1 x size x size] tensor
        """
        if isinstance(depths, torch.Tensor):
            return depths

        tensor_batch = []
        num_views = len(depths[0])
        for inner_list in depths:
            assert len(inner_list) == num_views
            inner_batch = []
            for arr in inner_list:
                tensor = torch.from_numpy(arr).clamp(max=self.max_depth)
                tensor = F.interpolate(
                    tensor[None, None],
                    (self.image_size,) * 2,
                    mode="nearest",
                )
                inner_batch.append(tensor.to(device=self.device, dtype=torch.float32))
            tensor_batch.append(torch.cat(inner_batch, dim=0))
        return torch.stack(tensor_batch, dim=0)

    def dense_pose_cameras_to_tensor(
        self, cameras: Union[torch.Tensor, List[List[ProjectiveCamera]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (rays, z_directions) where
            - rays: [batch, num_views, height, width, 2, 3] tensor of camera information.
            - z_directions: [batch, num_views, 3] tensor of camera z directions.
        """
        if isinstance(cameras, torch.Tensor):
            raise NotImplementedError

        for inner_list in cameras:
            assert len(inner_list) == len(cameras[0])

        camera = cameras[0][0]
        flat_camera = DifferentiableProjectiveCamera(
            origin=torch.from_numpy(
                np.stack(
                    [cam.origin for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            x=torch.from_numpy(
                np.stack(
                    [cam.x for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            y=torch.from_numpy(
                np.stack(
                    [cam.y for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            z=torch.from_numpy(
                np.stack(
                    [cam.z for inner_list in cameras for cam in inner_list],
                    axis=0,
                )
            ).to(self.device),
            width=camera.width,
            height=camera.height,
            x_fov=camera.x_fov,
            y_fov=camera.y_fov,
        )
        batch_size = len(cameras) * len(cameras[0])
        coords = (
            flat_camera.image_coords()
            .to(flat_camera.origin.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        rays = flat_camera.camera_rays(coords)
        return (
            rays.view(
                len(cameras), len(cameras[0]), camera.height, camera.width, 2, 3
            ).to(self.device),
            flat_camera.z.view(len(cameras), len(cameras[0]), 3).to(self.device),
        )
