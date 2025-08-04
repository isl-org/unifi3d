"""
Meta-learning modules based on: https://github.com/tristandeleu/pytorch-meta

MIT License

Copyright (c) 2019-2020 Tristan Deleu

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

import itertools
import math
import re
from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unifi3d.utils.model.shap_e_utils import AttrDict


def subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ""):
        return dictionary
    key_re = re.compile(r"^{0}\.(.+)".format(re.escape(key)))
    return AttrDict(
        OrderedDict(
            (key_re.sub(r"\1", k), value)
            for (k, value) in dictionary.items()
            if key_re.match(k) is not None
        )
    )


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).

    Based on SIREN's torchmeta with some additional features/changes.

    All meta weights must not have the batch dimension, as they are later tiled
    to the given batch size after unsqueezing the first dimension (e.g. a
    weight of dimension [d_out x d_in] is tiled to have the dimension [batch x
    d_out x d_in]).  Requiring all meta weights to have a batch dimension of 1
    (e.g. [1 x d_out x d_in] from the earlier example) could be a more natural
    choice, but this results in silent failures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta_state_dict = set()
        self._meta_params = set()

    def register_meta_buffer(self, name: str, param: nn.Parameter):
        """
        Registers a trainable or nontrainable parameter as a meta buffer. This
        can be later retrieved by meta_state_dict
        """
        self.register_buffer(name, param)
        self._meta_state_dict.add(name)

    def register_meta_parameter(self, name: str, parameter: nn.Parameter):
        """
        Registers a meta parameter so it is included in named_meta_parameters
        and meta_state_dict.
        """
        self.register_parameter(name, parameter)
        self._meta_params.add(name)
        self._meta_state_dict.add(name)

    def register_meta(self, name: str, parameter: nn.Parameter, trainable: bool = True):
        if trainable:
            self.register_meta_parameter(name, parameter)
        else:
            self.register_meta_buffer(name, parameter)

    def register(
        self, name: str, parameter: nn.Parameter, meta: bool, trainable: bool = True
    ):
        if meta:
            if trainable:
                self.register_meta_parameter(name, parameter)
            else:
                self.register_meta_buffer(name, parameter)
        else:
            if trainable:
                self.register_parameter(name, parameter)
            else:
                self.register_buffer(name, parameter)

    def named_meta_parameters(self, prefix="", recurse=True):
        """
        Returns an iterator over all the names and meta parameters
        """

        def meta_iterator(module):
            meta = module._meta_params if isinstance(module, MetaModule) else set()
            for name, param in module._parameters.items():
                if name in meta:
                    yield name, param

        gen = self._named_members(
            meta_iterator,
            prefix=prefix,
            recurse=recurse,
        )
        for name, param in gen:
            yield name, param

    def named_nonmeta_parameters(self, prefix="", recurse=True):
        def _iterator(module):
            meta = module._meta_params if isinstance(module, MetaModule) else set()
            for name, param in module._parameters.items():
                if name not in meta:
                    yield name, param

        gen = self._named_members(
            _iterator,
            prefix=prefix,
            recurse=recurse,
        )
        for name, param in gen:
            yield name, param

    def nonmeta_parameters(self, prefix="", recurse=True):
        for _, param in self.named_nonmeta_parameters(prefix=prefix, recurse=recurse):
            yield param

    def meta_state_dict(self, prefix="", recurse=True):
        """
        Returns an iterator over all the names and meta parameters/buffers.

        One difference between module.state_dict() is that this preserves
        requires_grad, because we may want to compute the gradient w.r.t. meta
        buffers, but don't necessarily update them automatically.
        """

        def meta_iterator(module):
            meta = module._meta_state_dict if isinstance(module, MetaModule) else set()
            for name, param in itertools.chain(
                module._buffers.items(), module._parameters.items()
            ):
                if name in meta:
                    yield name, param

        gen = self._named_members(
            meta_iterator,
            prefix=prefix,
            recurse=recurse,
        )
        return dict(gen)

    def update(self, params=None):
        """
        Updates the parameter list before the forward prop so that if `params`
        is None or doesn't have a certain key, the module uses the default
        parameter/buffer registered in the module.
        """
        if params is None:
            params = AttrDict()
        params = AttrDict(params)
        named_params = set([name for name, _ in self.named_parameters()])
        for name, param in self.named_parameters():
            params.setdefault(name, param)
        for name, param in self.state_dict().items():
            if name not in named_params:
                params.setdefault(name, param)
        return params


def batch_meta_state_dict(net, batch_size):
    state_dict = AttrDict()
    meta_parameters = set([name for name, _ in net.named_meta_parameters()])
    for name, param in net.meta_state_dict().items():
        state_dict[name] = (
            param.clone().unsqueeze(0).repeat(batch_size, *[1] * len(param.shape))
        )
    return state_dict


def encode_position(version: str, *, position: torch.Tensor):
    if version == "v1":
        freqs = get_scales(0, 10, position.dtype, position.device).view(1, -1)
        freqs = position.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(
            *position.shape[:-1], -1
        )
    elif version == "nerf":
        return posenc_nerf(position, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def encode_channels(version: str, *, channels: torch.Tensor):
    if version == "v1":
        freqs = get_scales(0, 10, channels.dtype, channels.device).view(1, -1)
        freqs = channels.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(
            *channels.shape[:-1], -1
        )
    elif version == "nerf":
        return posenc_nerf(channels, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def position_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_position(version, position=torch.zeros(1, 1)).shape[-1]


def channel_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_channels(version, channels=torch.zeros(1, 1)).shape[-1]


class PosEmbLinear(nn.Linear):
    def __init__(
        self,
        posemb_version: Optional[str],
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        super().__init__(
            in_features * position_encoding_channels(posemb_version),
            out_features,
            **kwargs,
        )
        self.posemb_version = posemb_version

    def forward(self, x: torch.Tensor):
        if self.posemb_version is not None:
            x = encode_position(self.posemb_version, position=x)
        return super().forward(x)


class MultiviewPointCloudEmbedding(nn.Conv2d):
    def __init__(
        self,
        posemb_version: Optional[str],
        n_channels: int,
        out_features: int,
        stride: int = 1,
        **kwargs,
    ):
        in_features = (
            n_channels * channel_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
        )
        super().__init__(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            padding=1,
            **kwargs,
        )
        self.posemb_version = posemb_version
        self.register_parameter(
            "unk_token", nn.Parameter(torch.randn(in_features, **kwargs) * 0.01)
        )
        self.unk_token: torch.Tensor

    def forward(
        self,
        channels: torch.Tensor,
        origin: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param channels: [batch_shape, inner_batch_shape, n_channels, height, width]
        :param origin: [batch_shape, inner_batch_shape, 3, height, width]
        :param position: [batch_shape, inner_batch_shape, 3, height, width]
        :return: [*batch_shape, out_features, height, width]
        """

        if self.posemb_version is not None:
            channels = channels.permute(0, 1, 3, 4, 2)
            origin = origin.permute(0, 1, 3, 4, 2)
            position = position.permute(0, 1, 3, 4, 2)
            channels = encode_channels(self.posemb_version, channels=channels).permute(
                0, 1, 4, 2, 3
            )
            origin = encode_position(self.posemb_version, position=origin).permute(
                0, 1, 4, 2, 3
            )
            position = encode_position(self.posemb_version, position=position).permute(
                0, 1, 4, 2, 3
            )
        x = torch.cat([channels, origin, position], dim=-3)
        unk_token = torch.broadcast_to(self.unk_token.view(1, 1, -1, 1, 1), x.shape)
        x = torch.where(mask, x, unk_token)
        *batch_shape, in_features, height, width = x.shape
        return (
            super()
            .forward(x.view(-1, in_features, height, width))
            .view(*batch_shape, -1, height, width)
        )


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = get_scales(min_deg, max_deg, x.dtype, x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


@lru_cache
def get_scales(
    min_deg: int,
    max_deg: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return 2.0 ** torch.arange(min_deg, max_deg, device=device, dtype=dtype)


def maybe_encode_direction(
    version: str,
    *,
    position: torch.Tensor,
    direction: Optional[torch.Tensor] = None,
):
    if version == "v1":
        sh_degree = 4
        if direction is None:
            return torch.zeros(*position.shape[:-1], sh_degree**2).to(position)
        return spherical_harmonics_basis(direction, sh_degree=sh_degree)
    elif version == "nerf":
        if direction is None:
            return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
        return posenc_nerf(direction, min_deg=0, max_deg=8)
    else:
        raise ValueError(version)


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def swish(x):
    return x * torch.sigmoid(x)


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def torch_gelu(x):
    return torch.nn.functional.gelu(x)


def geglu(x):
    v, gates = x.chunk(2, dim=-1)
    return v * gelu(gates)


class SirenSin:
    def __init__(self, w0=30.0):
        self.w0 = w0

    def __call__(self, x):
        return torch.sin(self.w0 * x)


def get_act(name):
    return {
        "relu": torch.nn.functional.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "swish": swish,
        "tanh": torch.tanh,
        "gelu": gelu,
        "quick_gelu": quick_gelu,
        "torch_gelu": torch_gelu,
        "gelu2": quick_gelu,
        "geglu": geglu,
        "sigmoid": torch.sigmoid,
        "sin": torch.sin,
        "sin30": SirenSin(w0=30.0),
        "softplus": F.softplus,
        "exp": torch.exp,
        "identity": lambda x: x,
    }[name]


def std_init(affine, init_scale: float = 1.0):
    n_in = affine.weight.shape[1]
    stddev = init_scale / math.sqrt(n_in)
    nn.init.normal_(affine.weight, std=stddev)
    if affine.bias is not None:
        nn.init.constant_(affine.bias, 0.0)


def mlp_init(affines, init: Optional[str] = None, init_scale: float = 1.0):
    if init == "siren30":
        for idx, affine in enumerate(affines):
            init = siren_init_first_layer if idx == 0 else siren_init_30
            init(affine, init_scale=init_scale)
    elif init == "siren":
        for idx, affine in enumerate(affines):
            init = siren_init_first_layer if idx == 0 else siren_init
            init(affine, init_scale=init_scale)
    elif init is None:
        for affine in affines:
            std_init(affine, init_scale=init_scale)
    else:
        raise NotImplementedError(init)


class MetaLinear(MetaModule):
    def __init__(
        self,
        n_in,
        n_out,
        bias: bool = True,
        meta_scale: bool = True,
        meta_shift: bool = True,
        meta_proj: bool = False,
        meta_bias: bool = False,
        trainable_meta: bool = False,
        **kwargs,
    ):
        super().__init__()
        # n_in, n_out, bias=bias)
        register_meta_fn = (
            self.register_meta_parameter
            if trainable_meta
            else self.register_meta_buffer
        )
        if meta_scale:
            register_meta_fn("scale", nn.Parameter(torch.ones(n_out, **kwargs)))
        if meta_shift:
            register_meta_fn("shift", nn.Parameter(torch.zeros(n_out, **kwargs)))

        register_proj_fn = (
            self.register_parameter if not meta_proj else register_meta_fn
        )
        register_proj_fn("weight", nn.Parameter(torch.empty((n_out, n_in), **kwargs)))

        if not bias:
            self.register_parameter("bias", None)
        else:
            register_bias_fn = (
                self.register_parameter if not meta_bias else register_meta_fn
            )
            register_bias_fn("bias", nn.Parameter(torch.empty(n_out, **kwargs)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _bcast(self, op, left, right):
        if right.ndim == 2:
            # Has dimension [batch x d_output]
            right = right.unsqueeze(1)
        return op(left, right)

    def forward(self, x, params=None):
        params = self.update(params)

        batch_size, *shape, d_in = x.shape
        x = x.view(batch_size, -1, d_in)

        if params.weight.ndim == 2:
            h = torch.einsum("bni,oi->bno", x, params.weight)
        elif params.weight.ndim == 3:
            h = torch.einsum("bni,boi->bno", x, params.weight)

        if params.bias is not None:
            h = self._bcast(torch.add, h, params.bias)

        if params.scale is not None:
            h = self._bcast(torch.mul, h, params.scale)

        if params.shift is not None:
            h = self._bcast(torch.add, h, params.shift)

        h = h.view(batch_size, *shape, -1)
        return h


class PointSetEmbedding(nn.Module):
    def __init__(
        self,
        *,
        radius: float,
        n_point: int,
        n_sample: int,
        d_input: int,
        d_hidden: List[int],
        patch_size: int = 1,
        stride: int = 1,
        activation: str = "swish",
        group_all: bool = False,
        padding_mode: str = "zeros",
        fps_method: str = "fps",
        **kwargs,
    ):
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        self.act = get_act(activation)
        self.patch_size = patch_size
        self.stride = stride
        last_channel = d_input + 3
        for out_channel in d_hidden:
            self.mlp_convs.append(
                nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=(patch_size, 1),
                    stride=(stride, 1),
                    padding=(patch_size // 2, 0),
                    padding_mode=padding_mode,
                    **kwargs,
                )
            )
            last_channel = out_channel
        self.group_all = group_all
        self.fps_method = fps_method

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, d_hidden[-1], n_point]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.n_point,
                self.radius,
                self.n_sample,
                xyz,
                points,
                deterministic=not self.training,
                fps_method=self.fps_method,
            )
        # new_xyz: sampled points position data, [B, n_point, C]
        # new_points: sampled points data, [B, n_point, n_sample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, n_sample, n_point]
        for i, conv in enumerate(self.mlp_convs):
            new_points = self.act(self.apply_conv(new_points, conv))

        new_points = new_points.mean(dim=2)
        return new_points

    def apply_conv(self, points: torch.Tensor, conv: nn.Module):
        batch, channels, n_samples, _ = points.shape
        # Shuffle the representations
        if self.patch_size > 1:
            # TODO shuffle deterministically when not self.training
            _, indices = torch.rand(
                batch, channels, n_samples, 1, device=points.device
            ).sort(dim=2)
            points = torch.gather(points, 2, torch.broadcast_to(indices, points.shape))
        return conv(points)


"""
Based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py

MIT License

Copyright (c) 2019 benny

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


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint,
    radius,
    nsample,
    xyz,
    points,
    returnfps=False,
    deterministic=False,
    fps_method: str = "fps",
):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if fps_method == "fps":
        fps_idx = farthest_point_sample(
            xyz, npoint, deterministic=deterministic
        )  # [B, npoint, C]
    elif fps_method == "first":
        fps_idx = torch.arange(npoint)[None].repeat(B, 1)
    else:
        raise ValueError(f"Unknown FPS method: {fps_method}")
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
