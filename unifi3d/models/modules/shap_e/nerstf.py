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
from typing import Any, Dict, Optional, Tuple
from functools import partial
import numpy as np

import torch
import torch.nn as nn

from unifi3d.models.modules.shap_e.nn import MetaModule, MetaLinear, mlp_init, get_act
from unifi3d.models.modules.shap_e.nn import (
    encode_position,
    maybe_encode_direction,
    subdict,
)
from unifi3d.utils.model.shap_e_utils import (
    ArrayType,
    AttrDict,
    Query,
    append_tensor,
    checkpoint,
)


class Model(ABC):
    @abstractmethod
    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        """
        Predict an attribute given position
        """

    def forward_batched(
        self,
        query: Query,
        query_batch_size: int = 4096,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        if not query.position.numel():
            # Avoid torch.cat() of zero tensors.
            return self(query, params=params, options=options)

        if options.cache is None:
            created_cache = True
            options.cache = AttrDict()
        else:
            created_cache = False

        results_list = AttrDict()
        for i in range(0, query.position.shape[1], query_batch_size):
            out = self(
                query=query.map_tensors(lambda x, i=i: x[:, i : i + query_batch_size]),
                params=params,
                options=options,
            )
            results_list = results_list.combine(out, append_tensor)

        if created_cache:
            del options["cache"]

        return results_list.map(lambda key, tensor_list: torch.cat(tensor_list, dim=1))


class MLPModel(MetaModule, Model):
    def __init__(
        self,
        n_output: int,
        output_activation: str,
        # Positional encoding parameters
        posenc_version: str = "v1",
        # Direction related channel prediction
        insert_direction_at: Optional[int] = None,
        # MLP parameters
        d_hidden: int = 256,
        n_hidden_layers: int = 4,
        activation: str = "relu",
        init: Optional[str] = None,
        init_scale: float = 1.0,
        meta_parameters: bool = False,
        trainable_meta: bool = False,
        meta_proj: bool = True,
        meta_bias: bool = True,
        meta_start: int = 0,
        meta_stop: Optional[int] = None,
        n_meta_layers: Optional[int] = None,
        register_freqs: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        if register_freqs:
            self.register_buffer(
                "freqs", 2.0 ** torch.arange(10, device=device).view(1, 10)
            )

        # Positional encoding
        self.posenc_version = posenc_version
        dummy = torch.eye(1, 3)
        d_posenc_pos = encode_position(posenc_version, position=dummy).shape[-1]
        d_posenc_dir = maybe_encode_direction(posenc_version, position=dummy).shape[-1]

        # Instantiate the MLP
        mlp_widths = [d_hidden] * n_hidden_layers
        input_widths = [d_posenc_pos, *mlp_widths]
        output_widths = mlp_widths + [n_output]

        self.meta_parameters = meta_parameters

        # When this model is used jointly to express NeRF, it may have to
        # process directions as well in which case we simply concatenate
        # the direction representation at the specified layer.
        self.insert_direction_at = insert_direction_at
        if insert_direction_at is not None:
            input_widths[self.insert_direction_at] += d_posenc_dir

        linear_cls = lambda meta: (
            partial(
                MetaLinear,
                meta_scale=False,
                meta_shift=False,
                meta_proj=meta_proj,
                meta_bias=meta_bias,
                trainable_meta=trainable_meta,
            )
            if meta
            else nn.Linear
        )

        if meta_stop is None:
            if n_meta_layers is not None:
                assert n_meta_layers > 0
                meta_stop = meta_start + n_meta_layers - 1
            else:
                meta_stop = n_hidden_layers

        if meta_parameters:
            metas = [
                meta_start <= layer <= meta_stop for layer in range(n_hidden_layers + 1)
            ]
        else:
            metas = [False] * (n_hidden_layers + 1)

        self.mlp = nn.ModuleList(
            [
                linear_cls(meta)(d_in, d_out, device=device)
                for meta, d_in, d_out in zip(metas, input_widths, output_widths)
            ]
        )

        mlp_init(self.mlp, init=init, init_scale=init_scale)

        self.activation = get_act(activation)
        self.output_activation = get_act(output_activation)

        self.device = device
        self.to(device)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        """
        :param position: [batch_size x ... x 3]
        :param params: Meta parameters
        :param options: Optional hyperparameters
        """

        # query.direction is None typically for SDF models and training
        h_final, _h_directionless = self._mlp(
            query.position, query.direction, params=params, options=options
        )
        return self.output_activation(h_final)

    def _run_mlp(
        self,
        position: torch.Tensor,
        direction: torch.Tensor,
        params: AttrDict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: the final and directionless activations at the given query
        """
        h_preact = h = encode_position(self.posenc_version, position=position)
        h_directionless = None
        for i, layer in enumerate(self.mlp):
            if i == self.insert_direction_at:
                h_directionless = h_preact
                h_direction = maybe_encode_direction(
                    self.posenc_version, position=position, direction=direction
                )
                h = torch.cat([h, h_direction], dim=-1)
            if isinstance(layer, MetaLinear):
                h = layer(h, params=subdict(params, f"mlp.{i}"))
            else:
                h = layer(h)
            h_preact = h
            if i < len(self.mlp) - 1:
                h = self.activation(h)
        h_final = h
        if h_directionless is None:
            h_directionless = h_preact
        return h_final, h_directionless

    def _mlp(
        self,
        position: torch.Tensor,
        direction: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param position: [batch_size x ... x 3]
        :param params: Meta parameters
        :param options: Optional hyperparameters
        :return: the final and directionless activations at the given query
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        mlp = partial(self._run_mlp, direction=direction, params=params)
        parameters = []
        for i, layer in enumerate(self.mlp):
            if isinstance(layer, MetaLinear):
                parameters.extend(list(subdict(params, f"mlp.{i}").values()))
            else:
                parameters.extend(layer.parameters())

        h_final, h_directionless = checkpoint(
            mlp, (position,), parameters, options.checkpoint_stf_model
        )

        return h_final, h_directionless


class NeRFModel(ABC):
    """
    Parametric scene representation whose outputs are integrated by NeRFRenderer
    """

    @abstractmethod
    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        """
        :param query: the points in the field to query.
        :param params: Meta parameters
        :param options: Optional hyperparameters
        :return: An AttrDict containing at least
            - density: [batch_size x ... x 1]
            - channels: [batch_size x ... x n_channels]
            - aux_losses: [batch_size x ... x 1]
        """


class VoidNeRFModel(MetaModule, NeRFModel):
    """
    Implements the default empty space model where all queries are rendered as
    background.
    """

    def __init__(
        self,
        background: ArrayType,
        trainable: bool = False,
        channel_scale: float = 255.0,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        background = nn.Parameter(
            torch.from_numpy(np.array(background)).to(
                dtype=torch.float32, device=device
            )
            / channel_scale
        )
        if trainable:
            self.register_parameter("background", background)
        else:
            self.register_buffer("background", background)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict:
        _ = params
        default_bg = self.background[None]
        background = (
            options.get("background", default_bg) if options is not None else default_bg
        )

        shape = query.position.shape[:-1]
        ones = [1] * (len(shape) - 1)
        n_channels = background.shape[-1]
        background = torch.broadcast_to(
            background.view(background.shape[0], *ones, n_channels),
            [*shape, n_channels],
        )
        return background


class MLPNeRSTFModel(MLPModel):
    def __init__(
        self,
        sdf_activation="tanh",
        density_activation="exp",
        channel_activation="sigmoid",
        direction_dependent_shape: bool = True,  # To be able to load old models. Set this to be False in future models.
        separate_nerf_channels: bool = False,
        separate_coarse_channels: bool = False,
        initial_density_bias: float = 0.0,
        initial_sdf_bias: float = -0.1,
        **kwargs,
    ):
        h_map, h_directionless_map = indices_for_output_mode(
            direction_dependent_shape=direction_dependent_shape,
            separate_nerf_channels=separate_nerf_channels,
            separate_coarse_channels=separate_coarse_channels,
        )
        n_output = index_mapping_max(h_map)
        super().__init__(
            n_output=n_output,
            output_activation="identity",
            **kwargs,
        )
        self.direction_dependent_shape = direction_dependent_shape
        self.separate_nerf_channels = separate_nerf_channels
        self.separate_coarse_channels = separate_coarse_channels
        self.sdf_activation = get_act(sdf_activation)
        self.density_activation = get_act(density_activation)
        self.channel_activation = get_act(channel_activation)
        self.h_map = h_map
        self.h_directionless_map = h_directionless_map
        self.mlp[-1].bias.data.zero_()
        layer = -1 if self.direction_dependent_shape else self.insert_direction_at
        self.mlp[layer].bias[0].data.fill_(initial_sdf_bias)
        self.mlp[layer].bias[1].data.fill_(initial_density_bias)

    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        options = AttrDict() if options is None else AttrDict(options)
        h, h_directionless = self._mlp(
            query.position, query.direction, params=params, options=options
        )
        activations = map_indices_to_keys(self.h_map, h)
        activations.update(
            map_indices_to_keys(self.h_directionless_map, h_directionless)
        )

        if options.nerf_level == "coarse":
            h_density = activations.density_coarse
        else:
            h_density = activations.density_fine

        if options.get("rendering_mode", "stf") == "nerf":
            if options.nerf_level == "coarse":
                h_channels = activations.nerf_coarse
            else:
                h_channels = activations.nerf_fine
        else:
            h_channels = activations.stf
        return AttrDict(
            density=self.density_activation(h_density),
            signed_distance=self.sdf_activation(activations.sdf),
            channels=self.channel_activation(h_channels),
        )


IndexMapping = AttrDict[str, Tuple[int, int]]


def indices_for_output_mode(
    direction_dependent_shape: bool,
    separate_nerf_channels: bool,
    separate_coarse_channels: bool,
) -> Tuple[IndexMapping, IndexMapping]:
    """
    Get output mappings for (h, h_directionless).
    """
    h_map = AttrDict()
    h_directionless_map = AttrDict()
    if direction_dependent_shape:
        h_map.sdf = (0, 1)
        if separate_coarse_channels:
            assert separate_nerf_channels
            h_map.density_coarse = (1, 2)
            h_map.density_fine = (2, 3)
            h_map.stf = (3, 6)
            h_map.nerf_coarse = (6, 9)
            h_map.nerf_fine = (9, 12)
        else:
            h_map.density_coarse = (1, 2)
            h_map.density_fine = (1, 2)
            if separate_nerf_channels:
                h_map.stf = (2, 5)
                h_map.nerf_coarse = (5, 8)
                h_map.nerf_fine = (5, 8)
            else:
                h_map.stf = (2, 5)
                h_map.nerf_coarse = (2, 5)
                h_map.nerf_fine = (2, 5)
    else:
        h_directionless_map.sdf = (0, 1)
        h_directionless_map.density_coarse = (1, 2)
        if separate_coarse_channels:
            h_directionless_map.density_fine = (2, 3)
        else:
            h_directionless_map.density_fine = h_directionless_map.density_coarse
        h_map.stf = (0, 3)
        if separate_coarse_channels:
            assert separate_nerf_channels
            h_map.nerf_coarse = (3, 6)
            h_map.nerf_fine = (6, 9)
        else:
            if separate_nerf_channels:
                h_map.nerf_coarse = (3, 6)
            else:
                h_map.nerf_coarse = (0, 3)
            h_map.nerf_fine = h_map.nerf_coarse
    return h_map, h_directionless_map


def map_indices_to_keys(
    mapping: IndexMapping, data: torch.Tensor
) -> AttrDict[str, torch.Tensor]:
    return AttrDict({k: data[..., start:end] for k, (start, end) in mapping.items()})


def index_mapping_max(mapping: IndexMapping) -> int:
    return max(end for _, (_, end) in mapping.items())
