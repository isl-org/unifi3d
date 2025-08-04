# --------------------------------------------------------
# Code adapted from
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
# --------------------------------------------------------

from copy import deepcopy
from einops import rearrange
import numpy as np
import torch
from typing import Any

from unifi3d.utils.model.model_utils import load_net_from_safetensors
from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder
from unifi3d.models.modules.quantizer import VectorQuantizer


def init_weights(net, init_type="normal", gain=0.01) -> None:
    def init_func(m) -> None:
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # Propagate to children
    for m in net.children():
        m.apply(init_func)


class VAE3D(BaseEncoderDecoder):
    def __init__(self, config, ckpt_path=None, is_vae=False) -> None:
        """
        Initializes the VQVAE network with the given configuration.

        Args:
            config (Dict): Configuration dictionary for the network.
        """
        super(VAE3D, self).__init__()

        # Variables initialization
        self.config = deepcopy(config)
        self.sdf2occ_factor = self.config.get("sdf2occ_factor", -1.0)
        self.embed_dim = self.config.get("embed_dim", 1.0)
        self.n_embed = self.config.get("n_embed", 1.0)
        self.z_channels = self.config.get("z_channels", 1.0)
        self.logvar_min_threshold = np.log(self.config.get("exp_min_threshold", 1e-8))
        self.logvar_max_threshold = np.log(self.config.get("exp_max_threshold", 3e4))

        self.is_vae = is_vae

        if not is_vae:
            res_enc = self.config.get("resolution", 64) // 4
            self.layer_norm = torch.nn.LayerNorm(
                [self.embed_dim, res_enc, res_enc, res_enc]
            )

        # Network architecture definition
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": Encoder3D(**self.config),
                "decoder": Decoder3D(**self.config),
                "mu_conv": torch.nn.Conv3d(self.z_channels, self.embed_dim, 1),
                "var_conv": torch.nn.Conv3d(self.z_channels, self.embed_dim, 1),
            }
        )

        # Network weights initialization
        init_weights(self.network_dict["encoder"], "normal", 0.02)
        init_weights(self.network_dict["decoder"], "normal", 0.02)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        return

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, log_var):
        return (
            -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1).mean()
        )

    def encode(self, x):
        encoder_out = self.network_dict["encoder"](x)

        if self.is_vae:
            mu = self.network_dict["mu_conv"](encoder_out)
            log_var = self.network_dict["var_conv"](encoder_out)
            # clamp to avoid infs with log_var.exp(), especially with mixed precision
            log_var = torch.clamp(
                log_var, self.logvar_min_threshold, self.logvar_max_threshold
            )
            return mu, log_var
        else:
            return self.layer_norm(encoder_out)

    def decode(self, latent):
        out = self.network_dict["decoder"](latent)
        return out

    def encode_wrapper(self, data) -> Any:
        if self.is_vae:
            # use mean as latent
            latent, _ = self.encode(data["sdf"])
        else:
            latent = self.encode(data["sdf"])
        return latent

    def decode_wrapper(self, latent) -> Any:
        return self.decode(latent)

    def forward(self, input_pack) -> dict:
        output = {}

        sdf_in = input_pack["sdf"]

        # Encode
        if self.is_vae:
            mean, log_var = self.encode(sdf_in)
            z = self.reparameterize(mean, log_var)
        else:
            z = self.encode(sdf_in)

        if self.is_vae and self.config.get("codebook_weight", 0.1) > 0:
            kl_divergence = self.kl_divergence(mean, log_var)
            output["codebook_loss"] = kl_divergence
        else:
            output["codebook_loss"] = torch.zeros(1)

        # Decode
        sdf_hat = self.decode(z)

        output["sdf_hat"] = sdf_hat

        return output


class VQVAE(BaseEncoderDecoder):
    """
    3D Vector Quantised Variational Auto-Encoder (VQVAE) Network for encoding and decoding 3D TSDF.
    Adapted from:
    - VQVAE: https://github.com/nadavbh12/VQ-VAE
    - Encoder: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
    """

    def __init__(self, config, ckpt_path=None) -> None:
        """
        Initializes the VQVAE network with the given configuration.

        Args:
            config (Dict): Configuration dictionary for the network.
        """
        super(VQVAE, self).__init__()

        # Variables initialization
        self.config = deepcopy(config)
        self.sdf2occ_factor = self.config.get("sdf2occ_factor", -1.0)
        self.embed_dim = self.config.get("embed_dim", 1.0)
        self.n_embed = self.config.get("n_embed", 1.0)
        self.z_channels = self.config.get("z_channels", 1.0)

        # Network architecture definition
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": Encoder3D(**self.config),
                "decoder": Decoder3D(**self.config),
                "quantize": VectorQuantizer(self.n_embed, self.embed_dim, beta=1.0),
                "quant_conv": torch.nn.Conv3d(self.z_channels, self.embed_dim, 1),
                "post_quant_conv": torch.nn.Conv3d(self.embed_dim, self.z_channels, 1),
            }
        )

        # Network weights initialization
        init_weights(self.network_dict["encoder"], "normal", 0.02)
        init_weights(self.network_dict["decoder"], "normal", 0.02)
        init_weights(self.network_dict["quant_conv"], "normal", 0.02)
        init_weights(self.network_dict["post_quant_conv"], "normal", 0.02)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        return

    def load_checkpoint(self, ckpt_path):
        """
        Overwriting super method because special case for pth
        """
        if ckpt_path.endswith("pth"):
            map_fn = lambda storage, loc: storage
            state_dict = torch.load(ckpt_path, map_location=map_fn)
            self.network_dict.load_state_dict(state_dict)
            print("Loaded checkpoint from", ckpt_path)
        else:
            super().load_checkpoint(ckpt_path)

    def decode(self, quant) -> Any:
        """
        Decodes the Signed Distance Field (SDF) given query points, a latent vector, and conditioning information.

        This function applies positional encoding to the query points if required and then passes them
        through the SDF decoder network. It can return either the raw SDF values or a Bernoulli distribution
        based on the SDF values, depending on the `return_sdf` flag.

        Args:
            quant (Tensor): Conditioning information that is passed to the SDF decoder along with query points.
            return_sdf (bool): A flag to determine the return type. If True, the function returns raw SDF values.
                            If False, it returns a Bernoulli distribution parameterized by the SDF values.

        Returns:
            Tensor or Bernoulli distribution: If `return_sdf` is True, returns a tensor of SDF values.
                                            Otherwise, returns a Bernoulli distribution parameterized by the
                                            SDF values, which can be used for further probabilistic modeling.
        """
        quant = self.network_dict["post_quant_conv"](quant)
        sdf = self.network_dict["decoder"](quant)

        return sdf

    def decode_no_quant(self, h, force_not_quantize=False) -> Any:
        """
        Decodes the given tensor 'h' into a reconstructed output. Optionally bypasses the quantization step.

        Args:
            h (Tensor): The input tensor to be decoded.
            force_not_quantize (bool): If True, skips the quantization process; defaults to False.

        Returns:
            Tensor: The decoded output, post quantization (if not bypassed) and decoding.
        """
        # If quantization is not forced to be skipped, pass 'h' through the quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.network_dict["quantize"](h, is_voxel=True)
        else:
            # If bypassing quantization, use 'h' directly
            quant = h

        # Apply post-quantization convolution
        quant = self.network_dict["post_quant_conv"](quant)
        # Decode the quantized tensor
        dec = self.network_dict["decoder"](quant)

        return dec

    def decode_from_quant(self, quant_code) -> Any:
        """
        Decodes the quantized code into an embedded representation.

        Args:
            quant_code (Tensor): The quantized code to be decoded.

        Returns:
            Tensor: The decoded embedded representation from the quantized code.
        """
        # Retrieve the embedding from the quantized code
        embed_from_code = self.network_dict["quantize"].embedding(quant_code)

        return embed_from_code

    def decode_enc_idices(self, enc_indices, z_spatial_dim=8) -> Any:
        """
        Decodes encoded indices into a reconstructed output, suitable for transformer models.

        Args:
            enc_indices (Tensor): Encoded indices from the encoder.
            z_spatial_dim (int): The spatial dimension size for reshaping; defaults to 8.

        Returns:
            Tensor: The decoded output from the encoded indices.
        """
        # Reshape encoded indices for transformer compatibility
        enc_indices = rearrange(enc_indices, "t bs -> (bs t)")
        # Obtain the quantized tensor from the encoded indices
        z_q = self.network_dict["quantize"].embedding(enc_indices)
        # Reshape the quantized tensor
        z_q = rearrange(
            z_q,
            "(bs d1 d2 d3) zd -> bs zd d1 d2 d3",
            d1=z_spatial_dim,
            d2=z_spatial_dim,
            d3=z_spatial_dim,
        )
        # Decode the reshaped quantized tensor
        dec = self.decode(z_q)

        return dec

    def decode_code(self, code_b) -> Any:
        """
        Decodes the given code 'code_b' into a reconstructed output.

        Args:
            code_b (Tensor): The code to be decoded.

        Returns:
            Tensor: The decoded output from the given code.
        """
        # Embed the code into a quantized form
        quant_b = self.network_dict["quantize"].embed_code(code_b)
        # Decode the quantized tensor
        dec = self.decode(quant_b)

        return dec

    def encode_no_quant(self, x) -> Any:
        """
        Encodes the input 'x' into a latent representation, bypassing quantization.

        Args:
            x (Tensor): The input data to be encoded.

        Returns:
            Tensor: The encoded latent representation without quantization.
        """
        # Encode the input data
        h = self.network_dict["encoder"](x)
        # Apply convolution to the encoded data
        h = self.network_dict["quant_conv"](h)
        # quant, emb_loss, info = self.network_dict["quantize"](h, is_voxel=True)
        return h

    def encode(self, x) -> tuple[Any, Any, Any]:
        """
        Encodes the input 'x' into a latent representation.

        Args:
            x (Tensor): The input data to be encoded.

        Returns:
            Tensor: The quantized latent vector.
            float: The embedding loss from quantization.
            dict: Additional information from the quantization process.
        """
        # Encode the input point cloud
        h = self.network_dict["encoder"](x)
        # Apply convolution to the encoded data
        h = self.network_dict["quant_conv"](h)
        # Quantize the convolution-applied encoded data
        quant, emb_loss, info = self.network_dict["quantize"](h, is_voxel=True)

        return quant, emb_loss, info

    def encode_wrapper(self, data) -> Any:
        latent = self.encode_no_quant(data["sdf"])
        return latent

    def decode_wrapper(self, quant) -> Any:
        return self.decode_no_quant(quant, force_not_quantize=True)

    def forward(self, input_pack) -> dict:
        """
        Forward pass of the PartAE network.

        Args:
            input_pack (Dict): The input data batch.
            visualize (bool): Flag to indicate whether to include visualization data.

        Returns:
            Dict: The output data pack containing predictions and additional data.
        """
        output = {}
        # phase = input_pack["phase"]
        # epoch = input_pack["epoch"]

        sdf_in = input_pack["sdf"]

        # Encode
        quant, codebook_loss, _ = self.encode(sdf_in)

        # Decode
        sdf_hat = self.decode(quant)

        output["sdf_hat"] = sdf_hat
        output["codebook_loss"] = codebook_loss

        return output


# VQVAE modules
def nonlinearity(x) -> Any:
    """
    Swish nonlinearity
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32) -> torch.nn.GroupNorm:
    if in_channels <= 32:
        num_groups = in_channels // 4
    elif in_channels % num_groups != 0:
        num_groups = 30
    # else:
    # num_groups = 32

    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, with_conv) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, with_conv) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb=None) -> Any:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x) -> Any:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d * h * w)
        q = q.permute(0, 2, 1)  # b,dhw,c
        k = k.reshape(b, c, d * h * w)  # b,c,dhw
        w_ = torch.bmm(q, k)  # b,dhw,dhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, d * h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder3D(torch.nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        activ="gelu",
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        if activ == "lrelu":
            self.nonlinearity = torch.nn.LeakyReLU()
        elif activ == "swish":
            self.nonlinearity = nonlinearity
        elif activ == "gelu":
            self.nonlinearity = torch.nn.GELU()

        # print(activ)

        # downsampling
        self.conv_in = torch.nn.Conv3d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print(
                        "[*] Enc has Attn at i_level, i_block: %d, %d"
                        % (i_level, i_block)
                    )
                    attn.append(AttnBlock(block_in))
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x) -> Any:
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        # hs = [self.conv_in(x)]
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions - 1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        # h = nonlinearity(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(torch.nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        activ="gelu",
        **ignorekwargs,
    ) -> None:
        super().__init__()

        if activ == "lrelu":
            self.nonlinearity = torch.nn.LeakyReLU()
        elif activ == "swish":
            self.nonlinearity = nonlinearity
        elif activ == "gelu":
            self.nonlinearity = torch.nn.GELU()

        # print(activ)

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(
                self.num_res_blocks
            ):  # change this to align with encoder
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print(
                        "[*] Dec has Attn at i_level, i_block: %d, %d"
                        % (i_level, i_block)
                    )
                    attn.append(AttnBlock(block_in))
            up = torch.nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z) -> Any:
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks):  # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h
