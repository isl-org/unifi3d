"""
Point Cloud - SDF AutoEncoder used in NAP

From: https://github.com/JiahuiLei/NAP/blob/main/core/models/arti_partshape_ae.py
      https://github.com/JiahuiLei/NAP/blob/main/core/lib/point_encoder/pointnet.py
      https://github.com/JiahuiLei/NAP/blob/main/core/lib/implicit_func/onet_decoder.py
      https://github.com/JiahuiLei/NAP/blob/main/core/lib/implicit_func/siren_dcoder.py
      https://github.com/autonomousvision/occupancy_flow
"""

import copy
import os
import torch
import yaml
from unifi3d.utils.data.sdf_utils import (
    generate_query_points,
    mesh_to_sdf,
    sdf_to_mesh,
)

from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder


class PC2SDF_AE(BaseEncoderDecoder):
    """
    Point Cloud 2 Signed Distance Field Autoencoder (PC2SDF_AE) Network.
    Adapted from https://github.com/autonomousvision/occupancy_flow

    Args:
        config (Dict): Configuration dictionary for the network.
        ckpt_path (string): Path of existing checkpoint
    """

    def __init__(self, config, ckpt_path=None) -> None:
        super(PC2SDF_AE, self).__init__()

        # Variables initialization
        self.config = copy.deepcopy(config)
        latent_dim = self.config.get("c_dim", 128)
        dim_encoder = self.config.get("dim_encoder", 3)
        hidden_encoder = self.config.get("hidden_encoder", 512)
        dim_decoder = self.config.get("dim_decoder", 51)
        hidden_decoder = self.config.get("hidden_decoder", 256)
        decoder_class = {
            "decoder": Decoder,
            "cbatchnorm": DecoderCBatchNorm,
            "decoder_siren": DecoderSIREN,
        }[self.config.get("decoder_type", "decoder")]
        self.res = self.config.get("res", 64)
        self.query = generate_query_points(grid_resolution=self.res, padding=0.1)

        # Network architecture definition
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": ResnetPointnet(
                    c_dim=latent_dim, dim=dim_encoder, hidden_dim=hidden_encoder
                ),
                "decoder": decoder_class(
                    dim=dim_decoder,
                    z_dim=0,
                    c_dim=latent_dim,
                    hidden_size=hidden_decoder,
                    leaky=True,
                ),
            }
        )

        # Loss weights
        self.sdf2occ_factor = self.config.get("sdf2occ_factor", -1)
        self.w_sdf = self.config.get("w_sdf", 1.0)
        self.w_reg = self.config.get("w_reg", 0.001)
        self.near_th = self.config.get("near_th", -1)
        if self.near_th > 0:
            self.w_near = self.config.get("w_near", 1.0)
            self.w_far = self.config.get("w_far", 1.0)

        # Number of positional encoding terms.
        self.N_pe = self.config.get("N_pe", 8)
        if self.N_pe > 0:
            self.freq = 2 ** torch.Tensor([i for i in range(self.N_pe)])

        return

    def encode_wrapper(self, data_sample, **kwargs):
        """Auxiliary method that only takes the dataset item as input"""
        return self.encode(data_sample)

    def decode_wrapper(self, codes):
        """Auxiliary method that only takes the codes as input"""
        # Get the SDF value encoded in the latent in multiple query points
        # Important note: this formulation supports adaptive sampling strategies such as:
        # https://github.com/autonomousvision/occupancy_flow/blob/master/im2mesh/utils/libmise/mise.pyx
        # This is expected to provide much better results than classical grid sampling...
        self.query = (
            torch.from_numpy(self.query)
            .float()
            .cuda()
            .unsqueeze(0)
            .expand(codes.shape[0], -1, -1)
        )
        sdf = self.decode(self.query, None, codes, return_sdf=True)
        sdf_grid = sdf.view(-1, 1, self.res, self.res, self.res)

        return sdf_grid

    def encode(self, pcl):
        """
        Encodes the input point clouds into a latent space representation.

        Args:
            pcl (torch.Tensor): The input point cloud (batch_size, num_points, 3).

        Returns:
            torch.Tensor: The encoded latent vector (batch_size, latent_dim).
        """
        return self.network_dict["encoder"](pcl)

    def decode(self, query, z_none, c, return_sdf: bool = False):
        """
        Decodes the Signed Distance Field (SDF) given query points, a latent vector, and conditioning information.

        Args:
            query (torch.Tensor): The input tensor containing query points.
            z_none (torch.Tensor): A latent vector that is currently not used in the function.
            c (torch.Tensor): Conditioning information that is passed to the SDF decoder along with query points.
            return_sdf (bool): A flag to determine the return type.

        Returns:
            torch.Tensor or Bernoulli distribution: Depending on the `return_sdf` flag.
        """

        # Positional encoding if required
        if self.N_pe > 0:
            batch_size, N, _ = query.shape
            w = self.freq.to(query.device)
            pe = w[None, None, None, ...] * query[..., None]
            pe = pe.reshape(batch_size, N, -1)
            query = torch.cat([query, torch.cos(pe), torch.sin(pe)], -1)

        # Decode using the decoder network
        sdf = self.network_dict["decoder"](query, None, c)

        # Return raw SDF values or Bernoulli distribution
        if return_sdf:
            return sdf
        else:
            return torch.distributions.Bernoulli(logits=self.sdf2occ_factor * sdf)

    def forward(self, input_pack) -> dict:
        """
        Forward pass of the PC2SDF_AE network.

        Args:
            input_pack (dict): The input data pack containing various inputs.
            visualize (bool): Flag to indicate whether to include visualization data.

        Returns:
            dict: The output data pack containing predictions and additional data.
        """
        output = {}

        # Extract input data
        query = input_pack["query_points"]  # B,N,3
        pcl = input_pack["pcl"]  # B,M,3
        sdf_gt = input_pack["sampled_sdf"]  # B,N

        # Encode input point cloud
        z = self.encode(pcl)

        # Decode (note that sdf_hat is a 1D array of distances!)
        sdf_hat = self.decode(query, None, z, return_sdf=True)

        # Loss function
        loss_sdf = abs(sdf_hat - sdf_gt)
        if self.near_th > 0:
            near_mask = sdf_gt < self.near_th
            loss_sdf_near = loss_sdf[near_mask].sum() / (near_mask.sum() + 1e-7)
            loss_sdf_far = loss_sdf[~near_mask].sum() / ((~near_mask).sum() + 1e-7)
            loss_sdf = self.w_near * loss_sdf_near + self.w_far * loss_sdf_far
            loss_near = loss_sdf_near.detach()
            loss_far = loss_sdf_far.detach()
        else:
            loss_sdf = loss_sdf.mean()
            loss_near = torch.zeros_like(loss_sdf)
            loss_far = torch.zeros_like(loss_sdf)

        loss_reg = (z**2).mean()
        output["sdf_hat"] = sdf_hat
        output["loss"] = self.w_sdf * loss_sdf + self.w_reg * loss_reg
        output["loss_sdf"] = loss_sdf
        output["loss_reg"] = loss_reg
        output["loss_near"] = loss_near
        output["loss_far"] = loss_far
        output["z"] = z.detach()

        return output


# Resnet Blocks
class ResnetBlockFC(torch.nn.Module):
    """
    Fully Connected ResNet Block class.
    Implements a basic ResNet block with fully connected layers.

    Args:
        size_in (int): Input dimension.
        size_out (int): Output dimension.
        size_h (int): Hidden dimension.
        act_method (torch.nn.Module): Activation method to use.
    """

    def __init__(self, size_in, size_out=None, size_h=None, actvn=torch.nn.ReLU()):
        super(ResnetBlockFC, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = torch.nn.Linear(size_in, size_h)
        self.fc_1 = torch.nn.Linear(size_h, size_out)
        self.actvn = actvn

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Linear(size_in, size_out, bias=False)

        # Initialization
        torch.nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockFC_BN(torch.nn.Module):
    """
    Fully Connected ResNet Block with Batch Normalization.
    Similar to ResnetBlockFC but includes Batch Normalization for each layer.

    Args:
        size_in (int): Input dimension.
        size_out (int): Output dimension (optional, defaults to size_in).
        size_h (int): Hidden dimension (optional, defaults to min(size_in, size_out)).
        act_method (torch.nn.Module): Activation method to use.
    """

    def __init__(
        self, size_in, size_out=None, size_h=None, act_method=torch.nn.ReLU
    ) -> None:
        super(ResnetBlockFC_BN, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = torch.nn.Linear(size_in, size_h)
        self.bn_0 = torch.nn.BatchNorm1d(size_in)
        self.fc_1 = torch.nn.Linear(size_h, size_out)
        self.bn_1 = torch.nn.BatchNorm1d(size_h)
        self.actvn = act_method()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Linear(size_in, size_out, bias=False)

        # Initialization
        torch.nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        """
        Performs a max pooling operation on the input tensor.

        Args:
            x (Tensor): Input tensor.
            dim (int): The dimension over which to perform the max pooling.
            keepdim (bool): Whether to retain the reduced dimension.

        Returns:
            Tensor: Max pooled tensor.
        """
        x = self.bn_0(x.permute(0, 2, 1)).permute(0, 2, 1)
        net = self.fc_0(self.actvn(x))
        net = self.bn_1(net.permute(0, 2, 1)).permute(0, 2, 1)
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    """
    Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetPointnet(torch.nn.Module):
    """
    PointNet-based ResNet encoder network.
    Encodes 3D point clouds into a latent space using ResNet blocks.

    Args:
        c_dim (int): Dimension of the latent space.
        dim (int): Dimension of the input points.
        hidden_dim (int): Hidden dimension of the network.
    """

    def __init__(self, c_dim: int = 128, dim: int = 3, hidden_dim: int = 512) -> None:
        super(ResnetPointnet, self).__init__()

        self.c_dim = c_dim

        # Fully connected layer for position encoding
        self.fc_pos = torch.nn.Linear(dim, 2 * hidden_dim)

        # ResNet blocks for encoding
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)

        # Fully connected layer for latent code output
        self.fc_c = torch.nn.Linear(hidden_dim, c_dim)

        # Activation function
        self.actvn = torch.nn.ReLU()

        # Pooling function
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        """
        Forward pass through the PointNet-based ResNet encoder.

        Args:
            p (torch.Tensor): Input point cloud tensor.
            return_unpooled (bool): If True, returns additional unpooled features.

        Returns:
            torch.Tensor: The encoded latent vector.
            torch.Tensor (optional): Unpooled features if return_unpooled is True.
        """
        # batch_size, T, D = p.size()

        # Encode positions
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


class AttentionPooling(torch.nn.Module):
    """
    Attention-based pooling module.
    Applies attention mechanism to pool features from the input tensor.

    Args:
        dim (int): Dimension of the input and output tensors.

    """

    def __init__(self, dim) -> None:
        super(AttentionPooling, self).__init__()

        self.dim = dim
        self.K = ResnetBlockFC(dim, dim, dim)

    def forward(self, x, keepdim=True):
        """
        Forward pass through the attention pooling module.

        Args:
            x (torch.Tensor): Input tensor.
            keepdim (bool): Whether to retain the reduced dimension in output.

        Returns:
            torch.Tensor: Attention pooled tensor.
        """

        # B,N,C, pool over dim 1
        assert x.ndim == 3 and x.shape[-1] == self.dim
        k = self.K(x.max(dim=1, keepdim=True).values)  # B,1,C
        alpha = (k * x).sum(dim=-1, keepdim=True)
        weight = torch.softmax(alpha, dim=1)
        pooled = (weight * x).sum(dim=1, keepdim=keepdim)
        return pooled


class ResnetPointnetAttenPooling(torch.nn.Module):
    """
    PointNet-based ResNet encoder with attention pooling.
    Similar to ResnetPointnet, but uses attention mechanism for pooling.

    Args:
        c_dim (int): Dimension of the latent space.
        dim (int): Dimension of the input points.
        hidden_dim (int): Hidden dimension of the network.
    """

    def __init__(self, c_dim: int = 128, dim: int = 3, hidden_dim: int = 512) -> None:
        super(ResnetPointnetAttenPooling, self).__init__()

        self.c_dim = c_dim

        # Fully connected intput layer.
        self.fc_pos = torch.nn.Linear(dim, 2 * hidden_dim)

        # Residual Blocks with attention pooling layers
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_0 = AttentionPooling(hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_1 = AttentionPooling(hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_2 = AttentionPooling(hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_3 = AttentionPooling(hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_4 = AttentionPooling(hidden_dim)

        # Fully connected output layer.
        self.fc_c = torch.nn.Linear(hidden_dim, c_dim)

        # Activation function.
        self.actvn = torch.nn.ReLU()

        # Maxpooling operation.
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        """
        Forward pass through the PointNet-based ResNet encoder with attention pooling.

        Args:
            p (torch.Tensor): Input point cloud tensor.
            return_unpooled (bool): If True, returns additional unpooled features.

        Returns:
            torch.Tensor: The encoded latent vector.
            torch.Tensor (optional): Unpooled features if return_unpooled is True.
        """
        batch_size, T, D = p.size()

        # Process points through the first layer.
        net = self.fc_pos(p)

        # Sequential processing residual blocks with attention pooling
        net = self.block_0(net)
        pooled = self.atten_pool_0(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.atten_pool_1(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.atten_pool_2(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.atten_pool_3(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        # Recude to  B x F
        net = self.block_4(net)
        ret = self.atten_pool_4(net, keepdim=False)

        # Final output layer with activation.
        c = self.fc_c(self.actvn(ret))

        # Return output and features if required.
        if return_unpooled:
            return c, net
        else:
            return c


class ResnetPointnetMasked(torch.nn.Module):
    """
    PointNet-based ResNet encoder with masked pooling.
    Encodes 3D point clouds into a latent space using ResNet blocks and masked pooling.

    Args:
        c_dim (int): Dimension of the latent space.
        dim (int): Dimension of the input points.
        hidden_dim (int): Hidden dimension of the network.
        use_bn (bool): Flag to use batch normalization.
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512, use_bn=False) -> None:
        super(ResnetPointnetMasked, self).__init__()

        self.c_dim = c_dim
        if use_bn:
            blk_class = ResnetBlockFC_BN
        else:
            blk_class = ResnetBlockFC

        # Fully connected input layer.
        self.fc_pos = torch.nn.Linear(dim, 2 * hidden_dim)

        # Residual Blocks.
        self.block_0 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_1 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_2 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_3 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_4 = blk_class(2 * hidden_dim, hidden_dim)

        # Fully connected output layer.
        self.fc_c = torch.nn.Linear(hidden_dim, c_dim)

        # Activation function.
        self.actvn = torch.nn.ReLU()

    def pool(self, x, m, keepdim=False):
        # The mask is along the T direction
        return (x * m.float().unsqueeze(-1)).sum(1, keepdim=keepdim)

    def forward(self, p, m, return_unpooled=False):
        """
        Forward pass through the PointNet-based ResNet encoder with masked pooling.

        Args:
            p (Tensor): Input point cloud tensor.
            m (Tensor): Mask tensor for pooling.
            return_unpooled (bool): If True, returns additional unpooled features.

        Returns:
            Tensor: The encoded latent vector.
            Tensor (optional): Unpooled features if return_unpooled is True.
        """
        B, N, D = p.size()

        # Process points through the first layer.
        net = self.fc_pos(p)

        # Sequential processing through residual blocks.
        net = self.block_0(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        # Recude to  B x C
        net = self.block_4(net)
        ret = self.pool(net, m)

        # Final output layer with activation
        c = self.fc_c(self.actvn(ret))

        # Return output and features if required.
        if return_unpooled:
            return c, net
        else:
            return c


class CResnetBlockConv1d(torch.nn.Module):
    """
    Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): Dimension of latent conditioned code c.
        size_in (int): Input dimension.
        size_h (int, optional): Hidden dimension. Defaults to input dimension if None.
        size_out (int, optional): Output dimension. Defaults to input dimension if None.
        norm_method (str): Normalization method. Choices are 'batch_norm', 'instance_norm', 'group_norm'.
        legacy (bool): Whether to use legacy blocks.
    """

    def __init__(
        self,
        c_dim,
        size_in,
        size_h=None,
        size_out=None,
        norm_method="batch_norm",
        legacy=False,
    ):
        super(CResnetBlockConv1d, self).__init__()

        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Conditional Batch Normalization layers
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim, size_h, norm_method=norm_method)

        # Convolutional layers
        self.fc_0 = torch.nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = torch.nn.Conv1d(size_h, size_out, 1)

        # Activation function
        self.actvn = torch.nn.ReLU()

        # Shortcut connection
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialize weights of the second convolutional layer to zero
        torch.nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        """
        Forward pass through the Resnet block.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Latent conditioned code c.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Forward pass through the Resnet block
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        # Residual connection
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(torch.nn.Module):
    """
    Conditional batch normalization layer class.

    Args:
        c_dim (int): Dimension of latent conditioned code c.
        f_dim (int): Feature dimension.
        norm_method (str): Normalization method. Choices are 'batch_norm', 'instance_norm', 'group_norm'.
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super(CBatchNorm1d, self).__init__()

        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method

        # Submodules for affine transformation
        self.conv_gamma = torch.nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = torch.nn.Conv1d(c_dim, f_dim, 1)

        # Normalization layer
        if norm_method == "batch_norm":
            self.bn = torch.nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = torch.nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = torch.nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")

        # Initialize parameters of affine transformation submodules
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.conv_gamma.weight)
        torch.nn.init.zeros_(self.conv_beta.weight)
        torch.nn.init.ones_(self.conv_gamma.bias)
        torch.nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        """
        Forward pass through the conditional batch normalization layer.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Latent conditioned code c.

        Returns:
            torch.Tensor: Output tensor.
        """

        assert x.size(0) == c.size(0)
        assert c.size(1) == self.c_dim

        # Ensure c has the correct dimensions [batch_size x c_dim x T]
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(torch.nn.Module):
    """
    Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): Dimension of latent conditioned code c.
        f_dim (int): Feature dimension.
        norm_method (str): Normalization method. Choices are 'batch_norm', 'instance_norm', 'group_norm'.
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super(CBatchNorm1d_legacy, self).__init__()

        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method

        # Linear layers for affine transformation
        self.fc_gamma = torch.nn.Linear(c_dim, f_dim)
        self.fc_beta = torch.nn.Linear(c_dim, f_dim)

        # Normalization layer
        if norm_method == "batch_norm":
            self.bn = torch.nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = torch.nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = torch.nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")

        # Initialize parameters of affine transformation submodules
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fc_gamma.weight)
        torch.nn.init.zeros_(self.fc_beta.weight)
        torch.nn.init.ones_(self.fc_gamma.bias)
        torch.nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        """
        Forward pass through the legacy conditional batch normalization layer.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor, optional): Latent conditioned code c.

        Returns:
            torch.Tensor: Output tensor.
        """

        batch_size = x.size(0)

        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class Decoder(torch.nn.Module):
    """
    Basic Decoder network for OFlow class.
    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
        out_dim (int): dimension of output
        **kwargs: Additional arguments for compatibility.
    """

    def __init__(
        self,
        dim: int = 3,
        z_dim: int = 128,
        c_dim: int = 128,
        hidden_size: int = 128,
        leaky: bool = False,
        out_dim: int = 1,
        **kwargs,
    ):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # Initial linear layer for input points
        self.fc_p = torch.nn.Linear(dim, hidden_size)

        # Optional latent code layers
        if not z_dim == 0:
            self.fc_z = torch.nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = torch.nn.Linear(c_dim, hidden_size)

        # Residual blocks
        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_size, out_dim)

        # Activation function
        if not leaky:
            self.actvn = torch.nn.functional.relu
        else:
            self.actvn = lambda x: torch.nn.functional.leaky_relu(x, 0.2)

    def forward(self, p, z=None, c=None, **kwargs):
        """
        Performs a forward pass through the network.

        Args:
            p (torch.Tensor): Points tensor.
            z (torch.Tensor, optional): Latent code z.
            c (torch.Tensor, optional): Latent conditioned code c.

        Returns:
            torch.Tensor: Output tensor.
        """

        batch_size = p.shape[0]
        p = p.view(batch_size, -1, self.dim)

        # Process input points through the initial linear layer
        net = self.fc_p(p)

        # Combine with latent code z if provided
        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        # Combine with latent conditioned code c if provided
        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        # Sequential processing through residual blocks
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        # Final output layer with sine activation
        out = self.fc_out(self.actvn(net)).squeeze(-1)

        return out


class DecoderSIREN(torch.nn.Module):
    """
    Decoder network utilizing sine activation functions (SIREN).
    This decoder network maps input points, along with optional latent codes z and c,
    through several fully-connected residual blocks with sine activations.

    Args:
        dim (int): Dimension of input points.
        z_dim (int): Dimension of latent code z.
        c_dim (int): Dimension of latent conditioned code c.
        hidden_size (int): Dimension of hidden layers.
        out_dim (int): Dimension of output.
        **kwargs: Additional arguments for compatibility.
    """

    def __init__(
        self,
        dim: int = 3,
        z_dim: int = 128,
        c_dim: int = 128,
        hidden_size: int = 128,
        out_dim: int = 1,
        **kwargs,
    ) -> None:
        super(DecoderSIREN, self).__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # Initial linear layer for input points
        self.fc_p = torch.nn.Linear(dim, hidden_size)

        # Optional latent code layers
        if not z_dim == 0:
            self.fc_z = torch.nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = torch.nn.Linear(c_dim, hidden_size)

        # Residual blocks with sine activation function
        self.block0 = ResnetBlockFC(hidden_size, actvn=torch.sin)
        self.block1 = ResnetBlockFC(hidden_size, actvn=torch.sin)
        self.block2 = ResnetBlockFC(hidden_size, actvn=torch.sin)
        self.block3 = ResnetBlockFC(hidden_size, actvn=torch.sin)
        self.block4 = ResnetBlockFC(hidden_size, actvn=torch.sin)

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_size, out_dim)

        # Sine activation function
        self.actvn = torch.sin

    def forward(self, p, z=None, c=None, **kwargs):
        """
        Performs a forward pass through the network.

        Args:
            p (torch.Tensor): Points tensor.
            z (torch.Tensor, optional): Latent code z.
            c (torch.Tensor, optional): Latent conditioned code c.

        Returns:
            torch.Tensor: Output tensor.
        """

        batch_size = p.shape[0]

        # Process input points through the initial linear layer
        net = self.fc_p(p)

        # Combine with latent code z if provided
        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        # Combine with latent conditioned code c if provided
        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        # Sequential processing through residual blocks
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        # Final output layer with sine activation
        out = self.fc_out(self.actvn(net)).squeeze(-1)

        return out


class DecoderCat(torch.nn.Module):
    """
    Decoder network that concatenates input features directly.

    This decoder network directly processes the input features through
    several fully-connected residual blocks.

    Args:
        input_dim (int): dimension of input features.
        hidden_size (int): dimension of hidden layers.
        leaky (bool): whether to use leaky ReLUs as activation.
        out_dim (int): dimension of output.
        **kwargs: Additional arguments for compatibility.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 128,
        leaky: bool = False,
        out_dim: int = 1,
        **kwargs,
    ) -> None:
        super(DecoderCat, self).__init__()

        # Input layer
        self.fc_in = torch.nn.Linear(input_dim, hidden_size)

        # Residual blocks
        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_size, out_dim)

        # Activation function
        if not leaky:
            self.actvn = torch.nn.functional.relu
        else:
            self.actvn = lambda x: torch.nn.functional.leaky_relu(x, 0.2)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input features tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # Initial linear layer
        net = self.fc_in(x)

        # Sequential processing through residual blocks
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        # Final output layer with activation
        out = self.fc_out(self.actvn(net)).squeeze(-1)

        return out


class DecoderCBatchNorm(torch.nn.Module):
    """
    Conditioned Batch Norm Decoder network for OFlow class.

    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This decoder
    uses conditioned batch normalization to inject the latent codes.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
        out_dim (int): dimension of output
        legacy (bool): whether to use the legacy batch normalization
    """

    def __init__(
        self,
        dim: int = 3,
        z_dim: int = 128,
        c_dim: int = 128,
        hidden_size: int = 256,
        leaky: bool = False,
        out_dim: int = 1,
        legacy: bool = False,
    ) -> None:
        super(DecoderCBatchNorm, self).__init__()

        self.z_dim = z_dim
        self.dim = dim

        # Linear layer for the latent code z, processed if z_dim is not zero.
        if not z_dim == 0:
            self.fc_z = torch.nn.Linear(z_dim, hidden_size)

        # 1D convolution for the point input.
        self.fc_p = torch.nn.Conv1d(dim, hidden_size, 1)

        # Residual Blocks with conditioned batch normalization.
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        # Conditioned Batch Normalization Layer.
        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        # Fully connected output layer.
        self.fc_out = torch.nn.Conv1d(hidden_size, out_dim, 1)

        # Activation function.
        if not leaky:
            self.actvn = torch.nn.functional.relu
        else:
            self.actvn = lambda x: torch.nn.functional.leaky_relu(x, 0.2)

    def forward(self, p, z, c, return_feat_list=False, **kwargs):
        """
        Performs a forward pass through the network.

        Args:
            p (torch.Tensor): points tensor
            z (torch.Tensor): latent code z
            c (torch.Tensor): latent conditioned code c
            return_feat_list (bool): whether to return intermediate features
            **kwargs: Additional arguments for compatibility

        Returns:
            torch.Tensor: Output tensor.
            list[torch.Tensor]: (optional) List of intermediate features.
        """

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()

        # Process points through the first layer.
        net = self.fc_p(p)

        # Combine with latent code z if z_dim is not zero.
        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        # Initialize feature list if required.
        if return_feat_list:
            feat_list = [net]

        # Sequential processing through residual blocks.
        net = self.block0(net, c)
        if return_feat_list:
            feat_list.append(net)

        net = self.block1(net, c)
        if return_feat_list:
            feat_list.append(net)

        net = self.block2(net, c)
        if return_feat_list:
            feat_list.append(net)

        net = self.block3(net, c)
        if return_feat_list:
            feat_list.append(net)

        net = self.block4(net, c)
        if return_feat_list:
            feat_list.append(net)

        # Final output layer with activation and batch normalization.
        out = self.fc_out(self.actvn(self.bn(net, c))).squeeze(1)

        # Return output and features if required.
        if return_feat_list:
            return out, feat_list
        else:
            return out


def load_pc2sdf_ae(ckpt, device="cuda") -> PC2SDF_AE:
    assert type(ckpt) == str

    # Load config
    with open(
        os.path.join("configs", "model", "pc2sdf_ae.yaml"), "r", encoding="utf-8"
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize model
    pc2sdf_ae = PC2SDF_AE(config["model"]["config"])

    map_fn = lambda storage, loc: storage
    state_dict = torch.load(ckpt, map_location=map_fn)
    pc2sdf_ae.network_dict.load_state_dict(state_dict)

    print("PC2SDF_AE: weight successfully load from: %s" % ckpt)
    pc2sdf_ae.requires_grad = False

    pc2sdf_ae.to(device)
    pc2sdf_ae.eval()
    return pc2sdf_ae
