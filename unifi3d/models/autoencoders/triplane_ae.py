# --------------------------------------------------------
# Code adapted from https://github.com/autonomousvision/convolutional_occupancy_networks/tree/master/src/conv_onet/models
# Licensed under The MIT License
# --------------------------------------------------------

import os
import torch
import tempfile
import shutil

from torch import distributions as dist
import numpy as np
import open3d as o3d
from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder
from unifi3d.utils.triplane_utils.models.triplane_decoder import (
    LocalDecoder,
    PatchLocalDecoder,
    LocalPointDecoder,
)
from unifi3d.utils.triplane_utils.encoder import encoder_dict

decoder_dict = {
    "simple_local": LocalDecoder,
    "simple_local_crop": PatchLocalDecoder,
    "simple_local_point": LocalPointDecoder,
}


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


class GroupConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ) -> None:
        super(GroupConv, self).__init__()
        self.conv = torch.nn.Conv2d(
            3 * in_channels,
            3 * out_channels,
            kernel_size,
            stride,
            padding,
            groups=3,
        )

    def forward(self, data, scale=1.0):
        data = torch.concat(torch.chunk(data, 3, dim=-1), dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data, 3, dim=1), dim=-1)
        return data


class GroupConvTranspose(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ) -> None:
        super(GroupConvTranspose, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(
            3 * in_channels,
            3 * out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups=3,
        )

    def forward(self, data):
        data = torch.concat(torch.chunk(data, 3, dim=-1), dim=1)
        data = self.conv(data)
        data = torch.concat(torch.chunk(data, 3, dim=1), dim=-1)
        return data


class Conv3DAware(torch.nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels // 3, kernel_size, stride, padding
        )
        self.out_channels = out_channels

    def perception_3d_sdf(self, x):
        # x = torch.concatenate(torch.chunk(x,chunks=3,dim=1),dim=3)
        _, _, h, w = x.shape
        # fea_yx, fea_zx, fea_yz = x[..., 0:w // 3], x[..., w // 3:(w // 3) * 2], x[..., (w // 3) * 2:]
        fea_yx, fea_zx, fea_yz = torch.chunk(x, chunks=3, dim=1)
        fea_yx_mean_y = torch.mean(fea_yx, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_yx_mean_x = torch.mean(fea_yx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_zx_mean_z = torch.mean(fea_zx, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_zx_mean_x = torch.mean(fea_zx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yz_mean_y = torch.mean(fea_yz, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_yz_mean_z = torch.mean(fea_yz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yx_3d_aware = torch.cat((fea_yx, fea_zx_mean_x, fea_yz_mean_y), dim=1)
        fea_zx_3d_aware = torch.cat((fea_zx, fea_yx_mean_x, fea_yz_mean_z), dim=1)
        fea_yz_3d_aware = torch.cat((fea_yz, fea_yx_mean_y, fea_zx_mean_z), dim=1)
        fea = torch.cat([fea_yx_3d_aware, fea_zx_3d_aware, fea_yz_3d_aware], dim=3)

        return fea

    def forward(self, x, scale=1.0):
        triplane = self.perception_3d_sdf(x)
        result = self.conv(triplane)
        result = torch.cat(torch.chunk(result, chunks=3, dim=3), dim=1)
        return result


class Conv3DAwareTranspose(torch.nn.Module):
    """
    adapted for 3D aware convolution for triplane in Rodin: https://arxiv.org/pdf/2212.06135.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=1,
    ):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels // 3,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.out_channels = out_channels

    def perception_3d_sdf(self, x):
        # x = torch.concatenate(torch.chunk(x,chunks=3,dim=1),dim=3)
        _, _, h, w = x.shape
        # fea_yx, fea_zx, fea_yz = x[..., 0:w // 3], x[..., w // 3:(w // 3) * 2], x[..., (w // 3) * 2:]
        fea_yx, fea_zx, fea_yz = torch.chunk(x, chunks=3, dim=1)
        fea_yx_mean_y = torch.mean(fea_yx, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_yx_mean_x = torch.mean(fea_yx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_zx_mean_z = torch.mean(fea_zx, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_zx_mean_x = torch.mean(fea_zx, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yz_mean_y = torch.mean(fea_yz, dim=3, keepdim=True).repeat(1, 1, 1, w)
        fea_yz_mean_z = torch.mean(fea_yz, dim=2, keepdim=True).repeat(1, 1, h, 1)

        fea_yx_3d_aware = torch.cat((fea_yx, fea_zx_mean_x, fea_yz_mean_y), dim=1)
        fea_zx_3d_aware = torch.cat((fea_zx, fea_yx_mean_x, fea_yz_mean_z), dim=1)
        fea_yz_3d_aware = torch.cat((fea_yz, fea_yx_mean_y, fea_zx_mean_z), dim=1)
        fea = torch.cat([fea_yx_3d_aware, fea_zx_3d_aware, fea_yz_3d_aware], dim=3)

        return fea

    def forward(self, x, scale=1.0):
        triplane = self.perception_3d_sdf(x)
        result = self.conv_transpose(triplane)
        result = torch.cat(torch.chunk(result, chunks=3, dim=3), dim=1)
        return result


class TriplaneAE(BaseEncoderDecoder):
    """
    Triplane Autoencoder:
    - Triplane: https://github.com/autonomousvision/convolutional_occupancy_networks/tree/master

    """

    def __init__(
        self,
        config,
        freeze_pretrained=False,
        latent_code_stats=None,
        density_grid_points=128,
        device="cuda",
        ckpt_path=None,
    ) -> None:

        super().__init__()
        self.latent_code_stats = latent_code_stats
        self.freeze_pretrained = freeze_pretrained

        self.config = config
        decoder, encoder, device = self.get_model_kwargs(config, device=device)
        self.decoder = decoder.to(device)
        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        # TODO: this is a copy paste from shape2vecset_ae.py. Make a class
        # called PointCloudEncoderDecoder from which both inherit
        # design grid query points
        self.density = density_grid_points

        self.queried_points = None
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        # Design grid query points
        self.density = density_grid_points
        x = np.linspace(-1, 1, self.density + 1)
        y = np.linspace(-1, 1, self.density + 1)
        z = np.linspace(-1, 1, self.density + 1)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.grid_query_points = (  # Shape [1, (self.density + 1)^3, 3]
            torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
            .view(3, -1)
            .transpose(0, 1)[None]
            .to("cuda", non_blocking=True)
        ) * 0.5  # query points must be in [-0.5, 0.5]

    def load_state_dict(self, state_dict):
        if "model" in state_dict:
            state_dict = state_dict["model"]

        model_dict = self.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        for k in model_dict.keys():
            if k not in state_dict:
                print(f"{k} not in state_dict")
        if self.freeze_pretrained:
            # Set all pretrained_dict variables to not learnable (frozen)
            for param in pretrained_dict.values():
                param.requires_grad = False

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        super().load_state_dict(model_dict)

    def forward(self, data):
        if "pointcloud_crop" in data.keys():
            raise NotImplementedError(
                "This is a legacy option from the convolutional_occupancy_network. It's not implemented."
            )
        if "pcl" in data:
            latent_code = self._encode(data["pcl"])
        else:
            latent_code = self._encode(data["inputs"])

        kwargs = {}
        # General points
        output = {}
        if "points" in data:
            output["decoded_output"] = self.decode(
                data["points"],
                latent_code,
                **kwargs,
            )
        else:
            output["decoded_output"] = self.decode(
                data["query_points"], latent_code, **kwargs
            )

        # Flatten and concatenate all tensors
        output["latent_code"] = torch.cat(
            [tensor.flatten() for tensor in latent_code.values()]
        )

        return output

    def _encode(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        if self.latent_code_stats:
            for key, val in c.items():
                c[key] = self.normalize(val)
        return c

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """
        return self._encode(inputs)

    def encode(self, **kwargs):
        return self._encode(**kwargs)

    def decode(self, p, c, **kwargs):  # legacy function for interface
        """Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        return self.decode_wrapper(c, p)

    def _decode(self, query_points, encoded, **kwargs):

        if self.latent_code_stats:
            for key, val in encoded.items():
                encoded[key] = self.inverse_normalize(val)
        return self.decoder(query_points, encoded, **kwargs)

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model

    def encode_wrapper(self, data_sample, **kwargs):
        """
        Encode the batch given as a dictionary.
        Args:
            data_sample (dict): A dictionary containing the batch data.
        Returns:
            torch.Tensor or tuple of tensors: The encoded representations.
        """
        if "pcl" in data_sample:
            encoded_dict = self._encode(data_sample["pcl"])
        else:
            encoded_dict = self._encode(data_sample["inputs"])
        triplane_features = self.convert_planes_to_token(encoded_dict)
        return triplane_features

    def convert_planes_to_token(self, encoded_dict):
        batch_size = encoded_dict["xz"].shape[0]
        features = encoded_dict["xz"].shape[1]

        # Stack the list, such that it's [amount_planes, batch_size, feature, amount_points_per_plane]
        # Then permute to [batch_size, features, amount_planes, amount_points_per_plane]
        if self.config["model"]["triplane_feature_mode"] == "image_token":
            triplane_features = [encoded_dict[key] for key in ["xy", "xz", "yz"]]
            triplane_features = torch.cat(triplane_features, dim=-1)
            triplane_features = triplane_features.reshape(batch_size, features, -1)
            return triplane_features.permute(0, 2, 1)
        elif self.config["model"]["triplane_feature_mode"] == "image":
            triplane_features = [encoded_dict[key] for key in ["xy", "xz", "yz"]]
            triplane_features = torch.cat(triplane_features, dim=-1)
            return triplane_features
        elif self.config["model"]["triplane_feature_mode"] == "image_channel_stack":
            triplane_features = [encoded_dict[key] for key in ["xy", "xz", "yz"]]
            triplane_features = torch.cat(triplane_features, dim=1)
            return triplane_features
        elif self.config["model"]["triplane_feature_mode"] == "token":
            # Get all feature planes as list
            triplane_features = [
                encoded_dict[key].view(batch_size, features, -1)
                for key in ["xy", "xz", "yz"]
            ]
            triplane_features = torch.stack(triplane_features, dim=0)

            triplane_features = triplane_features.permute(1, 3, 2, 0)
            # Then reshape into [batch_size, features, amount_planes * amount_points_per_plane]
            height_width_size = triplane_features.shape[1]
            triplane_features = triplane_features.reshape(
                batch_size, height_width_size, -1
            )
            return triplane_features
        else:
            raise NotImplementedError(
                f"Triplane feature mode {self.config['model']['triplane_feature_mode']} not recognized."
            )

    def convert_token_to_planes(self, triplane_features):
        batch_size = triplane_features.shape[0]
        if self.config["model"]["triplane_feature_mode"] == "image":
            assert (
                len(triplane_features.shape) == 4
            ), "Image must be of dim (batch_size, channels, height, width*3)"
            resolution = triplane_features.shape[2]
            features = triplane_features.shape[1]

            triplane_features = triplane_features.view(
                batch_size, features, resolution, 3, resolution
            )
            xy_plane, xz_plane, yz_plane = torch.unbind(triplane_features, dim=-2)

            decoded_dict = {
                "xy": xy_plane,  # (B, C, res, res)
                "xz": xz_plane,
                "yz": yz_plane,
            }
            return decoded_dict
        elif self.config["model"]["triplane_feature_mode"] == "image_channel_stack":
            assert (
                len(triplane_features.shape) == 4
            ), "Image must be of dim (batch_size, channels*3, height, width)"
            resolution = triplane_features.shape[2]
            features = triplane_features.shape[1] // 3

            triplane_features = triplane_features.view(
                batch_size, 3, features, resolution, resolution
            )
            xy_plane, xz_plane, yz_plane = torch.unbind(triplane_features, dim=1)

            decoded_dict = {
                "xy": xy_plane,
                "xz": xz_plane,
                "yz": yz_plane,
            }
            return decoded_dict

        elif self.config["model"]["triplane_feature_mode"] == "token":
            assert (
                triplane_features.shape[2] % 3 == 0
            ), "Triplane features should be divisible by 3"
            features = triplane_features.shape[2] // 3
            assert len(triplane_features.shape) == 3
            resolution = int(triplane_features.shape[1] ** 0.5)

            triplane_features = triplane_features.view(batch_size, -1, features, 3)
            triplane_features = triplane_features.permute(3, 0, 1, 2)

            # Split the tensor into individual planes
            xy_plane, xz_plane, yz_plane = torch.unbind(triplane_features, dim=0)

            # Convert back to dictionary format
            decoded_dict = {
                "xy": xy_plane.permute(0, 2, 1).reshape(
                    batch_size, features, resolution, resolution
                ),
                "xz": xz_plane.permute(0, 2, 1).reshape(
                    batch_size, features, resolution, resolution
                ),
                "yz": yz_plane.permute(0, 2, 1).reshape(
                    batch_size, features, resolution, resolution
                ),
            }
            return decoded_dict
        elif self.config["model"]["triplane_feature_mode"] == "image_token":

            assert len(triplane_features.shape) == 3
            features = triplane_features.shape[2]
            resolution = int((triplane_features.shape[1] / 3) ** 0.5)
            # Reshape to [batch_size, features, amount_planes, amount_points_per_plane]
            triplane_features = triplane_features.permute(0, 2, 1).reshape(
                batch_size, features, resolution, 3, resolution
            )
            # Split the tensor into individual planes
            xy_plane, xz_plane, yz_plane = torch.unbind(triplane_features, dim=-2)

            # Convert back to dictionary format
            decoded_dict = {
                "xy": xy_plane,
                "xz": xz_plane,
                "yz": yz_plane,
            }
            return decoded_dict
        else:
            raise NotImplementedError(
                f"Triplane feature mode {self.config['model']['triplane_feature_mode']} not recognized."
            )

    def decode_wrapper(self, encoded, query_points=None):
        """
        Decode the latents of a batch.
        Args:
            encoded (torch.Tensor or tuple of tensors): latent representation of batch
        Returns:
            torch.Tensor or tuple of tensors: The decoded data.
        """
        grid = False
        if query_points is None:
            query_points = self.grid_query_points  # Shape [1, (self.density + 1)^3, 3]
            grid = True

        if isinstance(encoded, dict):
            return self._decode(query_points, encoded, **{})
        elif isinstance(encoded, torch.Tensor):
            if len(encoded.shape) == 5:
                # this is for generated features of size (Batch size, 3, Feature Size, Resolution, Resolution)
                encoded = self.triplane_features_to_dict(encoded)
            else:
                # this is for diffusion, where shape is (Batch Size, Token Length, Feature Size)
                # or (Batch Size, Feature Size, Resolution, Resolution*3)
                encoded = self.convert_token_to_planes(encoded)

            if grid:
                return (
                    self._decode(query_points, encoded, **{})
                    .view(-1, self.density + 1, self.density + 1, self.density + 1)
                    .permute(0, 3, 1, 2)
                )
            else:
                return self._decode(query_points, encoded, **{})
        else:
            raise NotImplementedError(
                "Encoded type not recognized. Either has to be dict with keys == plane_type in config, or a triplane feature tensor."
            )

    def triplane_features_to_dict(self, encoded):
        """
        Convert a triplane feature tensor to a dictionary with keys == plane_type in config.
        Args:
            encoded (torch.Tensor): The triplane feature tensor.
        Returns:
            dict: The dictionary with keys == plane_type in config.
        """
        encoded_dict = {}
        for idx, key in enumerate(
            sorted(self.config["model"]["encoder_kwargs"]["plane_type"])
        ):
            encoded_dict[key] = encoded[:, idx, ...]
        return encoded_dict

    @staticmethod
    def get_model_kwargs(cfg, device=None):
        """Return the Occupancy Network model.

        Args:
            cfg (dict): imported yaml config
            device (device): pytorch device
        """
        decoder = cfg["model"]["decoder"]
        encoder = cfg["model"]["encoder"]
        dim = cfg["model"]["dim"]
        c_dim = cfg["model"]["c_dim"]
        decoder_kwargs = cfg["model"]["decoder_kwargs"]
        encoder_kwargs = cfg["model"]["encoder_kwargs"]
        padding = cfg["model"]["padding"]

        # local positional encoding
        if "local_coord" in cfg["model"].keys():
            encoder_kwargs["local_coord"] = cfg["model"]["local_coord"]
            decoder_kwargs["local_coord"] = cfg["model"]["local_coord"]
        if "pos_encoding" in cfg["model"]:
            encoder_kwargs["pos_encoding"] = cfg["model"]["pos_encoding"]
            decoder_kwargs["pos_encoding"] = cfg["model"]["pos_encoding"]

        decoder = decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding, **decoder_kwargs
        )

        if encoder is not None:
            if not (encoder in encoder_dict.keys()):
                raise RuntimeError(
                    f"Encoder {encoder} not found in encoder_dict in triplane_ae.py."
                )
            encoder = encoder_dict[encoder](
                dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs
            )
        else:
            raise RuntimeError("Encoder not passed. This won't work.")

        return decoder, encoder, device

    def load_weights_from_url(self, url):
        """Load a module dictionary from url.

        Args:
            url (str): url to saved model
        """
        print(url)
        print("=> Loading checkpoint from url...")
        from torch.utils import model_zoo

        state_dict = model_zoo.load_url(url, progress=True)

        self.load_state_dict(state_dict["model"])

    def convert_off_to_triangle_mesh(self, mesh, save_file_path=None):
        asset_file_name = "triplane_asset.off"

        if save_file_path is not None and "off" not in save_file_path:
            asset_file_name = f"{save_file_path}/{asset_file_name}"
        elif save_file_path is not None:
            asset_file_name = save_file_path
        else:
            tmp_dir = tempfile.mkdtemp()
            asset_file_name = os.path.join(tmp_dir, asset_file_name)

        mesh.export(asset_file_name)

        triangle_mesh = o3d.io.read_triangle_mesh(asset_file_name)
        if save_file_path is None:
            os.remove(asset_file_name)
            shutil.rmtree(tmp_dir)
        return triangle_mesh

    def generate_grid(self):
        x = np.linspace(-1, 1, self.density + 1)
        y = np.linspace(-1, 1, self.density + 1)
        z = np.linspace(-1, 1, self.density + 1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid_query_points = (
            torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
            .view(3, -1)
            .transpose(0, 1)[None]
            .to("cuda", non_blocking=True)
        ) * 0.5  # query points must be in [-0.5, 0.5]
        return grid_query_points

    def decoded_output_to_mesh(self, decoded_output):
        # Assuming points is an Nx3 array and logits is an N array
        occupancy_probability = torch.sigmoid(decoded_output.logits)

        # Threshold logits to get binary occupancy (1 if occupied, 0 if free)
        occupancy_threshold = 0.5
        occupancy = occupancy_probability > occupancy_threshold

        # Select the occupied points only
        occupied_points = self.grid_query_points[occupancy].cpu().numpy()

        # Create a PointCloud object with the occupied points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupied_points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # radii = [0.05, 0.1, 0.2]
        print("Creating mesh")
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     pcd, o3d.utility.DoubleVector(radii)
        # )

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        print("Mesh created")

        return mesh

    def normalize(self, data, mean=None, std=None):
        """
        Normalize the data using the provided mean and standard deviation.

        Parameters:
        data (numpy array): The data to be normalized.
        mean (float or numpy array): The mean value(s) for normalization.
        std (float or numpy array): The standard deviation value(s) for normalization.

        Returns:
        numpy array: The normalized data.
        """
        mean = mean if mean is not None else self.latent_code_stats["mean"]
        std = std if std is not None else self.latent_code_stats["std"]
        return (data - mean) / std

    def inverse_normalize(self, normalized_data, mean=None, std=None):
        """
        Convert the normalized data back to the original scale using the provided mean and standard deviation.

        Parameters:
        normalized_data (numpy array): The normalized data to be converted back.
        mean (float or numpy array): The mean value(s) used for normalization.
        std (float or numpy array): The standard deviation value(s) used for normalization.

        Returns:
        numpy array: The data in the original scale.
        """
        mean = mean if mean is not None else self.latent_code_stats["mean"]
        std = std if std is not None else self.latent_code_stats["std"]
        return (normalized_data * std) + mean
