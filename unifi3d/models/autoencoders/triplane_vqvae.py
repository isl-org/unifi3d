# --------------------------------------------------------
# Code adapted from https://github.com/autonomousvision/convolutional_occupancy_networks/tree/master/src/conv_onet/models
# Licensed under The MIT License
# --------------------------------------------------------

from unifi3d.models.autoencoders.triplane_ae import *
import torch


class TriplaneVQVAE(TriplaneAE):
    def __init__(
        self,
        config,
        quantizer,
        freeze_pretrained=False,
        latent_code_stats=None,
        density_grid_points=128,
        device="cuda",
        ckpt_path=None,
    ):

        super().__init__(
            config=config,
            freeze_pretrained=freeze_pretrained,
            latent_code_stats=latent_code_stats,
            density_grid_points=density_grid_points,
            device=device,
            ckpt_path=None,
        )

        # Additional layers for the VQVAE's variational part
        if config["model"]["triplane_feature_mode"] == "image_token":
            self.encoder_shape = config["model"]["c_dim"]
            token_length = (
                config["model"]["encoder_kwargs"]["plane_resolution"] ** 2 * 3
            )
            self.latent_shape = (1, token_length, self.encoder_shape)
        elif config["model"]["triplane_feature_mode"] == "image":
            self.encoder_shape = config["model"]["c_dim"]
            resolution = config["model"]["encoder_kwargs"]["plane_resolution"]
            self.latent_shape = (
                1,
                self.encoder_shape,
                resolution,
                resolution * 3,
            )
        elif config["model"]["triplane_feature_mode"] == "token":
            self.encoder_shape = config["model"]["c_dim"] * 3
            token_length = config["model"]["encoder_kwargs"]["plane_resolution"] ** 2
            self.latent_shape = (1, token_length, self.encoder_shape)
        else:
            raise NotImplementedError(
                "Only 'image_token' and 'image' are supported for triplane_feature_mode"
            )

        self.quant_conv = torch.nn.Sequential(
            GroupConv(self.encoder_shape, self.encoder_shape, kernel_size=1),
            torch.nn.ReLU(),
        ).to(device)
        self.post_quant_conv = torch.nn.Sequential(
            GroupConv(self.encoder_shape, self.encoder_shape, kernel_size=1),
            torch.nn.ReLU(),
        ).to(device)
        self.vector_quantizer = quantizer

        # Network weights initialization
        init_weights(self.quant_conv, "normal", 0.02)
        init_weights(self.post_quant_conv, "normal", 0.02)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def get_quantized_latent(self, latent_code):
        z_q, quantize_loss, min_encoding_indices = self.vector_quantizer(latent_code)

        # Check for NaNs and Infs in mu and log_var
        if torch.isnan(z_q).any() or torch.isinf(z_q).any():
            raise RuntimeError("mu contains NaNs or Infs")
        return z_q, quantize_loss, min_encoding_indices

    def encode_wrapper(self, data_sample, **kwargs):
        latent = super().encode_wrapper(data_sample)
        latent = self.quant_conv(latent)
        return latent

    def decode_wrapper(self, encoded, query_points=None):
        latent = self.post_quant_conv(encoded)
        decoded = super().decode_wrapper(latent, query_points)
        return decoded

    def forward(self, data):
        if "pointcloud_crop" in data.keys():
            raise NotImplementedError(
                "This is a legacy option from the convolutional_occupancy_network. It's not implemented."
            )

        latent_code = self.encode_wrapper(data)
        quantized_code, quantize_loss, _ = self.get_quantized_latent(latent_code)
        # Decode using the quantized code
        if "query_points" in data:
            reconstructed = self.decode_wrapper(quantized_code, data["query_points"])
        else:
            reconstructed = self.decode_wrapper(quantized_code, data["points"])

        latent_code_flattened = quantized_code.flatten()

        output = {
            "latent_code": latent_code_flattened,
            "decoded_output": reconstructed,
            "quantize_loss": quantize_loss,  # Vector quantization loss
        }
        return output

    def sample(self, device_str="cuda", decode=False):
        code = torch.zeros(*self.latent_shape).to(torch.device(device_str)).normal_()
        if decode:
            return self.decode_wrapper(code)
        else:
            return code
