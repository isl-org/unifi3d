# --------------------------------------------------------
# Code adapted from https://github.com/autonomousvision/convolutional_occupancy_networks/tree/master/src/conv_onet/models
# Licensed under The MIT License
# --------------------------------------------------------

from unifi3d.models.autoencoders.triplane_ae import *
import torch


class TriplaneVAE(TriplaneAE):
    def __init__(
        self,
        config,
        freeze_pretrained=False,
        latent_code_stats=None,
        density_grid_points=128,
        device="cuda",
        ckpt_path=None,
    ) -> None:

        super().__init__(
            config=config,
            freeze_pretrained=freeze_pretrained,
            latent_code_stats=latent_code_stats,
            density_grid_points=density_grid_points,
            device=device,
            ckpt_path=None,
        )
        # Additional layers for the VAE's variational part
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
                "Only 'image_token', 'token', 'image' are supported for triplane_feature_mode"
            )

        # Define the encoder networks for mu and log_var using GroupConv
        self.encoder_mu_fc = torch.nn.Sequential(
            GroupConv(self.encoder_shape, self.encoder_shape, kernel_size=1),
            torch.nn.ReLU(),
        ).to(device)

        self.encoder_var_fc = torch.nn.Sequential(
            GroupConv(self.encoder_shape, self.encoder_shape, kernel_size=1),
            torch.nn.ReLU(),
        ).to(device)

        self.logvar_min_threshold = np.log(config.get("exp_min_threshold", 1e-8))
        self.logvar_max_threshold = np.log(config.get("exp_max_threshold", 3e4))

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def kl_divergence(self, mean, log_var):
        return (
            -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), axis=1).mean()
        )

    def encode(self, x):
        encoder_out = super().encode_wrapper(x)
        mu = self.encoder_mu_fc(encoder_out)
        log_var = self.encoder_var_fc(encoder_out)
        # clamp to avoid infs with log_var.exp(), especially with mixed precision
        log_var = torch.clamp(
            log_var, self.logvar_min_threshold, self.logvar_max_threshold
        )
        return mu, log_var

    def encode_wrapper(self, data):
        # use mean as latent
        mu, _ = self.encode(data)
        return mu

    def forward(self, data):
        if "pointcloud_crop" in data.keys():
            raise NotImplementedError(
                "This is a legacy option from the convolutional_occupancy_network. It's not implemented."
            )

        mean, log_var = self.encode(data)
        z = self.reparameterize(mean, log_var)
        kl_divergence = self.kl_divergence(mean, log_var)

        if kl_divergence < 0:
            raise RuntimeError(f"KL loss is negative: {kl_divergence}")

        # Decode
        if "query_points" in data:
            reconstructed = self.decode_wrapper(z, data["query_points"])
        else:
            reconstructed = self.decode_wrapper(z, data["points"])

        latent_code_flattened = z.flatten()
        output = {
            "latent_code": latent_code_flattened,
            "decoded_output": reconstructed,
            "kl_loss": kl_divergence,
        }
        return output

    def sample(self, device_str="cuda", decode=False):

        code = torch.zeros(*self.latent_shape).to(torch.device(device_str)).normal_()
        if decode:
            return self.decode_wrapper(code)
        else:
            return code
