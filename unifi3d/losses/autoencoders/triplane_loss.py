import torch
import torch.nn as nn
from torch.nn import functional as F


class TriplaneLoss(nn.Module):
    def __init__(
        self,
        kl_weight=1e-3,
        num_samples=2048,
        codebook_weight=1.0,
        losses=["reconstruction"],
        extra_loss_start=0.0,
        extra_loss_end_exponential=0,
        extra_loss_anneal_rate_exponential=-2,  # this is per epoch!
    ):
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight
        self.codebook_weight = codebook_weight
        self.num_samples = num_samples // 2
        self.losses = losses
        self.loss_statistics = {}
        self.initial_extra_loss_val = None  # used to scale the extra loss to ~1.0
        self.reconstruction_loss_threshold = (
            100.0  # 100 is roughly acceptable recon loss for Triplane
        )
        self.extra_loss_weight = extra_loss_start
        self.extra_loss_end = 10 ** (extra_loss_end_exponential)
        self.extra_loss_anneal_rate = 10**extra_loss_anneal_rate_exponential
        if "kl_divergence" in losses or "vector_quantization" in losses:
            assert (
                self.extra_loss_anneal_rate > 0
            ), "extra_loss anneal rate must be positive"
            print(
                f"Using extra_loss annealing with rate {self.extra_loss_anneal_rate}, end value {self.extra_loss_end}"
            )

    def update_extra_loss_weight(self, epoch):
        self.extra_loss_weight = min(
            self.extra_loss_end,
            self.extra_loss_weight + self.extra_loss_anneal_rate * epoch,
        )

    def get_loss_statistics(self):
        return self.loss_statistics

    def forward(self, data, output):
        if "occupancy" in data:
            labels = data["occupancy"]
        else:
            labels = data["points.occ"]
        reconstructions = output["decoded_output"]
        loss_vol = self.criterion(
            reconstructions[:, : self.num_samples],
            labels[:, : self.num_samples],
        )
        loss_near = self.criterion(
            reconstructions[:, self.num_samples :],
            labels[:, self.num_samples :],
        )
        loss_reconstruction = loss_vol + 0.1 * loss_near

        loss = 0
        for loss_type in self.losses:
            if loss_type == "reconstruction":
                loss += loss_reconstruction
                self.loss_statistics["reconstruction"] = loss_reconstruction
            elif loss_type == "kl_divergence":
                if not self.initial_extra_loss_val:
                    self.initial_extra_loss_val = output["kl_loss"].detach()
                scaling_factor = (
                    self.reconstruction_loss_threshold / self.initial_extra_loss_val
                )
                # # scaling factor ensures that the extra loss is roughly as large as reconstruction loss
                # kl_loss = self.kl_weight * self.extra_loss_weight * output["kl_loss"] * scaling_factor
                kl_loss = self.kl_weight * output["kl_loss"].mean()
                # kl_loss = self.kl_weight * output["kl_loss"]
                print("LOSS: ", loss, " | ", kl_loss)
                loss += kl_loss
                output.pop("kl_loss", None)
                self.loss_statistics["kl_loss"] = kl_loss
                self.loss_statistics["kl_loss_weight"] = self.extra_loss_weight
                self.loss_statistics["scaling_factor"] = scaling_factor
            elif loss_type == "vector_quantization":
                if not self.initial_extra_loss_val:
                    self.initial_extra_loss_val = output["quantize_loss"].detach()
                # loss for vqvae, that latent_code is close to the codebook
                scaling_factor = (
                    self.reconstruction_loss_threshold / self.initial_extra_loss_val
                )
                # scaling factor ensures that the extra loss is roughly as large as reconstruction loss
                # quantization_loss = (
                #     self.extra_loss_weight * output["quantize_loss"] * scaling_factor
                # )
                quantization_loss = (
                    self.codebook_weight * output["quantize_loss"].mean()
                )
                print("LOSS: ", loss, " | ", quantization_loss)
                loss += quantization_loss
                self.loss_statistics["vector_quantization"] = quantization_loss
                self.loss_statistics["vector_quantization_weight"] = (
                    self.extra_loss_weight
                )
                self.loss_statistics["scaling_factor"] = scaling_factor

                output.pop("vector_quantization", None)
            else:
                raise NotImplementedError(f"Loss type {loss_type} not implemented")

        return loss
