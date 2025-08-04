import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ShapELoss(nn.Module):
    def __init__(self, coarse_weight: float = 1.0, kl_weight=1e-3, quant_weight=1e-3):
        super().__init__()

        self.img_res = 64
        self.coarse_weight = (
            coarse_weight  # weight of losses on NeRF rendering coarse stage
        )
        self.kl_weight = kl_weight
        self.quant_weight = quant_weight
        self.l1loss = nn.L1Loss()

    def forward(self, batch, output, return_loss_log=False):
        """
        Args:
            batch -- the batch of data fed into the autoencoder
            output -- the reconstruction (renders in this case) output by the autoencoder

        Returns:
            loss -- the training loss
        """

        gt_color_arr = []
        for i in range(len(batch["views"])):
            datum_views = batch["views"][i]
            gts = []
            for view in output["batch_camera_idxs"][i]:
                camera_view = datum_views[view]
                gts.append(
                    np.asarray(
                        camera_view.resize((self.img_res, self.img_res), Image.LANCZOS)
                    )
                )
            gts = np.stack(gts)
            gt_color_arr.append(gts)
        gt_color_arr = np.stack(gt_color_arr) / 255.0
        gt_color_tsr = torch.from_numpy(gt_color_arr).to(batch["points"].device)
        loss_nerf_fine_color = self.l1loss(output["batch_fine_renders"], gt_color_tsr)
        loss_nerf_coarse_color = self.l1loss(
            output["batch_coarse_renders"], gt_color_tsr
        )

        gt_alpha_arr = []
        for i in range(len(batch["view_alphas"])):
            datum_views = batch["view_alphas"][i]
            gts = []
            for view in output["batch_camera_idxs"][i]:
                camera_view = datum_views[view]
                gts.append(
                    np.asarray(
                        camera_view.resize((self.img_res, self.img_res), Image.LANCZOS)
                    )
                )
            gts = np.stack(gts)
            gt_alpha_arr.append(gts)
        gt_alpha_arr = np.stack(gt_alpha_arr) / 255.0
        gt_alpha_tsr = torch.unsqueeze(torch.from_numpy(gt_alpha_arr), -1).to(
            batch["points"].device
        )
        loss_nerf_fine_transmittance = self.l1loss(
            output["batch_fine_transmittance"], gt_alpha_tsr
        )
        loss_nerf_coarse_transmittance = self.l1loss(
            output["batch_coarse_transmittance"], gt_alpha_tsr
        )

        loss = (
            loss_nerf_fine_color
            + loss_nerf_fine_transmittance
            + self.coarse_weight
            * (loss_nerf_coarse_color + loss_nerf_coarse_transmittance)
        )

        if "loss_kl" in output.keys():
            loss_kl = output["loss_kl"]
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss += self.kl_weight * loss_kl
        if "loss_quantized" in output.keys():
            loss_ql = output["loss_quantized"]
            loss_ql = torch.sum(loss_ql) / loss_ql.shape[0]
            loss += self.quant_weight * loss_ql

        if not return_loss_log:
            return loss

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_nerf_fine_color": loss_nerf_fine_color.detach().mean(),
            "loss_nerf_coarse_color": loss_nerf_coarse_color.detach().mean(),
            "loss_nerf_fine_transmittance": loss_nerf_fine_transmittance.detach().mean(),
            "loss_nerf_coarse_transmittance": loss_nerf_coarse_transmittance.detach().mean(),
        }

        if "loss_kl" in output.keys():
            log["loss_kl"] = loss_kl
        if "loss_quantized" in output.keys():
            log["loss_quantized"] = loss_ql

        return loss, log
