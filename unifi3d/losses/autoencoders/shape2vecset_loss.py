import torch
from typing import Any


class Shape2VecSetLoss:
    def __init__(
        self,
        kl_weight=1e-3,
        codebook_weight=1.0,
        num_samples=2048,
        near_loss_weight=0.1,
    ):
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight
        self.codebook_weight = codebook_weight
        self.num_samples = num_samples // 2
        self.near_loss_weight = near_loss_weight

    def __call__(self, batch, outputs) -> Any:
        labels = batch["occupancy"]
        # with torch.cuda.amp.autocast(enabled=False):
        #         outputs = model(surface, points)
        if "kl" in outputs:
            loss_kl = outputs["kl"].mean()
            # loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            loss_vq = None
        elif "quantize_loss" in outputs:
            loss_kl = None
            loss_vq = outputs["quantize_loss"].mean()
        else:
            loss_kl = None
            loss_vq = None

        reconstructions = outputs["logits"]

        # version in original repo -> distinguishign near and vol points
        # https://github.com/1zb/3DShape2VecSet/blob/bedbd1091664be8e2409d580c4e1630df1a37c89/engine_ae.py#L64
        loss_vol = self.criterion(
            reconstructions[:, : self.num_samples],
            labels[:, : self.num_samples],
        )

        loss_near = self.criterion(
            reconstructions[:, self.num_samples :],
            labels[:, self.num_samples :],
        )

        if loss_kl is not None:
            loss = (
                loss_vol + self.near_loss_weight * loss_near + self.kl_weight * loss_kl
            )
        elif loss_vq is not None:
            loss = (
                loss_vol
                + self.near_loss_weight * loss_near
                + self.codebook_weight * loss_vq
            )
            # print("LOSS: ", loss, loss_vol, loss_near, self.codebook_weight, loss_vq)
        else:
            loss = loss_vol + self.near_loss_weight * loss_near
        outputs["loss_vol"] = loss_vol
        outputs["loss_near"] = loss_near

        return loss
