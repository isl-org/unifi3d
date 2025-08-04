import torch
import torch.nn as nn


class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, batch, output, return_loss_log=False):
        reconstructions = output["sdf_hat"]
        inputs = batch["sdf"]
        codebook_loss = output["codebook_loss"]

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = torch.mean(rec_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        if not return_loss_log:
            return loss

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_codebook": codebook_loss.detach().mean(),
            "loss_nll": nll_loss.detach().mean(),
            "loss_rec": rec_loss.detach().mean(),
        }

        return loss, log


def get_loss_from_output(batch, output):
    return output["batch_loss"]
