import torch


class PC2SDFLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, output, return_loss_log=False):
        loss = output["loss"]
        loss_sdf = output["loss_sdf"]
        loss_reg = output["loss_reg"]

        if not return_loss_log:
            return loss

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_sdf": loss_sdf.detach().mean(),
            "loss_reg": loss_reg.detach().mean(),
            "loss_sdf_near": output["loss_near"],
            "loss_sdf_far": output["loss_far"],
        }

        return loss, log
