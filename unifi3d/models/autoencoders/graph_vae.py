# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn
from torch.distributions import Normal
import numpy as np
from unifi3d.models.modules import dual_octree
from unifi3d.models.autoencoders import graph_ae


class GraphVAE(graph_ae.GraphAE):
    def __init__(
        self,
        depth,
        channel_in,
        nout,
        full_depth=2,
        depth_out=6,
        resblk_type="bottleneck",
        bottleneck=4,
        code_channel=8,
        ckpt_path=None,
        add_kl_loss=True,
        kl_beta=1.0,
    ):
        super().__init__(
            depth=depth,
            channel_in=channel_in,
            nout=nout,
            full_depth=full_depth,
            depth_out=depth_out,
            resblk_type=resblk_type,
            bottleneck=bottleneck,
            code_channel=code_channel,
        )

        # Right now it's default to 8.
        self.encoder_mu_fc = torch.nn.Linear(self.code_channel, self.code_channel)
        self.encoder_var_fc = torch.nn.Linear(self.code_channel, self.code_channel)
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        self.add_kl_loss = add_kl_loss
        self.kl_beta = kl_beta

    def encode(self, doctree, in_batch_shape=True):
        encoder_out = super().encode(doctree, in_batch_shape=in_batch_shape)
        # Add Variational part here:
        # maintain batch size and flatten the latent: already in a projected dimension [8192, 256], including octree and dual octree. Inference time only octree is sampled.
        if in_batch_shape:
            batch_size, channel, h, w, d = encoder_out.shape
            encoder_out = encoder_out.view(batch_size, channel, -1).permute(0, 2, 1)

        else:
            channel = encoder_out.shape[-1]
            batch_size = doctree.batch_size
            encoder_out = encoder_out.view(batch_size, -1, channel)
        mu = self.encoder_mu_fc(encoder_out)
        log_var = self.encoder_var_fc(encoder_out)

        std = torch.exp(0.5 * log_var)
        dist = Normal(mu, std)
        z = dist.rsample()

        if in_batch_shape:
            z = z.permute(0, 2, 1).view(batch_size, channel, h, w, d)
        else:
            z = z.view(-1, channel)
        return z, mu, log_var

    def encode_wrapper(self, data_sample, in_batch_shape=True):
        """
        Equivalent to extract_code in the original repository.
        """
        octree_in = data_sample["octree_in"].cuda()
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        code, _, _ = self.encode(doctree_in, in_batch_shape=in_batch_shape)
        return code

    def forward(self, batch):
        octree_in = batch["octree_in"]
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        in_batch_shape = False
        code, mu, log_var = self.encode(doctree_in, in_batch_shape=in_batch_shape)

        # run decoder
        octree_out = batch["octree_gt"]
        pos = batch["pos"].requires_grad_()

        update_octree = octree_out is None
        if update_octree:
            octree_out = ocnn.create_full_octree(self.full_depth, self.nout)
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        out = self.decode(
            code=code,
            doctree_out=doctree_out,
            update_octree=update_octree,
            in_batch_shape=in_batch_shape,
        )
        # As long as the key contains loss, it will be aggregated in lit_tokenizer
        output = {
            "logits": out[0],
            "reg_voxs": out[1],
            "octree_out": out[2],
        }
        if self.add_kl_loss:
            kl_loss = (
                -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.kl_beta
            )
            output.update({"loss_kl": kl_loss})

        # compute function value with mpu
        if pos is not None:
            output["mpus"] = self.neural_mpu(pos, out[1], out[2])

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]

        output["neural_mpu"] = _neural_mpu
        batch_size = batch["octree_in"].batch_size
        code_shaped = code.view(batch_size, -1)
        code_min = torch.min(code_shaped, dim=1)
        code_max = torch.max(code_shaped, dim=1)
        code_mean = torch.mean(code_shaped, dim=1)
        output["code_min"] = code_min.values
        output["code_max"] = code_max.values
        output["code_mean"] = code_mean
        return output

    def sample(self, num_cells=512, device_str="cuda"):
        latent_shape = (num_cells, self.code_channel)
        code = torch.zeros(*latent_shape).to(torch.device(device_str)).normal_()
        return self.decode_wrapper(code=code)
