# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch.nn
from unifi3d.models.modules import graph_modules, dual_octree
from unifi3d.models.autoencoders import graph_ae


class GraphVQVAE(graph_ae.GraphAE):
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
        quantizer=None,
        ckpt_path=None,
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

        # self.pre_quant_conv = graph_modules.Conv1x1Bn(
        #     self.code_channel, self.code_channel
        # )
        # SDFusion
        self.pre_quant_conv = torch.nn.Conv3d(self.code_channel, self.code_channel, 1)

        self.quantizer = quantizer
        # self.post_quant_conv = graph_modules.Conv1x1BnRelu(
        #     self.code_channel, self.code_channel
        # )
        self.post_quant_conv = torch.nn.Conv3d(self.code_channel, self.code_channel, 1)
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def decode_indices(self, code_indices):
        """
        Decodes the given code 'code_indices' into a reconstructed output.

        Args:
            code_indices (Tensor): The code to be decoded.

        Returns:
            Tensor: The decoded output from the given code.
        """
        quantized_output = self.quantizer.quantize_indices(code_indices)
        return self.decode(quantized_output)

    def encode(self, doctree, in_batch_shape=True, no_quant=False):
        """
        Encodes the input 'doctree' into a latent representation.

        Args:
            doctree (DualOctree): The input data to be encoded.

        Returns:
            Tensor: The quantized latent vector.
            float: The embedding loss from quantization.
            quant_idxs: Additional information from the quantization process.
        """
        encoder_out = super().encode(doctree, in_batch_shape=in_batch_shape)
        print("encoder_out", encoder_out.shape)
        h = self.pre_quant_conv(encoder_out)
        b, n_channels, height, width, depth = h.shape
        if no_quant:
            code = h
        else:
            code, quantize_losses, quant_idxs = self.quantizer(h, is_voxel=True)

        if not in_batch_shape:
            code = code.reshape(-1, n_channels)

        if no_quant:
            return h
        else:
            return code, quantize_losses, quant_idxs

    def decode(self, code, doctree_out=None, update_octree=True, in_batch_shape=True):
        convs = self.post_quant_conv(code)
        out = super().decode(
            convs, doctree_out, update_octree, in_batch_shape=in_batch_shape
        )
        return out

    def encode_wrapper(self, data_sample, in_batch_shape=True):
        """
        Equivalent to extract_code in the original repository.
        """
        octree_in = data_sample["octree_in"].cuda()
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        code = self.encode(doctree_in, in_batch_shape=in_batch_shape, no_quant=True)
        return code

    def forward(self, batch):
        # run encoder
        octree_in = batch["octree_in"]
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        in_batch_shape = True
        quant_output, quantize_losses, quant_idxs = self.encode(
            doctree_in, in_batch_shape=in_batch_shape
        )

        # run decoder
        octree_out = batch["octree_gt"]
        pos = batch["pos"].requires_grad_()

        update_octree = octree_out is None
        if update_octree:
            octree_out = ocnn.create_full_octree(self.full_depth, self.nout)
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        out = self.decode(
            code=quant_output,
            doctree_out=doctree_out,
            update_octree=update_octree,
            in_batch_shape=in_batch_shape,
        )
        output = {
            "logits": out[0],
            "reg_voxs": out[1],
            "octree_out": out[2],
            "quantized_output": quant_output,
            "loss_quantize": quantize_losses,
            "quantized_indices": quant_idxs,
        }

        # compute function value with mpu
        if pos is not None:
            output["mpus"] = self.neural_mpu(pos, out[1], out[2])

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]

        output["neural_mpu"] = _neural_mpu

        batch_size = batch["octree_in"].batch_size
        code_shaped = quant_output.view(batch_size, -1)
        code_min = torch.min(code_shaped, dim=1)
        code_max = torch.max(code_shaped, dim=1)
        code_mean = torch.mean(code_shaped, dim=1)
        output["code_min"] = code_min.values
        output["code_max"] = code_max.values
        output["code_mean"] = code_mean
        return output
