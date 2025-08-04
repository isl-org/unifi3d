# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn
from ocnn.octree import Points, Octree

from unifi3d.utils.model import mpu_utils
from unifi3d.models.modules import graph_modules, dual_octree
from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder


class GraphAE(BaseEncoderDecoder):
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
        layer_norm_encoding=False,
    ):
        super().__init__()
        self.depth = depth
        self.channel_in = channel_in
        self.nout = nout
        self.full_depth = full_depth
        self.depth_out = depth_out
        self.resblk_type = resblk_type
        self.bottleneck = bottleneck
        self.neural_mpu = mpu_utils.NeuralMPU(self.full_depth, self.depth_out)
        self._setup_channels_and_resblks()
        self.layer_norm_encoding = layer_norm_encoding
        n_edge_type, avg_degree = 7, 7

        # encoder
        self.conv1 = graph_modules.GraphConvBnRelu(
            channel_in, self.channels[depth], n_edge_type, avg_degree, depth - 1
        )
        self.encoder = torch.nn.ModuleList(
            [
                graph_modules.GraphResBlocks(
                    self.channels[d],
                    self.channels[d],
                    self.resblk_num[d],
                    bottleneck,
                    n_edge_type,
                    avg_degree,
                    d - 1,
                    resblk_type,
                )
                for d in range(depth, full_depth - 1, -1)
            ]
        )
        self.downsample = torch.nn.ModuleList(
            [
                graph_modules.GraphDownsample(self.channels[d], self.channels[d - 1])
                for d in range(depth, full_depth, -1)
            ]
        )

        # decoder
        self.upsample = torch.nn.ModuleList(
            [
                graph_modules.GraphUpsample(self.channels[d - 1], self.channels[d])
                for d in range(full_depth + 1, depth + 1)
            ]
        )
        self.decoder = torch.nn.ModuleList(
            [
                graph_modules.GraphResBlocks(
                    self.channels[d],
                    self.channels[d],
                    self.resblk_num[d],
                    bottleneck,
                    n_edge_type,
                    avg_degree,
                    d - 1,
                    resblk_type,
                )
                for d in range(full_depth, depth + 1)
            ]
        )

        # header
        self.predict = torch.nn.ModuleList(
            [
                self._make_predict_module(self.channels[d], 2)
                for d in range(full_depth, depth + 1)
            ]
        )
        self.regress = torch.nn.ModuleList(
            [
                self._make_predict_module(self.channels[d], 4)
                for d in range(full_depth, depth + 1)
            ]
        )

        # this is to make the encoder and decoder symmetric
        self.code_channel = code_channel
        self.code_dim = self.code_channel * 2 ** (3 * self.full_depth)
        channel_in = self.channels[self.full_depth]
        self.project1 = graph_modules.Conv1x1Bn(channel_in, self.code_channel)
        self.project2 = graph_modules.Conv1x1BnRelu(self.code_channel, channel_in)

        # Add layer norm
        if self.layer_norm_encoding:
            self.layer_norm = torch.nn.LayerNorm(
                normalized_shape=((2**self.full_depth) ** 3, self.code_channel)
            )
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def _setup_channels_and_resblks(self):
        # self.resblk_num = [3] * 7 + [1] + [1] * 9
        self.resblk_num = [3] * 16
        self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24]

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
        return torch.nn.Sequential(
            graph_modules.Conv1x1BnRelu(channel_in, num_hidden),
            graph_modules.Conv1x1(num_hidden, channel_out, use_bias=True),
        )

    def _get_input_feature(self, doctree):
        return doctree.get_input_feature()

    def encode(self, doctree, in_batch_shape=True):
        depth, full_depth = self.depth, self.full_depth
        data = self._get_input_feature(doctree)
        convs = dict()
        convs[depth] = data
        for i, d in enumerate(range(depth, full_depth - 1, -1)):
            # perform graph conv
            convd = convs[d]  # get convd
            edge_idx = doctree.graph[d]["edge_idx"]
            edge_type = doctree.graph[d]["edge_dir"]  # [0, 1, 2, 3 ,4, 5, 6]
            node_type = doctree.graph[d]["node_type"]  # [0, 1, 2, 3, 4]
            if d == self.depth:  # the first conv
                convd = self.conv1(convd, edge_idx, edge_type, node_type)
            convd = self.encoder[i](convd, edge_idx, edge_type, node_type)
            convs[d] = convd  # update convd

            # downsampleing
            if d > full_depth:  # init convd
                nnum = doctree.nnum[d]
                lnum = doctree.lnum[d - 1]
                leaf_mask = doctree.node_child(d - 1) < 0
                convs[d - 1] = self.downsample[i](convd, leaf_mask, nnum, lnum)
        # reduce the dimension
        code = self.project1(convs[self.full_depth])
        # constrain the code in [-1, 1]
        code = torch.tanh(code)
        batch_size = doctree.batch_size
        _, n_channels = code.shape

        if self.layer_norm_encoding:
            code = code.view(batch_size, -1, n_channels)
            code = self.layer_norm(code)
            code = code.view(-1, n_channels)
        if in_batch_shape:
            bnd = 2**self.full_depth
            code_permuted = code.view(batch_size, -1, n_channels).permute(0, 2, 1)
            # Reshaping to be patch embed3D expected format.
            return code_permuted.view(batch_size, n_channels, bnd, bnd, bnd)
        return code

    def encode_wrapper(self, data_sample, in_batch_shape=True):
        """
        Equivalent to extract_code in the original repository.
        """
        octree_in = data_sample["octree_in"].cuda()
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()
        code = self.encode(doctree_in, in_batch_shape=in_batch_shape)
        return code

    def decode_wrapper(self, code):
        """
        Equivalent to decode_code in the original repository.
        """
        # Assuming input code for wrapper is always in batch shape of (bs, n_tokens, n_channels)
        out = self.decode(code=code, update_octree=True, in_batch_shape=True)
        output = {"logits": out[0], "reg_voxs": out[1], "octree_out": out[2]}

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]

        output["neural_mpu"] = _neural_mpu
        return output

    def _create_full_octree(self, batch_size, device):
        octree_out = Octree(self.full_depth, self.full_depth, batch_size, device)
        for d in range(self.full_depth + 1):
            octree_out.octree_grow_full(depth=d)
        return octree_out

    def decode(self, code, doctree_out=None, update_octree=True, in_batch_shape=True):
        if doctree_out is None:
            assert in_batch_shape
            batch_size = code.shape[0]
            # generate dual octrees
            assert (
                batch_size == 1
            ), f"ocnn.octree_batch function returns segmentation fault when called here on created full octree. Please ensure your inference batch is of size 1 not {batch_size}."
            octree_out = self._create_full_octree(
                batch_size=batch_size, device=code.device
            )
            doctree_out = dual_octree.DualOctree(octree_out)
            doctree_out.post_processing_for_docnn()

        if in_batch_shape:
            batch_size, n_channels, _, _, _ = code.shape
            code_permuted = code.view(batch_size, n_channels, -1).permute(0, 2, 1)
            code = code_permuted.contiguous().view(-1, n_channels)

        logits = dict()
        reg_voxs = dict()
        deconvs = dict()

        deconvs[self.full_depth] = self.project2(code)
        for i, d in enumerate(range(self.full_depth, self.depth_out + 1)):
            if d > self.full_depth:
                nnum = doctree_out.nnum[d - 1]
                leaf_mask = doctree_out.node_child(d - 1) < 0
                deconvs[d] = self.upsample[i - 1](deconvs[d - 1], leaf_mask, nnum)

            edge_idx = doctree_out.graph[d]["edge_idx"]
            edge_type = doctree_out.graph[d]["edge_dir"]
            node_type = doctree_out.graph[d]["node_type"]
            deconvs[d] = self.decoder[i](deconvs[d], edge_idx, edge_type, node_type)

            # predict the splitting label
            logit = self.predict[i](deconvs[d])
            nnum = doctree_out.nnum[d]
            logits[d] = logit[-nnum:]

            # update the octree according to predicted labels
            if update_octree:
                label = logits[d].argmax(1).to(torch.int32)
                octree_out = doctree_out.octree
                octree_out.octree_split(label, d)
                if d < self.depth_out:
                    octree_out.octree_grow(d + 1)
                doctree_out = dual_octree.DualOctree(octree_out)
                doctree_out.post_processing_for_docnn()

            # predict the signal
            reg_vox = self.regress[i](deconvs[d])
            # pad zeros to reg_vox to reuse the original code for ocnn
            node_mask = doctree_out.graph[d]["node_mask"]
            shape = (node_mask.shape[0], reg_vox.shape[1])
            reg_vox_pad = torch.zeros(shape, device=reg_vox.device, dtype=reg_vox.dtype)
            reg_vox_pad[node_mask] = reg_vox
            reg_voxs[d] = reg_vox_pad

        return logits, reg_voxs, doctree_out.octree

    def forward(self, batch):
        # run encoder and decoder
        in_batch_shape = False
        code = self.encode_wrapper(data_sample=batch, in_batch_shape=in_batch_shape)

        # During training use gt as octree_out

        # Assume forward is only called during training.
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
        output = {"logits": out[0], "reg_voxs": out[1], "octree_out": out[2]}

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
