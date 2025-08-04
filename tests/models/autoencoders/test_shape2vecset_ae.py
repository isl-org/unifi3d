import os
import pytest
import open3d as o3d
from omegaconf import OmegaConf
import hydra
import torch
import numpy as np

from unifi3d.models.autoencoders.shape2vecset_ae import Shape2VecSetAE

CKPT_PATH = "/path/to/your/checkpoint.pth"  # Replace with your checkpoint path


@pytest.fixture
def test_sample_batch():
    """Create a dataset and get one test sample"""
    data_conf = OmegaConf.load(
        os.path.join("tests", "configs", "shape2vecset_dataset.yaml")
    )
    datamodule = hydra.utils.instantiate(data_conf)
    datamodule.setup()
    data_loader = datamodule.test_dataloader()

    # create iterator
    data_loader_iter = iter(data_loader)
    return next(data_loader_iter)


class TestShape2VecSetAE:
    def test_encode_wrapper(self, test_sample_batch):
        ae = Shape2VecSetAE(density_grid_points=32).cuda()
        encoded = ae.encode_wrapper(test_sample_batch)
        assert encoded.shape == (
            1,
            512,
            512,
        ), f"Wrong shape {encoded.shape}, should be (1, 512, 512)"

    def test_decode_wrapper(self):
        ae = Shape2VecSetAE(density_grid_points=32).cuda()
        decoded = ae.decode_wrapper(torch.rand(1, 512, 512).cuda())
        assert decoded.shape == (
            1,
            33,
            33,
            33,
        ), f"Wrong shape {decoded.shape}, should be (1, 33, 33, 33)"

    def test_encode_decode_w_ckpt(self, test_sample_batch):
        """Test reconstruction"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip(f"Checkpoint not found at {CKPT_PATH}")
        ae = Shape2VecSetAE(density_grid_points=32).cuda()
        ae.load_checkpoint(CKPT_PATH)
        # encode and decode
        encoded = ae.encode(test_sample_batch["pcl"])
        decoded = ae.decode(encoded, test_sample_batch["query_points"].cuda())

        pred = torch.sigmoid(decoded).detach().cpu().numpy().squeeze()
        gt = test_sample_batch["occupancy"].detach().cpu().numpy().squeeze()

        assert (
            np.mean(pred[gt == 0]) < 0.5
        ), "predicted occupancy for empty voxels too high"
        assert (
            np.mean(pred[gt == 1]) > 0.5
        ), "predicted occupancy for occupied voxels too low"

        loss = torch.mean(
            (torch.sigmoid(decoded) - test_sample_batch["occupancy"].cuda()) ** 2
        )
        assert (
            loss.item() <= 0.4
        ), f"loss with this pretrained model must be <0.4 but is {loss}"
