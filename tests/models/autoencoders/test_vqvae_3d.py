import os
import pytest
from omegaconf import OmegaConf
import hydra
import torch
import open3d as o3d

from unifi3d.utils.data.sdf_utils import sdf_to_mesh
from unifi3d.models.autoencoders.vqvae_3d import VQVAE


@pytest.fixture
def test_sample_batch():
    """Create a dataset and get one test sample"""
    data_conf = OmegaConf.load(os.path.join("tests", "configs", "sdf_dataset.yaml"))
    data_conf.config.res = 64
    data_conf.config.sample_res = 64
    datamodule = hydra.utils.instantiate(data_conf)
    datamodule.setup()
    data_loader = datamodule.test_dataloader()

    # create iterator
    data_loader_iter = iter(data_loader)
    return next(data_loader_iter)


MODEL_CONF = OmegaConf.load(os.path.join("tests", "configs", "vqvae3d_model.yaml"))
CKPT_PATH = "/path/to/your/checkpoint.pth"  # Replace with your checkpoint path


class TestVQVAE3d:
    def test_encode_wrapper(self, test_sample_batch):
        ae = VQVAE(MODEL_CONF).cuda()
        encoded = ae.encode_wrapper(test_sample_batch)
        assert encoded.shape == (
            2,
            3,
            16,
            16,
            16,
        ), f"Wrong shape {encoded.shape}, should be (1, 16, 16, 16)"

    def test_decode_wrapper(self):
        ae = VQVAE(MODEL_CONF).cuda()
        decoded = ae.decode_wrapper(torch.rand(1, 3, 16, 16, 16).cuda())
        assert decoded.shape == (
            1,
            1,
            64,
            64,
            64,
        ), f"Wrong shape {decoded.shape}, should be (1, 1, 64, 64, 64)"

    def test_encode_decode_w_ckpt(self, test_sample_batch):
        """Test reconstruction"""
        if not os.path.exists(CKPT_PATH):
            pytest.skip(f"Checkpoint not found at {CKPT_PATH}")
        ae = VQVAE(MODEL_CONF).cuda()
        ae.load_checkpoint(CKPT_PATH)
        # encode and decode
        encoded = ae.encode_wrapper(test_sample_batch)
        decoded = ae.decode_wrapper(encoded)
        # # save mesh
        # sdf_np = decoded[0].detach().cpu().numpy()
        # mesh = sdf_to_mesh(sdf_np[0], padding=0, level=0)
        # o3d.io.write_triangle_mesh("test.obj", mesh)

        loss = torch.mean((decoded - test_sample_batch["sdf"]) ** 2)
        assert (
            loss.item() <= 0.002
        ), f"loss with this pretrained model must be <0.002 but is {loss}"
