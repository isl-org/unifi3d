import os
import pytest
from omegaconf import OmegaConf
import hydra
import torch
import open3d as o3d

from unifi3d.utils.data.sdf_utils import sdf_to_mesh
from unifi3d.models.autoencoders.pc2sdf_ae import PC2SDF_AE


DATA_CONF = OmegaConf.load(os.path.join("tests", "configs", "pc2sdf_dataset.yaml"))
MODEL_CONF = OmegaConf.load(os.path.join("tests", "configs", "pc2sdf_ae_model.yaml"))


@pytest.fixture
def simple_config():
    return DATA_CONF


class TestPC2SDF_AE:
    def test_encode_wrapper(self, simple_config) -> None:
        # setup datamodule
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()
        # get one sample from the dataloader
        dataloader = datamodule.train_dataloader()
        sample_dict = next(iter(dataloader))

        ae = PC2SDF_AE(MODEL_CONF).cuda()
        encoded = ae.encode_wrapper(sample_dict["pcl"])
        assert encoded.shape == (
            simple_config.batch_size,
            MODEL_CONF.c_dim,
        ), f"Wrong shape {encoded.shape}, should be (B, 128)"

    def test_decode_wrapper(self, simple_config) -> None:
        ae = PC2SDF_AE(MODEL_CONF).cuda()
        decoded = ae.decode_wrapper(
            torch.rand(simple_config.batch_size, MODEL_CONF.c_dim).cuda()
        )
        assert decoded.shape == (
            simple_config.batch_size,
            1,
            simple_config.config.res,
            simple_config.config.res,
            simple_config.config.res,
        ), f"Wrong shape {decoded.shape}, should be (1, 1, 64, 64, 64)"

    # def test_encode_decode_w_ckpt(self, test_sample_batch) -> None:
    #     """Test reconstruction"""
    #     ae = PC2SDF_AE(MODEL_CONF, CKPT_PATH).cuda()
    #     # encode and decode
    #     encoded = ae.encode_wrapper(test_sample_batch)
    #     decoded = ae.decode_wrapper(encoded)
    #     # # save mesh
    #     # sdf_np = decoded[0].detach().cpu().numpy()
    #     # mesh = sdf_to_mesh(sdf_np[0], padding=0, level=0)
    #     # o3d.io.write_triangle_mesh("test.obj", mesh)

    #     loss = torch.mean((decoded - test_sample_batch["sdf"]) ** 2)
    #     assert (
    #         loss.item() <= 0.002
    #     ), f"loss with this pretrained model must be <0.002 but is {loss}"
