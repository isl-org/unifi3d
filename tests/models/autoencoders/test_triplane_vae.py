import os
from omegaconf import OmegaConf
import hydra
import torch
import open3d as o3d
import pytest
from tqdm import tqdm

from unifi3d.models.autoencoders.triplane_vae import TriplaneVAE


def load_config(config_path):
    """Load the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return OmegaConf.load(config_path)


def get_sample_batch():
    """Create a dataset and get one test sample"""
    data_conf = OmegaConf.load(
        os.path.join("tests", "configs", "triplane_dataset.yaml")
    )
    datamodule = hydra.utils.instantiate(data_conf)
    datamodule.setup()
    data_loader = datamodule.test_dataloader()

    # create iterator
    data_loader_iter = iter(data_loader)
    return next(data_loader_iter), data_loader


CONFIG_PATH = (
    f"{os.path.dirname(os.path.abspath(__file__))}/../../configs/triplane_model.yaml"
)


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def initialize_model(config=None):
    config = load_config(CONFIG_PATH) if config is None else config
    model = TriplaneVAE(config)
    return config, model


@pytest.fixture(scope="module")
def ae(config=None):
    cfg = load_config(CONFIG_PATH) if config is None else config
    cfg.model.triplane_feature_mode = "image"
    cfg, model = initialize_model(cfg)
    assert model.config.model.triplane_feature_mode == "image"
    return model


class TestTriplaneVAE:

    def test_forward(self, ae):
        batch, _ = get_sample_batch()
        amount_of_points = batch["query_points"].shape[1]
        output = ae(batch)
        assert (
            output["decoded_output"].logit().shape[1] == amount_of_points
        ), "Wrong shape of output"
