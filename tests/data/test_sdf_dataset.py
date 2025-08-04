import os
import pytest
import open3d as o3d
from omegaconf import OmegaConf
import hydra
import torch
import numpy as np
from unifi3d.data.data_iterators import TestIterator


@pytest.fixture
def example_mesh():
    return o3d.data.KnotMesh()


DATA_CONF = OmegaConf.load(os.path.join("tests", "configs", "sdf_dataset.yaml"))


@pytest.fixture
def simple_config():
    return DATA_CONF


class TestSDFDataset:
    def test_init_dataset(self, simple_config):
        """Test whether the dataset is correctly initialized"""
        # setup datamodule
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()

        # check one sample from the train dataloader (pure meshes)
        dataloader = datamodule.train_dataloader()
        sample_dict = next(iter(dataloader))
        assert "sdf" in sample_dict.keys(), "SDF not generated"
        sample = sample_dict["sdf"]
        assert sample.shape == (
            2,
            1,
            16,
            16,
            16,
        ), f"Wrong shape {sample.shape}, should be (2,1,16,16,16)"
        assert isinstance(
            sample, torch.Tensor
        ), f"Wrong type {type(sample)}, should be torch.Tensor"
        assert np.isclose(
            torch.max(sample).item(), 0.2
        ), f"Due to clamping, max should be 0.2, but is {torch.max(sample)}"
        assert (
            torch.min(sample).item() >= -0.2
        ), f"Due to clamping, min should be -0.2, but is {torch.min(sample)}"

        # check one sample from the val dataloader (already preprocessed)
        dataloader = datamodule.val_dataloader()
        sample_dict = next(iter(dataloader))
        assert "sdf" in sample_dict.keys(), "SDF not generated"
        sample = sample_dict["sdf"]
        assert sample.shape == (
            2,
            1,
            16,
            16,
            16,
        ), f"Wrong shape {sample.shape}, should be (2,1,16,16,16)"
        assert isinstance(
            sample, torch.Tensor
        ), f"Wrong type {type(sample)}, should be torch.Tensor"

    def test_length(self, simple_config):
        """Test whether lengths of dataset are alright"""
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()
        assert len(datamodule.data_train) == 10
        assert len(datamodule.data_test) == 3
        assert len(datamodule.data_val) == 100

    def test_clamp(self, simple_config):
        simple_config.config.thresh_clamp_sdf = 0.1
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()
        sample = datamodule.data_train[0]["sdf"]
        assert torch.max(sample) == 0.1, "Max must be lower than clamp threshold"
