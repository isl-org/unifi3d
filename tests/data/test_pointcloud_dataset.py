import os
import pytest
from omegaconf import OmegaConf
import hydra

DATA_CONF = OmegaConf.load(
    os.path.join("tests", "configs", "shape2vecset_dataset.yaml")
)


@pytest.fixture
def simple_config():
    return DATA_CONF


class TestPointCloudDataset:
    def test_dataset_sample(self, simple_config):
        """Test whether one sample of the dataset looks alright"""
        # setup datamodule
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()
        # get one sample from the dataloader
        dataloader = datamodule.train_dataloader()
        sample_dict = next(iter(dataloader))

        # chec keys
        assert "occupancy" in sample_dict.keys(), "occupancy not generated"
        assert "pcl" in sample_dict.keys(), "pcl not generated"
        assert "query_points" in sample_dict.keys(), "query_points not generated"

        # check shapes
        assert sample_dict["occupancy"].shape == (
            1,
            simple_config.config.num_occu_points,
        ), f"Wrong shape {sample_dict['occupancy'].shape}, should be (1,4096)"

        assert sample_dict["query_points"].shape == (
            1,
            simple_config.config.num_occu_points,
            3,
        ), f"Wrong shape {sample_dict['query_points'].shape}, should be (1,4096, 3)"

        assert sample_dict["pcl"].shape == (
            1,
            simple_config.config.num_samples,
            3,
        ), f"Wrong shape {sample_dict['pcl'].shape}, should be (1,2048,3)"
