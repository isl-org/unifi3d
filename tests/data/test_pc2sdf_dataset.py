import os
import pytest
from omegaconf import OmegaConf
import hydra

DATA_CONF = OmegaConf.load(os.path.join("tests", "configs", "pc2sdf_dataset.yaml"))


@pytest.fixture
def simple_config():
    return DATA_CONF


class TestPointCloudSDFDataset:
    def test_dataset_sample(self, simple_config):
        """Test whether one sample of the dataset looks alright"""
        # setup datamodule
        datamodule = hydra.utils.instantiate(simple_config)
        datamodule.setup()
        # get one sample from the dataloader
        dataloader = datamodule.train_dataloader()
        sample_dict = next(iter(dataloader))

        # chec keys
        assert "sdf" in sample_dict.keys(), "sdf not generated"
        assert "pcl" in sample_dict.keys(), "pcl not generated"
        assert "query_points" in sample_dict.keys(), "query_points not generated"

        # check shapes
        res = simple_config.config.res
        assert sample_dict["sdf"].shape == (
            simple_config.batch_size,
            1,
            res,
            res,
            res,
        ), f"Wrong shape {sample_dict['sdf'].shape}, should be (B, 1, {res}, {res}, {res})"

        assert sample_dict["query_points"].shape == (
            simple_config.batch_size,
            simple_config.config.num_near + simple_config.config.num_far,
            3,
        ), f"Wrong shape {sample_dict['query_points'].shape}, should be (B, N, 3)"

        assert sample_dict["pcl"].shape == (
            simple_config.batch_size,
            simple_config.config.num_samples,
            3,
        ), f"Wrong shape {sample_dict['pcl'].shape}, should be (B, M, 3)"
