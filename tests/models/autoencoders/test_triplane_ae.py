import os
from omegaconf import OmegaConf
import hydra
import torch
import open3d as o3d
import pytest
from tqdm import tqdm

from unifi3d.models.autoencoders.triplane_ae import TriplaneAE


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
    model = TriplaneAE(config)
    ckpt_path = "/path/to/your/checkpoint.pth"  # Replace with your checkpoint path

    # Load the checkpoint
    state_dict = torch.load(ckpt_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    return config, model


@pytest.fixture(scope="module")
def ae(config):
    return TriplaneAE(config)


class TestTriplaneAE:

    @pytest.mark.parametrize("feature_mode", ["image_token", "image", "token"])
    def test_planes_to_token_and_back(self, config, feature_mode):
        config["model"]["triplane_feature_mode"] = feature_mode
        model = TriplaneAE(config)

        batch, _ = get_sample_batch()
        encoded_output = model.encode_inputs(batch["pcl"])

        # just using shape of encoded_output. Using linspace for easier debug
        encoded_output["xy"] = torch.linspace(
            1.0, 2.0, steps=encoded_output["xy"].numel()
        ).view_as(encoded_output["xy"])
        encoded_output["xz"] = torch.linspace(
            2.0, 3.0, steps=encoded_output["xz"].numel()
        ).view_as(encoded_output["xz"])
        encoded_output["yz"] = torch.linspace(
            3.0, 4.0, steps=encoded_output["yz"].numel()
        ).view_as(encoded_output["yz"])

        # Convert planes to tokens
        triplane_features = model.convert_planes_to_token(encoded_output)

        # Convert tokens back to planes
        decoded_dict = model.convert_token_to_planes(triplane_features)

        # Assert that the final output matches the initial input
        for key in encoded_output:
            assert torch.allclose(encoded_output[key], decoded_dict[key], atol=1e-6)

    @pytest.mark.parametrize("feature_mode", ["image_token", "image", "token"])
    def test_token_to_planes_and_back(self, config, feature_mode):
        config["model"]["triplane_feature_mode"] = feature_mode
        model = TriplaneAE(config)

        batch, _ = get_sample_batch()
        encoded_output = model.encode_inputs(batch["pcl"])

        # Convert planes to tokens
        triplane_features = model.convert_planes_to_token(encoded_output)

        # Convert tokens back to planes
        triplane_features_back = model.convert_planes_to_token(
            model.convert_token_to_planes(triplane_features)
        )

        # Assert that the final output matches the initial input
        assert torch.allclose(triplane_features, triplane_features_back, atol=1e-6)

    def test_forward(self, ae):
        batch, _ = get_sample_batch()
        amount_of_points = batch["query_points"].shape[1]
        output = ae(batch)
        assert (
            output["decoded_output"].shape[1] == amount_of_points
        ), "Wrong shape of output"

    def test_encode_input(self, ae, config):
        batch, _ = get_sample_batch()
        encoded_output = ae.encode_inputs(batch["pcl"])
        for key, val in encoded_output.items():
            assert key in config["model"]["encoder_kwargs"]["plane_type"]
            assert val.shape[0] == 1
            assert val.shape[1] == config["model"]["c_dim"]
            assert val.shape[2] == config["model"]["encoder_kwargs"]["plane_resolution"]
            assert val.shape[3] == config["model"]["encoder_kwargs"]["plane_resolution"]

    def test_decode(self, ae):
        batch, _ = get_sample_batch()

        latent_code = ae.encode_inputs(batch["pcl"])

        kwargs = {}

        output = ae.decode(batch["query_points"], latent_code, **kwargs)
        amount_of_points = batch["query_points"].shape[1]
        assert output.shape[1] == amount_of_points, "Wrong shape of output"

    def test_encode_wrapper(self, ae, config):
        batch, _ = get_sample_batch()
        encoded_output = ae.encode_wrapper(batch)
        resolution = config["model"]["encoder_kwargs"]["plane_resolution"]
        channels = config["model"]["c_dim"]
        batch_size = 1
        if config["model"]["triplane_feature_mode"] == "image_token":
            assert len(encoded_output.shape) == 3
            assert encoded_output.shape[0] == batch_size
            assert encoded_output.shape[1] == resolution**2 * 3
            assert encoded_output.shape[2] == channels
        elif config["model"]["triplane_feature_mode"] == "image":
            assert len(encoded_output.shape) == 4
            assert encoded_output.shape[0] == batch_size
            assert encoded_output.shape[1] == channels
            assert encoded_output.shape[2] == resolution
            assert encoded_output.shape[3] == resolution * 3
        elif config["model"]["triplane_feature_mode"] == "token":
            assert len(encoded_output.shape) == 3
            assert encoded_output.shape[0] == batch_size
            assert encoded_output.shape[1] == resolution**2
            assert encoded_output.shape[2] == 3 * channels
        else:
            raise NotImplementedError(
                f"Feature mode {config['model']['triplane_feature_mode']} not implemented"
            )

    def test_decode_wrapper_dict(self, ae):
        batch, _ = get_sample_batch()

        latent_code = ae.encode_inputs(batch["pcl"])

        output = ae.decode_wrapper(latent_code, query_points=batch["query_points"])
        amount_of_points = batch["query_points"].shape[1]
        assert output.shape[1] == amount_of_points, "Wrong shape of output"


if __name__ == "__main__":
    test_obj = TestTriplaneAE()
    # feature_modes = ["image_token", "image", "token"]
    feature_modes = ["image_channel_stack"]
    config = load_config(CONFIG_PATH)
    # Create an instance of the test class
    for feature_mode in feature_modes:
        test_obj.test_planes_to_token_and_back(config, feature_mode)
        test_obj.test_token_to_planes_and_back(config, feature_mode)
        # Load the configuration and model

        cfg = load_config(CONFIG_PATH)
        cfg.model.triplane_feature_mode = feature_mode
        cfg, model = initialize_model(cfg)
        # # Manually run the test methods
        # test_obj.test_forward(model)
        # test_obj.test_encode_input(model, cfg)
        # test_obj.test_decode(model)
        # test_obj.test_encode_wrapper(model, cfg)
        # test_obj.test_decode_wrapper_dict(model)

        name = f"test_generation_{feature_mode}.off"
        test_obj.test_generation(model, name)
    print("All tests passed!")
