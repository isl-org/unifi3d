import os
import pytest
import open3d as o3d
from omegaconf import OmegaConf
import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
from unifi3d.data.dualoctree_dataset import collate_func
import pytest


def _test_sample_batch(batch_size):
    """Create a dataset and get one test sample"""
    data_conf = OmegaConf.load(
        os.path.join("tests", "configs", "shapenet_dataset.yaml")
    )
    data_conf.dataset.num_samples = 10
    dataset = hydra.utils.instantiate(data_conf)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        collate_fn=collate_func,
        pin_memory=True,
    )

    # create iterator
    data_loader_iter = iter(data_loader)
    return next(data_loader_iter)


@pytest.mark.parametrize(
    "encode_net_name, batch_size",
    [
        ("doctree_ae_model", 1),
        ("doctree_ae_model", 2),
        ("doctree_vae_model", 1),
        ("doctree_vae_model", 2),
        ("doctree_vqvae_model", 1),
        ("doctree_vqvae_model", 2),
    ],
)
def test_encode_wrapper(encode_net_name, batch_size):
    model_conf = OmegaConf.load(
        os.path.join("tests", "configs", f"{encode_net_name}.yaml")
    )

    encode_model = hydra.utils.instantiate(model_conf).cuda()
    test_sample_batch = _test_sample_batch(batch_size)
    encoded = encode_model.encode_wrapper(test_sample_batch)
    target_shape = (batch_size, 8, 4, 4, 4)
    assert (
        encoded.shape == target_shape
    ), f"Wrong shape {encoded.shape}, should be {target_shape}"


@pytest.mark.parametrize(
    "encode_net_name", [("doctree_ae_model"), ("doctree_vae_model")]
)
def test_decode_wrapper(encode_net_name):
    target_keys = [2, 3, 4, 5, 6]
    model_conf = OmegaConf.load(
        os.path.join("tests", "configs", f"{encode_net_name}.yaml")
    )
    encode_model = hydra.utils.instantiate(model_conf).cuda()
    decoded = encode_model.decode_wrapper(torch.rand(1, 8, 4, 4, 4).cuda())
    print(decoded.keys())
    assert "logits" in decoded
    assert "reg_voxs" in decoded
    assert "octree_out" in decoded
    assert "neural_mpu" in decoded

    logits = decoded["logits"]
    print(f"{len(logits)}logits")
    first_item_shape = logits[target_keys[0]].shape
    first_item_target_shape = (64, 2)
    assert (
        first_item_shape == first_item_target_shape
    ), f"Wrong shape {first_item_shape}, should be {first_item_target_shape}"
    for key, target_key in zip(logits, target_keys):
        # print(key, target_key)
        assert key == target_key
        check_shape = logits[key].shape
        # print(key, check_shape)
        assert check_shape[1] == 2, f"Wrong channel {check_shape}, should be (x, 2) "
    reg_voxs = decoded["reg_voxs"]
    for key, target_key in zip(reg_voxs, target_keys):
        assert key == target_key
        check_shape = reg_voxs[key].shape
        assert check_shape[1] == 4, f"Wrong shape {check_shape}, should be (x, 4)"
    depth = decoded["octree_out"].depth
    full_depth = decoded["octree_out"].full_depth
    # nnum = decoded["octree_out"].nnum
    # nenum = decoded["octree_out"].nnum_nempty
    # target_nnum = [1, 8, 64, 120, 480, 1320, 3784]
    # target_nenum = [1, 8, 15, 60, 165, 473, 3447]

    target_depth = 6
    target_full_depth = 2

    assert depth == target_depth, f"Wrong depth {depth}, should be {target_depth}"

    assert (
        full_depth == target_full_depth
    ), f"Wrong full depth {full_depth}, should be {target_full_depth}"
