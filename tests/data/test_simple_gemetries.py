from omegaconf import OmegaConf
import os
import hydra
import torch
import pytest
from functools import partial
import numpy as np

from unifi3d.utils.data.dataset_utils import (
    sample_surface_sphere,
    sdf_sphere,
    sample_surface_cube,
    sdf_cube,
    sample_surface_cylinder,
    sdf_cylinder,
    sample_surface_cone,
    sdf_cone,
    compute_occupancy_and_sdf,
)

EXPECTED_KEYS = ["pcl", "sdf", "query_points", "occupancy"]

default_data_cfg = OmegaConf.load(
    os.path.join("tests", "configs", "simple_geometries.yaml")
)


@pytest.mark.parametrize(
    "mode, shape_names",
    [("train", ["sphere", "cube"]), ("test", ["cube"]), ("val", ["cylinder"])],
)
def test_simple_geometries(mode, shape_names):
    target_num_samples = len(shape_names)
    if target_num_samples == 0:
        target_num_samples = 4
    if mode == "train":
        cfg_dataset_iterator = default_data_cfg.data_train.copy()
    elif mode == "test":
        cfg_dataset_iterator = default_data_cfg.data_test.copy()
    else:
        assert mode == "val"
        cfg_dataset_iterator = default_data_cfg.data_val.copy()
    cfg_dataset_iterator.config.shape_names = shape_names

    simple_geometries_iterator = hydra.utils.instantiate(cfg_dataset_iterator)
    num_samples = len(simple_geometries_iterator)
    assert num_samples == target_num_samples, f"{num_samples} != {target_num_samples}"
    iter_obj = iter(simple_geometries_iterator)
    first_item = next(iter_obj)
    assert len(first_item.keys()) == len(EXPECTED_KEYS)
    for key in EXPECTED_KEYS:
        assert key in first_item, f"{key} not found"

    points = first_item["pcl"]
    assert points.dtype == torch.float32
    assert torch.all(points >= -0.5) and torch.all(points <= 0.5)
    assert points.shape == (cfg_dataset_iterator.config.num_surface_points, 3)

    query_points = first_item["query_points"]
    assert query_points.dtype == torch.float32
    assert torch.all(query_points >= -0.5) and torch.all(points <= 0.5)
    assert query_points.shape == (cfg_dataset_iterator.config.num_query_points, 3)

    sdf = first_item["sdf"]
    assert sdf.dtype == torch.float32
    assert sdf.shape == (cfg_dataset_iterator.config.num_query_points,)

    occupancy = first_item["occupancy"]
    assert occupancy.dtype == torch.float32
    assert torch.all(occupancy >= 0) and torch.all(occupancy <= 1)


def test_simple_geometries_dataloader():
    default_data_cfg.data_train.config.shape_names = []
    default_data_cfg.train_batch_size = 4
    datamodule = hydra.utils.instantiate(default_data_cfg)
    datamodule.setup()
    data_loader = datamodule.train_dataloader()
    data_loader_iter = iter(data_loader)
    item = next(data_loader_iter)
    assert len(item.keys()) == len(EXPECTED_KEYS)
    for key in EXPECTED_KEYS:
        assert key in item

    B = default_data_cfg.train_batch_size
    NS = default_data_cfg.data_train.config.num_surface_points
    NQ = default_data_cfg.data_train.config.num_query_points
    D = 3
    points = item["pcl"]
    assert points.dtype == torch.float32
    assert points.shape == (B, NS, D)

    query_points = item["query_points"]
    assert query_points.dtype == torch.float32
    assert query_points.shape == (B, NQ, D)

    sdf = item["sdf"]
    assert sdf.dtype == torch.float32
    assert sdf.shape == (B, NQ)

    occupancy = item["occupancy"]
    assert occupancy.dtype == torch.float32
    assert occupancy.shape == (B, NQ)


R = 0.5
side = R * 3.0 / 4.0
radius = R / 2.0


def test_sphere():
    # Check on surface
    points = sample_surface_sphere(num_points=2000, radius=radius)
    partial_sdf = partial(sdf_sphere, radius=radius)
    sdf_values, occupancy = compute_occupancy_and_sdf(points, partial_sdf)
    assert np.allclose(sdf_values, 0, atol=1e-6)
    assert np.allclose(occupancy, 1, atol=1e-6)

    # print("points range", points.min(), points.max())
    # Check inside
    outer_samples = sample_surface_sphere(num_points=2000, radius=radius * 1.2)
    sdf_values, occupancy = compute_occupancy_and_sdf(outer_samples, partial_sdf)
    # print("outer_samples range", outer_samples.min(), outer_samples.max())

    assert np.all(sdf_values > 0)
    assert np.allclose(occupancy, 0, atol=1e-6)

    # Check outside
    inner_samples = sample_surface_sphere(num_points=2000, radius=radius * 0.5)
    sdf_values, occupancy = compute_occupancy_and_sdf(inner_samples, partial_sdf)
    # print("inner_samples range", inner_samples.min(), inner_samples.max())

    assert np.all(sdf_values < 0)
    assert np.allclose(occupancy, 1, atol=1e-6)


def test_cube():
    points = sample_surface_cube(num_points=2000, side_length=side)
    partial_sdf = partial(sdf_cube, side=side)
    sdf_values, occupancy = compute_occupancy_and_sdf(points, partial_sdf)
    assert np.allclose(sdf_values, 0, atol=1e-6)
    assert np.allclose(occupancy, 1, atol=1e-6)

    outer_samples = sample_surface_cube(num_points=2000, side_length=side * 1.2)
    sdf_values, occupancy = compute_occupancy_and_sdf(outer_samples, partial_sdf)
    assert np.all(sdf_values > 0), sdf_values
    assert np.allclose(occupancy, 0, atol=1e-6)

    inner_samples = sample_surface_cube(num_points=2000, side_length=side * 0.8)
    sdf_values, occupancy = compute_occupancy_and_sdf(inner_samples, partial_sdf)
    assert np.all(sdf_values < 0), sdf_values
    assert np.allclose(occupancy, 1, atol=1e-6)


def test_cylinder():
    points = sample_surface_cylinder(num_points=2000, radius=radius, height=side)
    partial_sdf = partial(sdf_cylinder, radius=radius, height=side)
    sdf_values, occupancy = compute_occupancy_and_sdf(points, partial_sdf)
    assert np.allclose(sdf_values, 0, atol=1e-6)
    assert np.allclose(occupancy, 1, atol=1e-6)

    outer_samples = sample_surface_cylinder(
        num_points=2000, radius=radius * 1.5, height=side * 1.5
    )
    sdf_values, occupancy = compute_occupancy_and_sdf(outer_samples, partial_sdf)
    assert np.all(sdf_values > 0), sdf_values
    assert np.allclose(occupancy, 0, atol=1e-6)

    inner_samples = sample_surface_cylinder(
        num_points=2000, radius=radius * 0.5, height=side * 0.5
    )
    sdf_values, occupancy = compute_occupancy_and_sdf(inner_samples, partial_sdf)
    assert np.all(sdf_values < 0), sdf_values
    assert np.allclose(occupancy, 1, atol=1e-6)


def test_cone():
    points = sample_surface_cone(num_points=2000, radius=radius, height=side)
    partial_sdf = partial(sdf_cone, radius=radius, height=side)
    sdf_values, occupancy = compute_occupancy_and_sdf(points, partial_sdf)
    assert np.allclose(sdf_values, 0, atol=1e-6)
    assert np.allclose(occupancy, 1, atol=1e-6)

    outer_samples = sample_surface_cone(
        num_points=2000, radius=radius * 1.2, height=side * 1.2
    )
    sdf_values, occupancy = compute_occupancy_and_sdf(outer_samples, partial_sdf)
    assert np.all(sdf_values > 0), sdf_values
    assert np.allclose(occupancy, 0, atol=1e-6)

    inner_samples = sample_surface_cone(
        num_points=2000, radius=radius * 0.5, height=side * 0.5
    )
    sdf_values, occupancy = compute_occupancy_and_sdf(inner_samples, partial_sdf)
    assert np.all(sdf_values < 0), sdf_values
    assert np.allclose(occupancy, 1, atol=1e-6)
