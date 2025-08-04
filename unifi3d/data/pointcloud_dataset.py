import os
import json
import open3d as o3d
import numpy as np
from typing import Optional
import torch
import warnings
import logging
import time
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from unifi3d.data.base_dataset import BaseDataset
from unifi3d.utils.data.mesh_utils import (
    load_normalize_mesh,
    load_npz_mesh,
    load_npz_pcl,
    save_o3d_pointcloud,
    save_np_pointcloud,
    save_o3d_mesh,
)

import rootutils
import sys

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)

from unifi3d.utils.data.sdf_utils import (
    generate_query_points,
    mesh_to_sdf,
    sdf_to_mesh,
    downsample_grid_sdf,
    coord_grid_from_bbox,
)
import unifi3d.utils.triplane_utils.data as data_triplane
from unifi3d.utils.rendering.cross_section import plot_cross_sections
from torchvision import transforms


def get_occupancy_and_query_points_from_sdf(
    data, sdf, query_points, sampled_inds, threshold=0.0
):
    data["query_points"] = torch.from_numpy(query_points[sampled_inds]).float()
    sampled_occu = (sdf[sampled_inds] < threshold).astype(int)
    data["occupancy"] = torch.from_numpy(sampled_occu).float()
    return data


def get_points_from_occupancy(occupancy, query_points):

    occupied_points = query_points[occupancy == 1]
    unoccupied_points = query_points[occupancy == 0]
    return occupied_points, unoccupied_points


class PointCloudDataset(BaseDataset):
    """
    PointCloudDataset to sample point clouds from meshes (for training Shape2VecSet)
    """

    def __init__(
        self,
        dataset,
        device="cuda",
        num_samples: int = 2048,
        num_occu_points: int = 4096,
        level: float = 0,
        padding: float = 0.2,
        sdf_fn: str = "sdf_octree_depth_6.npz",
        sampling_strategy: str = "near_far",
        threshold_near: float = 0.02,
        fraction_near: float = 0.5,
        sdf_threshold: float = 0.0,
        transform_data=False,
        sdf_file="sdf_octree_path",
        save_mesh=False,
    ):
        """
        Initializes the dataset object with configuration and mode.
        Arguments:
            threshold_near: threshold if sampling_strategy=near_far
            fraction_near: fraction of points to sample near the surface if
                sampling_strategy=near_far
        """
        super(PointCloudDataset, self).__init__(dataset, device=device)
        self.num_samples = num_samples
        self.num_occu_points = num_occu_points
        self.level = level
        self.mode = self.data.mode
        self.padding = padding
        self.sdf_threshold = sdf_threshold
        self.transform_data = transform_data
        self.sampling_strategy = sampling_strategy
        self.sdf_file = sdf_file

        self.save_mesh = save_mesh
        assert self.sampling_strategy in ["probabilities", "near_far", "uniform"]

        # if stragey is near_far, need to set number of points
        self.threshold_near = threshold_near
        self.num_near = int(fraction_near * self.num_occu_points)
        self.num_far = self.num_occu_points - self.num_near

        # if data is an object, we need to sample points from it to encode it
        # (those are subsampled latet to 2048 points so doesn't really matter)
        self.sample_res_pts = 64
        # generate query points for sampled sdf -> between -0.5 and 0.5
        self.query_points = generate_query_points(
            grid_resolution=self.sample_res_pts, padding=self.padding
        )
        self.sdf_fn = sdf_fn

        # for single test sample: only compute data once to save training time
        self.single_sample = None  # by default data is loaded at runtime
        if self.data.path == "knot":
            self.single_sample = self.mesh_to_representation(self.data[0])

        self.data_saved = False
        self.data_iter = 0

    def get_sdf_and_query_points(self, data):

        sdf_data = np.load(data[self.sdf_file])
        if self.sdf_file == "sdf_octree_path":
            query_points = sdf_data["points"]  # query points are between -0.5 and 0.5
            sdf = sdf_data["sdf"]
        elif self.sdf_file == "sdf_grid256_path" or self.sdf_file == "sdf_grid128_path":
            sdf = sdf_data["sdf"]
            res = sdf.shape[0]
            query_points = coord_grid_from_bbox(
                res, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
            )
            sdf = sdf.flatten()
            query_points = query_points.reshape(-1, 3)
            assert query_points.shape[0] == sdf.shape[0]
        else:
            raise ValueError(f"SDF file {self.sdf_file} not implemented")
        return sdf, query_points

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.
        """

        # for training and validation, load point cloud and occupancy at query points
        data = self.data[index]

        # At inference time, we sample the point cloud & sdf from the input mesh
        if "pointcloud_path" not in data.keys() or "sdf_octree_path" not in data.keys():
            warnings.warn("Preprocessed data not found. Generating data from mesh.")

            # load mesh and convert to point cloud - skipped if stored in memory
            if self.single_sample is None:
                data = self.mesh_to_representation(data)
            else:
                data = self.single_sample.copy()

            # sample points from surfaces of mesh
            sampled_vertices = np.asarray(
                data["mesh"]
                .sample_points_uniformly(number_of_points=self.num_samples)
                .points
            )
            # transform to point cloud
            data["pcl"] = torch.from_numpy(sampled_vertices).float().cuda()

            if self.sampling_strategy == "probabilities":
                data = self.sample_points_probabilities(
                    data, data["sdf"], self.query_points, data["sample_probs"]
                )
            elif self.sampling_strategy == "near_far":
                data = self.sample_points_near_far(data, data["sdf"], self.query_points)
            else:
                raise ValueError("Sampling strategy not implemented")

            # delete mesh and sdf to save memory (also bc mesh cannot be batched)
            del data["sdf"]
            del data["mesh"]
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)
            return data

        # Load precomputed points and sdf values
        # load pointcloud
        pcl_path = data["pointcloud_path"]
        point_cloud = np.load(pcl_path)["points"]

        if self.transform_data:
            transform = transforms.Compose(
                [
                    data_triplane.SubsamplePointcloud(self.num_samples),
                    data_triplane.PointcloudNoise(0.005),
                ]
            )
            point_cloud_dict = {None: point_cloud}
            point_cloud = transform(point_cloud_dict)[None]
            data["pcl"] = torch.from_numpy(point_cloud).float().cuda()
        else:
            pcl_subset_idx = np.random.permutation(len(point_cloud))[: self.num_samples]
            data["pcl"] = torch.from_numpy(point_cloud[pcl_subset_idx]).float().cuda()

        # load query points and occupancy
        sdf, query_points = self.get_sdf_and_query_points(data)
        if self.sampling_strategy == "probabilities":
            # sample points with higher prob if closer to surface
            sample_probs = 1 / np.clip(np.abs(sdf), 0.01, 0.3) / 10000
            sample_probs /= sample_probs.sum()
            self.sample_points_probabilities(data, sdf, query_points, sample_probs)
        elif self.sampling_strategy == "near_far":
            data = self.sample_points_near_far(data, sdf, query_points)
        elif self.sampling_strategy == "uniform":

            num_occu_points = (
                query_points.shape[0]
                if self.mode == "val" or self.mode == "test"
                else self.num_occu_points
            )

            points_transform = data_triplane.SubsamplePoints(num_occu_points)
            data_full = {}
            data_full = get_occupancy_and_query_points_from_sdf(
                data_full,
                sdf,
                query_points,
                list(range(query_points.shape[0])),
                threshold=self.sdf_threshold,
            )
            points_occupied = {
                None: data_full["query_points"],
                "occ": data_full["occupancy"],
            }
            data_full = points_transform(points_occupied)
            data["query_points"] = data_full[None]
            data["occupancy"] = data_full["occ"]
        else:
            raise ValueError("Sampling strategy not implemented")

        if self.save_mesh:
            self.data_iter += 1
            path = "outputs/pcl_dataset/"

            if self.data_iter % 10 == 0:
                save_np_pointcloud(
                    data["pcl"].cpu().numpy(), path, f"pcl{self.data_iter}"
                )
                occ, unocc = get_points_from_occupancy(
                    data["occupancy"].cpu().numpy(), data["query_points"].cpu().numpy()
                )
                print(f"Occupied points: {occ.shape[0]}/{data['occupancy'].shape[0]}")
                print(
                    f"Unoccupied points: {unocc.shape[0]}/{data['occupancy'].shape[0]}"
                )

                save_np_pointcloud(occ, path, f"occ{self.data_iter}")
                save_np_pointcloud(unocc, path, f"unocc{self.data_iter}")

            if not self.data_saved:
                self.data_idx = index

                mesh = load_npz_mesh(data["mesh_path"])
                save_o3d_mesh(mesh, path, "mesh")

                pcl_full = load_npz_pcl(pcl_path)
                save_o3d_pointcloud(pcl_full, path, f"pcl_full{index}")

                # occupancy and points full:
                data_full = {}

                data_full = get_occupancy_and_query_points_from_sdf(
                    data_full,
                    sdf,
                    query_points,
                    list(range(query_points.shape[0])),
                    threshold=self.sdf_threshold,
                )
                occ, unocc = get_points_from_occupancy(
                    data_full["occupancy"].cpu().numpy(),
                    data_full["query_points"].cpu().numpy(),
                )
                save_np_pointcloud(occ, path, f"occ_full{index}")
                save_np_pointcloud(unocc, path, f"unocc_full{index}")
        return data

    def sample_points_probabilities(self, data, sdf, query_points, sample_probs):
        """
        Sample query points with corresponding ground truth occupancy by setting the
        sample probabilities inverse to the distance from the surface
        """
        # sample from indices
        num_occu_points = (
            query_points.shape[0]
            if self.mode == "val" or self.mode == "test"
            else self.num_occu_points
        )

        sampled_inds = np.random.choice(
            np.arange(len(sdf)),
            p=sample_probs,
            size=num_occu_points,
            replace=False,
        )
        data = get_occupancy_and_query_points_from_sdf(
            data, sdf, query_points, sampled_inds, threshold=self.sdf_threshold
        )
        return data

    def sample_points_near_far(self, data, sdf, query_points):
        """
        Sample a fraction of the query points from near and the other part from far
        based on threshold
        """
        if self.mode == "train":
            # find near and far indicies
            abs_sdf = np.abs(sdf)
            near_inds = np.where(abs_sdf < self.threshold_near)[0]
            far_inds = np.where(abs_sdf >= self.threshold_near)[0]

            # sample -> try with replace=False but might fail if not enough near points
            try:
                near_inds_sample = np.random.choice(
                    near_inds, self.num_near, replace=False
                )
            except ValueError:
                near_inds_sample = np.random.choice(
                    near_inds, self.num_near, replace=True
                )
            far_inds_sample = np.random.choice(far_inds, self.num_far, replace=False)

            # combine indices and index points and occupancy
            sampled_inds = np.concatenate([near_inds_sample, far_inds_sample])
        else:
            sampled_inds = list(range(query_points.shape[0]))
        data = get_occupancy_and_query_points_from_sdf(
            data, sdf, query_points, sampled_inds, threshold=self.sdf_threshold
        )
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        return data

    def mesh_to_representation(self, data):
        """Loads mesh and converts into a sampled point cloud"""
        # read mesh
        mesh_scale = 1 - self.padding
        mesh = load_normalize_mesh(data["file_path"], mesh_scale)
        data["mesh"] = mesh
        data["sdf"] = (
            mesh_to_sdf(
                mesh,
                sample_res=64,
                grid_resolution=self.sample_res_pts,
                center_mesh=True,
            )
            .numpy()
            .flatten()
        )
        if self.sampling_strategy == "probabilities":
            # sample query points and occupancy from sampled SDF
            sample_probs = 1 / np.clip(np.abs(data["sdf"]), 0.01, 0.3) / 10000
            data["sample_probs"] = sample_probs / sample_probs.sum()

        return data

    def representation_to_mesh(self, representation):
        """
        Convert shape vec to mesh
        Args:
            representation {torch.Tensor} -- representation, currently 3Dshape2vec
        Returns:
            o3d.geometry.TriangleMesh -- Mesh representation of the SDF
        """
        sdf_np = representation.detach().cpu().numpy()
        # plot_cross_sections(sdf_np, path="outputs")
        return sdf_to_mesh(sdf_np, level=self.level, smaller_values_are_inside=False)

    def get_representation_from_batch(self, batch):
        if self.sampling_strategy == "near_far":
            if "sdf_grid128_path" not in batch.keys():
                warnings.warn("Preprocessed SDF not found. Generating SDF from mesh.")
                # For single test sample: only compute data once
                if self.single_sample is None:
                    sdf = self.mesh_to_representation(batch["mesh_gt"])
                else:
                    sdf = self.single_sample
            else:
                sdf = (
                    torch.from_numpy(np.load(batch["sdf_grid128_path"][0])["sdf"])
                    .unsqueeze(0)
                    .float()
                    .cuda()
                )
                # downsample if necessary
                sdf = downsample_grid_sdf(sdf, 64, sdf.shape[-1])
                # clamp
                sdf = torch.clamp(sdf, -0.2, 0.2)

            return sdf

        else:
            return batch["pcl"]


class AxisScaling(object):
    """Data augmentation - TODO: integrate"""

    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


class PointCloudDataModule(nn.Module):
    def __init__(
        self,
        config,
        dataset,
        batch_size,
        num_workers,
        val_dataset=None,
        test_dataset=None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.config = config
        self.dataset = dataset
        self.batch_size_per_device = batch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # TODO: put preprocessing script here.
        pass

    def setup(self):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # Assign train/val datasets for use in dataloaders
        if not self.data_train and self.dataset:
            self.data_train = PointCloudDataset(
                dataset=self.dataset, **dict(self.config)
            )
        if not self.data_val and self.val_dataset:
            # We use test set for validation for now.
            self.data_val = PointCloudDataset(
                dataset=self.val_dataset, **dict(self.config)
            )
        if not self.data_test and self.test_dataset:
            self.data_test = PointCloudDataset(
                dataset=self.test_dataset, **dict(self.config)
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
