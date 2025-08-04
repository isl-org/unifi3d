import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import open3d as o3d

from unifi3d.data.base_dataset import BaseDataset
from unifi3d.data.data_iterators import BaseIterator
from unifi3d.utils.data.sdf_utils import (
    generate_query_points,
    mesh_to_sdf,
    sdf_to_mesh,
    center_scale_mesh,
)
from unifi3d.utils.data.mesh_utils import load_normalize_mesh


class PCSDFDataset(BaseDataset):
    """
    Container for the PCSDF dataset required to train the PC_SDF_AE autoencoder.
    """

    def __init__(
        self,
        dataset: BaseIterator,
        res: int = 128,
        sample_res: int = 128,
        num_near: int = 4096,
        num_far: int = 4096,
        threshold: float = 0.001,
        padding: float = 0.0,
        level: float = 0.0,
        num_samples: int = 2048,
        thresh_clamp_sdf: float = 0.2,
    ) -> None:
        """
        Initializes the dataset object with configuration and mode.

        Args:
            dataset (BaseIterator): Dataset iterator.
            res (int): Resolution of the SDF grid fed to the model.
            sample_res (int): Resolution of the original SDF grid.
            num_near (int): Number of SDF samples close to the volume.
            num_far (int): Number of SDF samples far from the volume.
            padding (float): Additional space added to each side of the unit cube to
                         ensure the grid covers the entire region of interest.
            level (floar): Isovalue in the marchine cube algorithm.
            num_samples (int): Number of points sampled on the surface of the mesh.
            threshold (float): Distance from which the point is considered as far
                         from the surface.
            thresh_clamp_sdf (float): Maximum distance value in the SDF
        """
        super(PCSDFDataset, self).__init__(dataset)

        self.grid_resolution = res
        self.sample_res = sample_res
        self.padding = padding
        self.level = level
        self.thresh_clamp_sdf = thresh_clamp_sdf
        self.mode = self.data.mode
        self.num_samples = num_samples
        self.num_near = num_near
        self.num_far = num_far
        self.threshold = threshold
        self.query_points = generate_query_points(
            grid_resolution=self.grid_resolution, padding=self.padding
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Args:
            None

        Returns:
            int: size of the dataset
        """
        return len(self.data)

    def __getitem__(self, index) -> None:
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): index of the data iterator.

        Returns:
            dict: updated data dictionary
        """
        data = self.data[index]
        data = self.mesh_to_representation(data)
        return data

    def mesh_to_representation(self, data):
        """
        Loads mesh and converts into a SDF and a sampled point cloud

        Args:
            data (dict): dictionary containing data file_path.

        Returns:
            dict: updated data dictionary
        """
        # read mesh
        mesh_scale = 1 - self.padding
        mesh = load_normalize_mesh(data["file_path"], mesh_scale)
        # transform to sdf (function uses bbox [-0.5, 0.5])
        sdf = mesh_to_sdf(
            mesh,
            sample_res=self.sample_res,
            grid_resolution=self.grid_resolution,
        )
        if self.thresh_clamp_sdf != 0.0:
            sdf = torch.clamp(
                sdf, min=-self.thresh_clamp_sdf, max=self.thresh_clamp_sdf
            )

        # sample []
        sampled_sdf, sampled_query_points = self.sample_sdf(sdf, self.query_points)

        # sample points from surfaces of mesh
        sampled_vertices = np.asarray(
            mesh.sample_points_uniformly(number_of_points=self.num_samples).points
        )

        # append relevant fields to data dictionary
        data["sdf"] = sdf.cuda()  # N
        data["sampled_sdf"] = torch.from_numpy(sampled_sdf).float().cuda()  # N
        data["query_points"] = (
            torch.from_numpy(sampled_query_points).float().cuda()
        )  # N,3
        data["pcl"] = torch.from_numpy(sampled_vertices).float().cuda()  # M,3

        return data

    def sample_sdf(self, sdf, query_points) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a predefined number of "near" and "far" points.

        Args:
            sdf (np.ndarray): sdf grid of shape (B,1,N,N,N).
            query_points (np.ndarray): query points array of shape (B,N**3,3).

        Returns:
            np.ndarray: array of sampled sdf of shape (B,M).
            np.ndarray: array of sampled query points of shape (B,M,3).
        """
        # Identify near and far points
        sdf = sdf.reshape(-1)  # flatten grid to (B,N**3)
        mask_far = abs(sdf) >= self.threshold
        mask_near = ~mask_far
        sdf_far_query = query_points[mask_far]
        sdf_near_query = query_points[mask_near]
        sdf_far_dist = sdf[mask_far]
        sdf_near_dist = sdf[mask_near]

        # Sample a predefined number of "near" and "far" points.
        far_query, far_dist = self.sample(
            sdf_far_query, sdf_far_dist, self.num_far, train=self.mode == "train"
        )
        near_query, near_dist = self.sample(
            sdf_near_query, sdf_near_dist, self.num_near, train=self.mode == "train"
        )

        sampled_sdf = np.concatenate([far_dist, near_dist])
        sampled_query_points = np.concatenate([far_query, near_query])

        return sampled_sdf, sampled_query_points

    def sample(self, pcl, dist, N, train=True) -> tuple:
        """
        Samples points from the given point cloud.

        Args:
            pcl (np.ndarray): The point cloud from which to sample.
            pcl (np.ndarray): The corresponding distance values.
            N (int): The number of points to sample.
            train (bool): Indicates if the mode is training.

        Returns:
            np.ndarray: The sampled points.
        """
        if train:
            choice = np.random.choice(len(pcl), N, replace=True)
            return pcl[choice], dist[choice]
        else:
            return pcl[:N], dist[:N]

    def representation_to_mesh(self, sdf_decoded):
        """
        Convert SDF to mesh
        Args:
            sdf_decoded {torch.Tensor} -- SDF tensor of shape (1, res, res, res)
        Returns:
            o3d.geometry.TriangleMesh -- Mesh representation of the SDF
        """
        assert sdf_decoded.dim() == 4, "The provided SDF has the wrong shape!"
        nc = sdf_decoded.shape[0]
        assert nc == 1, "The provided SDF has more than one channel!"

        sdf_np = sdf_decoded[0].detach().cpu().numpy()
        return sdf_to_mesh(sdf_np, level=self.level)

    def get_representation_from_batch(self, batch) -> tuple:
        return batch["query_points"], batch["sdf"], batch["pcl"]


class PCSDFDataModule(nn.Module):
    def __init__(self, config, dataset, batch_size, num_workers) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.config = config
        self.dataset = dataset
        self.batch_size_per_device = batch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # TODO: put preprocessing script here.
        pass

    def setup(self) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        # Assign train/val datasets for use in dataloaders
        if not self.data_train and not self.data_val and not self.data_test:
            self.dataset.mode = "train"
            self.data_train = PCSDFDataset(dataset=self.dataset, **dict(self.config))

            self.dataset.mode = "val"
            # We use test set for validation for now.
            self.data_val = PCSDFDataset(dataset=self.dataset, **dict(self.config))

            self.dataset.mode = "test"
            self.data_test = PCSDFDataset(dataset=self.dataset, **dict(self.config))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=collate_func,
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=collate_func,
            # pin_memory=True,
        )
