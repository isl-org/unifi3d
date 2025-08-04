import numpy as np
import torch
import warnings
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from unifi3d.data.base_dataset import BaseDataset
from unifi3d.data.data_iterators import BaseIterator
from unifi3d.utils.data.sdf_utils import (
    generate_query_points,
    mesh_to_sdf,
    sdf_to_mesh,
    downsample_grid_sdf,
)
from unifi3d.utils.data.mesh_utils import load_normalize_mesh


class SDFDataset(BaseDataset):
    """
    Container for the SDF dataset required to train the VQVAE encoder.
    """

    def __init__(
        self,
        dataset: BaseIterator,
        res: int = 64,
        sample_res: int = 64,
        padding: float = 0.0,
        level: float = 0.0,
        thresh_clamp_sdf: float = 0.2,
    ) -> None:
        """
        Initializes the dataset object with configuration and mode.
        """
        super(SDFDataset, self).__init__(dataset)

        self.grid_resolution = res
        self.sample_res = sample_res
        self.padding = padding
        self.level = level
        self.thresh_clamp_sdf = thresh_clamp_sdf
        if res <= 128:
            self.sdf_path_key = "sdf_grid128_path"
        else:
            self.sdf_path_key = "sdf_grid256_path"

        # for single test sample: only compute data once to save training time
        self.single_sample = None  # by default data is loaded at runtime
        if self.data.path == "knot":
            self.single_sample = self.mesh_to_representation(self.data[0])

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index) -> None:
        """
        Retrieves an item from the dataset at the specified index.
        """
        data = self.data[index]
        if self.sdf_path_key not in data.keys():
            warnings.warn("Preprocessed SDF not found. Generating SDF from mesh.")
            # For single test sample: only compute data once
            if self.single_sample is None:
                sdf = self.mesh_to_representation(data)
            else:
                sdf = self.single_sample
        else:
            sdf = (
                torch.from_numpy(np.load(data[self.sdf_path_key])["sdf"])
                .unsqueeze(0)
                .float()
                .cuda()
            )
            # downsample if necessary
            sdf = downsample_grid_sdf(sdf, self.grid_resolution, sdf.shape[-1])
            # clamp
            sdf = torch.clamp(sdf, -self.thresh_clamp_sdf, self.thresh_clamp_sdf)

        data["sdf"] = sdf
        return data

    def mesh_to_representation(self, data):
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
        return sdf.cuda()

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
        return sdf_to_mesh(sdf_np, level=self.level, smaller_values_are_inside=True)

    def get_representation_from_batch(self, batch):
        return batch["sdf"]


class SDFDataModule(nn.Module):
    def __init__(
        self,
        config,
        dataset,
        batch_size,
        num_workers,
        val_dataset=None,
        test_dataset=None,
    ) -> None:
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
        if not self.data_train and self.dataset:
            self.data_train = SDFDataset(dataset=self.dataset, **dict(self.config))
        if not self.data_val and self.val_dataset:
            # We use test set for validation for now.
            self.data_val = SDFDataset(dataset=self.val_dataset, **dict(self.config))
        if not self.data_test and self.test_dataset:
            self.data_test = SDFDataset(dataset=self.test_dataset, **dict(self.config))

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
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
