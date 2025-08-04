import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from unifi3d.data.sdf_dataset import SDFDataset
from unifi3d.utils.data.sdf_utils import (
    sdf_to_mesh,
)


class VoxelDataset(SDFDataset):
    def __init__(self, dataset, **config):
        super().__init__(dataset, **config)

    def __getitem__(self, idx):
        # get sampled SDF
        sample = super().__getitem__(idx)
        # convert SDF to occupancy (NOTE: still named sdf for compatibility with VQ-VAE)
        sample["sdf"] = (sample["sdf"] < 0).float()
        return sample

    def __len__(self):
        return super().__len__()

    def representation_to_mesh(self, voxel):
        assert voxel.dim() == 4, "The provided voxel grid has the wrong shape!"
        nc = voxel.shape[0]
        assert nc == 1, "The provided SDF has more than one channel!"

        sdf_np = voxel[0].detach().cpu().numpy()
        return sdf_to_mesh(
            sdf_np,
            padding=self.padding,
            level=self.level,
            smaller_values_are_inside=False,
        )

    def get_representation_from_batch(self, batch):
        return batch["sdf"]


class VoxelDataModule(nn.Module):
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

    def setup(self):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # Assign train/val datasets for use in dataloaders
        if not self.data_train and self.dataset:
            self.data_train = VoxelDataset(dataset=self.dataset, **dict(self.config))
        if not self.data_val and self.val_dataset:
            # We use test set for validation for now.
            self.data_val = VoxelDataset(dataset=self.val_dataset, **dict(self.config))
        if not self.data_test and self.test_dataset:
            self.data_test = VoxelDataset(
                dataset=self.test_dataset, **dict(self.config)
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # pin_memory=True,
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
