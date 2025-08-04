from collections import OrderedDict
import os
import json
import yaml
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from unifi3d.data.base_dataset import BaseDataset
from unifi3d.utils.data.dataset_utils import custom_collate
from unifi3d.utils.data.shap_e_utils import load_or_create_multimodal_batch
from unifi3d.utils.model.shap_e_utils import AttrDict


class ShapEDataset(BaseDataset):
    """
    Container for the dataset used with Shap-E.
    """

    def __init__(self, dataset, config):
        """
        Initializes the dataset object with configuration and mode.
        """
        super(ShapEDataset, self).__init__(dataset)

        # load configs
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.padding = config.get("padding", 0.0)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.
        """
        data = self.data[index]
        if self.config.name == "shapenet":
            fpath = Path(data["file_path"]).parts
            cache_dir = os.path.join(
                self.config.cache_path, fpath[-3] + "/" + fpath[-2]
            )
        else:
            cache_dir = data["file_path"][:-4]
        pcl_image = load_or_create_multimodal_batch(
            self.device,
            model_path=data["file_path"],
            mv_light_mode=self.config.get("mv_light_mode", "basic"),
            mv_image_size=self.config.get("mv_image_size", 256),
            cache_dir=cache_dir,
            verbose=False,
        )
        data["points"] = pcl_image["points"][0]
        data["depths"] = pcl_image["depths"][0]
        data["views"] = pcl_image["views"][0]
        data["view_alphas"] = pcl_image["view_alphas"][0]
        data["cameras"] = pcl_image["cameras"][0]

        return data

    def mesh_to_representation(self, data):
        """
        Currently not implemented, as the Shap-E autoencoder does not satisfy the traditional autoencoder structure.
        The input is a colored point cloud with multi-view renders, and the output is the NeRF parameters.
        If needed, we can add the algorithm from Mesh2NeRF.
        """

        raise NotImplementedError


class ShapEDataModule(nn.Module):
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
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.
        Do not use it to assign state (self.x = y).
        """
        # TODO: put preprocessing script here.
        pass

    def setup(self):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices. This is for our reference check only, lightning should take care
        # of the batch division automatically.

        # Assign train/val datasets for use in dataloaders
        if not self.data_train and self.dataset:
            self.data_train = ShapEDataset(dataset=self.dataset, config=self.config)
        if not self.data_val and self.val_dataset:
            # We use test set for validation for now.
            self.data_val = ShapEDataset(dataset=self.val_dataset, config=self.config)
        if not self.data_test and self.test_dataset:
            self.data_test = ShapEDataset(dataset=self.test_dataset, config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=custom_collate,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=custom_collate,
        )
