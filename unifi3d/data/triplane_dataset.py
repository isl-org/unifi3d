import os
import json
import open3d as o3d
import numpy as np
from typing import Optional
import torch
import warnings
import time
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from unifi3d.utils.triplane_utils.common import decide_total_volume_range, update_reso
import unifi3d.utils.triplane_utils.data as data
from torchvision import transforms

import yaml


class TriplaneDataModule(nn.Module):
    def __init__(
        self,
        config,
        batch_size,
        num_workers,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # TODO: put preprocessing script here.
        pass

    def get_data_fields(self, mode):
        """Returns the data fields.

        Args:
            mode (str): the mode which is used
        """
        points_transform = (
            data.SubsamplePoints(self.config["data"]["points_subsample"])
            if mode == "train"
            else None
        )

        input_type = self.config["data"]["input_type"]
        fields = {}
        if self.config["data"]["points_file"] is not None:
            if input_type != "pointcloud_crop":
                fields["points"] = data.PointsField(
                    self.config["data"]["points_file"],
                    points_transform,
                    unpackbits=self.config["data"]["points_unpackbits"],
                    multi_files=self.config["data"]["multi_files"],
                )
            else:
                fields["points"] = data.PatchPointsField(
                    self.config["data"]["points_file"],
                    transform=points_transform,
                    unpackbits=self.config["data"]["points_unpackbits"],
                    multi_files=self.config["data"]["multi_files"],
                )

        if mode in ("val", "test"):
            points_iou_file = self.config["data"]["points_iou_file"]
            voxels_file = self.config["data"]["voxels_file"]
            if points_iou_file is not None:
                if input_type == "pointcloud_crop":
                    fields["points_iou"] = data.PatchPointsField(
                        points_iou_file,
                        unpackbits=self.config["data"]["points_unpackbits"],
                        multi_files=self.config["data"]["multi_files"],
                    )
                else:
                    fields["points_iou"] = data.PointsField(
                        points_iou_file,
                        unpackbits=self.config["data"]["points_unpackbits"],
                        multi_files=self.config["data"]["multi_files"],
                    )
            if voxels_file is not None:
                fields["voxels"] = data.VoxelsField(voxels_file)

        return fields

    def get_inputs_field(self, mode):
        """Returns the inputs fields.

        Args:
            mode (str): the mode which is used
            cfg (dict): config dictionary
        """
        input_type = self.config["data"]["input_type"]

        if input_type is None:
            inputs_field = None
        elif input_type == "pointcloud":
            transform = (
                transforms.Compose(
                    [
                        data.SubsamplePointcloud(self.config["data"]["pointcloud_n"]),
                        data.PointcloudNoise(self.config["data"]["pointcloud_noise"]),
                    ]
                )
                if mode == "train"
                else None
            )
            inputs_field = data.PointCloudField(
                self.config["data"]["pointcloud_file"],
                transform,
                multi_files=self.config["data"]["multi_files"],
            )
        elif input_type == "partial_pointcloud":
            transform = (
                transforms.Compose(
                    [
                        data.SubsamplePointcloud(self.config["data"]["pointcloud_n"]),
                        data.PointcloudNoise(self.config["data"]["pointcloud_noise"]),
                    ]
                )
                if mode == "train"
                else None
            )
            inputs_field = data.PartialPointCloudField(
                self.config["data"]["pointcloud_file"],
                transform,
                multi_files=self.config["data"]["multi_files"],
            )
        elif input_type == "pointcloud_crop":
            transform = (
                transforms.Compose(
                    [
                        data.SubsamplePointcloud(self.config["data"]["pointcloud_n"]),
                        data.PointcloudNoise(self.config["data"]["pointcloud_noise"]),
                    ]
                )
                if mode == "train"
                else None
            )

            inputs_field = data.PatchPointCloudField(
                self.config["data"]["pointcloud_file"],
                transform,
                multi_files=self.config["data"]["multi_files"],
            )

        elif input_type == "voxels":
            inputs_field = data.VoxelsField(self.config["data"]["voxels_file"])
        elif input_type == "idx":
            inputs_field = data.IndexField()
        else:
            raise ValueError("Invalid input type (%s)" % input_type)
        return inputs_field

    # Datasets
    def get_dataset(self, mode):
        """Returns the dataset.

        Args:
            model (nn.Module): the model which is used
            return_idx (bool): whether to include an ID field
        """
        dataset_folder = self.config["data"]["path"]
        categories = self.config["data"]["classes"]

        # Get split
        splits = {
            "train": self.config["data"]["train_split"],
            "val": self.config["data"]["val_split"],
            "test": self.config["data"]["test_split"],
        }

        split = splits[mode]

        fields = self.get_data_fields(mode)
        inputs_field = self.get_inputs_field(mode)
        if inputs_field is not None:
            fields["inputs"] = inputs_field

        dataset = data.Shapes3dDataset(
            dataset_folder, fields, split=split, categories=categories, cfg=self.config
        )

        return dataset

    def setup(self):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        # Assign train/val datasets for use in dataloaders
        self.data_train = self.get_dataset("train")
        self.data_val = self.get_dataset("val")
        self.data_test = self.get_dataset("test")

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
