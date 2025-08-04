import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class BaseDataModule(nn.Module):
    def __init__(
        self,
        num_workers,
        data_train=None,
        data_val=None,
        data_test=None,
        train_batch_size=16,
        val_batch_size=1,
        test_batch_size=1,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.data_train: Optional[Dataset] = data_train
        self.data_val: Optional[Dataset] = data_val
        self.data_test: Optional[Dataset] = data_test

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

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
        # We are now passing data_train, data_val and data_test directly as part of input
        # as they may not share as much commonalities based on the synthetic_room dataset example.
        pass

    # TODO: Added dummy function to accomadate benchmarking.py function.
    def setup_inference(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.test_collate_fn,
            pin_memory=True,
        )
