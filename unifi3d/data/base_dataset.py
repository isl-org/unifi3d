import torch

from unifi3d.data.data_iterators import BaseIterator


class BaseDataset(torch.utils.data.Dataset):
    """
    Container for datasets.

    Attributes:
        data (BaseIterator): Container for specific dataset.
        mode (str): Mode.
    """

    def __init__(self, dataset_iterator: BaseIterator, device="cuda") -> None:
        """
        Initializes the dataset object with a dataset iterator.

        Args:
            dataset_iterator (BaseIterator): Container for specific dataset, e.g.
                Objaverse.
        """
        super(BaseDataset, self).__init__()
        self.data = dataset_iterator
        self.mode = self.data.mode
        self.device = device

        return

    def __len__(self):
        return len(self.data)

    def get_sampler(self, shuffle=True, distributed=False):
        """
        Retrieve a sampler for balancing the dataset in training mode.

        Returns:
            RandomSampler: Returns the sampler object if conditions are met, or `None`
            if balancing is not required or if the mode is not set to "train".
        """
        if not self.balance_flag or self.mode != "train":
            return None

        if distributed:
            return torch.utils.data.distributed.DistributedSampler(shuffle=shuffle)

        if shuffle:
            return torch.utils.data.RandomSampler()

        else:
            return torch.utils.data.SequentialSampler()

    def load_data(self, dir: str, file_prefix: str, idx: int) -> dict:
        """
        Loads data from a slice of the dataset.

        Args:
            dir (str): Directory where the data is stored.
            file_prefix (str): Prefix for the data files.
            idx (int): Index of the mesh.
        Returns:
            dict: Data loaded from the slice.
        """

        pass
