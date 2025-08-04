import numpy as np
import os
import torch

from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint


class HistogramLogger(BaseLogger):
    """HistogramLogger logs histograms for batches and phases."""

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "hist"

        # Container for capturing histogram data
        self.metric_container = dict()

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)

    def log_iter(self, batch) -> None:
        """
        Log histogram data for the current iteration.

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        if self.NAME not in batch["output_parser"].keys():
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:  # No keys to process
            return

        # Update logger state
        self.phase = batch["phase"]
        self.epoch = batch["epoch"]

        for k in keys_list:
            if k not in batch:
                continue

            self._aggregate_metric(k, batch[k])

        self.log_aggregated()

    def log_aggregated(self) -> None:
        # Log the aggregated histogram at the end of a phase.
        for key, values in self.metric_container.items():
            self.logger_interface.log_histogram(
                tag=f"Hist/{key}/{self.phase}",
                hist=torch.tensor(values),
                epoch=self.epoch,
                context={"subset": self.phase},
            )

    def _aggregate_metric(self, key, data) -> None:
        """
        Helper method to aggregate metrics.

        :param key: Metric key to log.
        :param data: Data associated with the key.
        """
        # Ensure there's a container for the metric
        if key not in self.metric_container:
            self.metric_container[key] = []

        if isinstance(data, (torch.Tensor, np.ndarray)):
            # Squeeze tensor to ensure it's a 1D list (for histogram)
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().squeeze()
                assert data.dim() <= 1, "Supported tensor shapes are [B] or scalar"

            # Convert to list and extend the metric container
            data_list = data.tolist()
            if not isinstance(data_list, list):
                data_list = [data_list]
            self.metric_container[key].extend(data_list)
        elif isinstance(data, (int, float)):
            # Directly append scalar values
            self.metric_container[key].append(float(data))
        else:
            raise TypeError(
                f"Unsupported type for histogram data: {type(data).__name__}"
            )

    def log_epoch(self) -> None:
        self.log_aggregated()
        # Reset metric container for new phase
        self.metric_container.clear()

    def print_iter_log(self) -> None:
        """
        Prints out relevant logged quantity for the current iteration.
        """
        pass

    def print_epoch_log(self) -> None:
        """
        Prints out relevant logged quantity for the current epoch.
        """
        pass
