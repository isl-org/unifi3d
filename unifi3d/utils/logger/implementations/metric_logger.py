import os
from pprint import pformat
import torch

from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint


class MetricLogger(BaseLogger):
    """
    MetricLogger logs metrics for batches and phases, displays them, and saves summaries.
    """

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "metric"

        # Metric container to store metrics for averaging
        self.metric_container = dict()

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(os.path.join(self.log_path, "batchwise"), exist_ok=True)

    def append_metric(self, key, value, epoch):
        self.metric_container.setdefault(key, []).append(value)
        self.logger_interface.log_scalar(
            tag=f"Metric-BatchWise/{key}/{self.phase}",
            value=float(value),
            epoch=int(epoch),
            context={"subset": self.phase},
        )
        self.batch_print_dict[key] = float(value)

    def log_iter(self, batch) -> None:
        """
        Logs metrics for the current iteration.
        - add each metric to the BaseLogint
        - record each metric for epoch save
        - display in terminal, displayed metric is averaged

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        if self.NAME not in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:
            return

        # Update logger state
        self.phase = batch["phase"]
        self.epoch = batch["epoch"]
        self.batch_print_dict = {}

        for key in keys_list:
            if key not in batch:
                # Reverting runtime exception here.
                # Certain keys like val_loss will not exist in training batch, we
                # allow key that does not exist in the batch here, because we do not
                # differentiate train logging and val logging at the moment.
                print(f"For metric logging, key {key} not found in batch.")
                continue
            epoch = batch["total_iters"] + 1
            if isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    self.append_metric(sub_key, batch[key][sub_key], epoch)
            else:
                self.append_metric(key, batch[key], epoch)

    def log_epoch(self) -> None:
        """
        At the end of each phase, logs a summary of the metrics to LoggingInterface.
        """
        self.phase_print_dict = {}
        for key, values in self.metric_container.items():
            mean_value = sum(values) / len(values)
            self.logger_interface.log_scalar(
                tag=f"Metric-EpochWise/{key}/{self.phase}",
                value=mean_value,
                epoch=int(self.epoch),
                context={"subset": self.phase},
            )
            self.logger_interface.log_histogram(
                tag=f"Metric-Hist/{key}/{self.phase}",
                hist=torch.Tensor(values),
                epoch=int(self.epoch),
                context={"subset": self.phase},
            )
            self.phase_print_dict[key] = float(mean_value)

        self.metric_container.clear()

    def print_iter_log(self) -> None:
        """
        Prints out relevant logged quantity for the current iteration.
        """
        print(
            f"\nBatch metric:\n{pformat(self.batch_print_dict, indent=2, compact=True)}"
        )

    def print_epoch_log(self) -> None:
        """
        Prints out relevant logged quantity for the current epoch.
        """
        print(
            f"\nPhase metric:\n{pformat(self.phase_print_dict, indent=2, compact=True)}"
        )
