from argparse import Namespace
import os

from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint


class HyperparameterLogger(BaseLogger):
    """HyperparameterLogger logs modules hyperparameters."""

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "hparams"

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)

    def log_iter(self, batch) -> None:
        """
        Logs metrics for the current iteration.
        - add each metric to the BaseLogint
        - record each metric for epoch save
        - display in terminal, displayed metric is averaged

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        pass

    def log_epoch(self) -> None:
        """
        At the end of each phase, logs a summary of the metrics to LoggingInterface.
        """
        pass

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

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: The hyperparameters.
        :param metrics: Any metric.
        """
        params = self._convert_params(params)

        # Handle OmegaConf object
        try:
            from omegaconf import OmegaConf
        except ModuleNotFoundError:
            pass
        else:
            # Convert to primitives
            if OmegaConf.is_config(params):
                params = OmegaConf.to_container(params, resolve=True)

        if metrics is None:
            metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        self.logger_interface.log_hyperparams(params, metrics)

    @staticmethod
    def _convert_params(params):
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params
