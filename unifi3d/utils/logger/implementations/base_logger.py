from accelerate.logging import get_logger
from .interfaces.base_logint import BaseLogint


class BaseLogger(object):
    """
    BaseLogger serves as a base class for any specific logging mechanisms.
    It defines the interface for logging phases, batches, and ending logs.
    Derived classes must implement the abstract methods.
    """

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        """
        Initialize the BaseLogger with configurations, logging paths, and a TensorBoard logger instance.

        :param logger_interface: An instance of a BaseLogint.
        :param log_path: String representing the path where logs should be stored.
        :param config: Dictionary containing configurations for training and evaluation.
        """
        super().__init__()

        # An instance of the logging interface
        self.logger_interface = logger_interface

        # Ranked logger
        self.log = get_logger(__name__, log_level="INFO")

        # Path to store log files
        self.log_path = log_path

        # Configuration dictionary containing training and evaluation settings
        self.config = config

        # Base data
        self.NAME = "base"
        self.phase = None
        self.epoch = -1

        # For printing purposes
        self.batch_print_dict = {}
        self.phase_print_dict = {}

    def log_epoch(self) -> None:
        """
        Placeholder method to be implemented by subclasses for logging different phases like train, validate, or test.
        """
        pass

    def log_iter(self, batch) -> None:
        """
        Placeholder method to be implemented by subclasses for logging information specific to a batch.

        :param batch: The batch data that needs to be logged.
        """
        pass

    def end_log(self) -> None:
        """
        Placeholder method.
        """
        self.logger_interface.finalize()

    def print_iter_log(self) -> None:
        """
        Placeholder method to be implemented by subclasses for printing out relevant logged quantity for the current iteration.
        """
        pass

    def print_epoch_log(self) -> None:
        """
        Placeholder method to be implemented by subclasses for printing out relevant logged quantity for the current epoch.
        """
        pass
