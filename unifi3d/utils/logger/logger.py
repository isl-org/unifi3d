from accelerate.logging import get_logger
import os
from copy import deepcopy
from .implementations.interfaces import REGISTED_INTERFACE
from .implementations import (
    REGISTED_LOGGER,
    HyperparameterLogger,
)
import shutil
import time
import torch


class Logger:
    """
    Logger class that oversees multiple logging mechanisms such as TensorBoard and Aim.
    Manages the initialization, composition, and lifecycle of different logger instances.
    """

    def __init__(self, config) -> None:
        """
        Initialize the Logger with a deep copy of the configuration and compose logger instances.

        :param config: A dictionary containing the configuration for the logger.
        """
        # Make a deep copy of the configuration to prevent accidental mutations
        self.config = deepcopy(config)

        # Extra setting parameters
        self.phase = None
        self.epoch = -1
        self.max_epochs = -1
        self.batch_size = -1
        self.eval_batch_size = -1

        # Experiment name
        self.name = self.config.get("name", "default")

        # Construct the base path for log files
        self.root_dir = self.config.get("root_dir", "root")
        self.log_dir = os.path.join(self.root_dir, "log")
        self.ckpt_dir = os.path.join(self.root_dir, "checkpoints")

        # Logging interface configuration
        logint = self.config.get("logint", "aim")
        logint_config = {}
        if logint == "aim":
            logint_config["experiment"] = self.name
            logint_config["repo"] = self.config.get("repo", "")
        elif logint == "comet":
            logint_config["api_key"] = self.config.get("api_key", "")
            logint_config["project_name"] = self.name
            logint_config["workspace"] = self.config.get("workspace", "default")
        elif logint == "mlflow":
            logint_config["experiment"] = self.name
        elif logint == "neptune":
            logint_config["api_key"] = self.config.get("api_key", "")
            logint_config["project_name"] = self.name
            logint_config["run_name"] = self.config.get("run_name", [])
            logint_config["tags"] = self.config.get("tags", [])
        elif logint == "tensorboard":
            logint_config["log_dir"] = self.log_dir
        elif logint == "wandb":
            logint_config["name"] = self.name
        else:
            raise NotImplementedError

        # Logging interface instance creation
        self.logger = REGISTED_INTERFACE[logint](logint_config)

        # Compose the list of loggers as per configuration
        logger_implementations = self.config.get("loggers", ["metric"])
        self.logger_list = self.compose(logger_implementations)

        # Specifications of the logged quantities
        self.tracked_variables = self.config.get("tracked_variables", {})

        # Sanity check
        for keys in self.tracked_variables.keys():
            assert (
                keys in logger_implementations
            ), f"Key {keys} not found in the logger implementations"

        # A Log message is being printed every print_every_n_seconds seconds
        self.print_every_n_seconds = config.get("print_every_n_seconds", 2.0)

        # Store starting time of the phase
        self.phase_time_start = time.time()

        # Stores the last time logged data were printed
        self.print_time = time.time()

        # If True, then print the used vs. total GPU memory
        self.gpu_summarize_flag = config.get("gpu_summarize", True)

        # Ranked logger
        self.log = get_logger(__name__, log_level="INFO")

    @property
    def group_separator(self) -> str:
        """Return the default separator used by the logger to group the data into subfolders."""
        return "/"

    def compose(self, names) -> list:
        """
        Compose the list of logger instances based on the provided logger names.

        :param names: A list of logger names to be registered.
        :return: List of instantiated logger objects.
        """
        # Loop through each logger name and instantiate the corresponding logger
        loggers_list = list()
        for name in names:
            log_path = os.path.join(self.log_dir, name)
            if name in REGISTED_LOGGER.keys():
                logger_instance = REGISTED_LOGGER[name](
                    self.logger, log_path, self.config
                )
                loggers_list.append(logger_instance)
            else:
                raise Warning("Required logger " + name + " not found!")

        return loggers_list

    def log_epoch(self) -> None:
        """
        Invoke the log_epoch method on all registered loggers to log epoch-specific information.
        """
        for logger in self.logger_list:
            logger.log_epoch()

        self._print_epoch_log()

    def log_iter(self, batch) -> None:
        """
        Log information about a batch using all registered loggers.

        :param batch: The batch data that needs to be logged.
        """
        self.epoch = batch["epoch"]
        self.phase = batch["phase"]
        self.batch_size = batch["batch_size"]
        self.max_epochs = batch["max_epochs"]
        for logger in self.logger_list:
            logger.log_iter(batch)

        self._print_iter_log(batch)

    def log_hyperparams(self, hparams) -> None:
        """
        Record hyperparameters.

        :param hparams: The hyperparameters.
        """
        self.hyperparameter_logger.log_hyperparams(params=hparams)

    def end_log(self) -> None:
        """
        Invoke the end_log method on all registered loggers to finalize logging.
        """
        # Finalize the logger
        for logger in self.logger_list:
            logger.end_log()

    def _print_iter_log(self, batch) -> None:
        """
        Print a set of relevant iteration data every self.print_every_n_seconds seconds.
        """

        if time.time() - self.print_time > self.print_every_n_seconds:
            # Number of batches processed in the current epoch (reset to 1 at every epoch).
            current_batch_number = batch["batch_current"]

            # Total -- maximum -- number of batches to be processed in the current epoch.
            total_batch_number = batch["batch_total"]

            # Time spent since the last epoch
            time_spent = (time.time() - self.phase_time_start) / 60.0

            # Time the whole epoch is expected to take
            assert current_batch_number > 0, "The batch index should start at 1..."
            estimated_time_per_epoch = (
                total_batch_number * time_spent / current_batch_number
            )

            # Print high level log info on the console
            print(
                f'\nPhase "{self.phase}" | Epoch {self.epoch}/{self.max_epochs} |'
                f" Steps {current_batch_number}/{total_batch_number } |"
                f" Samples {current_batch_number * self.batch_size}/{total_batch_number * self.batch_size} |"
                f" Time {time_spent:.3f}min/{estimated_time_per_epoch:.3f}min"
            )

            # Print relevant batch log for every logger
            for logger in self.logger_list:
                logger.print_iter_log()

            print("." * 80)

            # Reset batch timer
            self.print_time = time.time()

    def _print_epoch_log(self) -> None:
        """
        Print a set of relevant data at the end of every epoch.
        """

        # Print relevant phase log for every logger
        for logger in self.logger_list:
            logger.print_epoch_log()

        # Print epoch log
        print(
            f'\nFinished Epoch {self.epoch} of phase "{self.phase}" in {(time.time() - self.phase_time_start) / 60.0:.2f}min'
        )

        # Print GPU data if needed
        if self.gpu_summarize_flag:
            if torch.cuda.is_available():
                _, total = torch.cuda.mem_get_info()
                reserved = torch.cuda.memory_reserved()
                mem_unit_conv = 2**30  # Convert to GB
                print(
                    f"\n---> GPU memory: {reserved/mem_unit_conv:.3f}GB/{total/mem_unit_conv:.3f}GB <---"
                )

        # Print epoch separator
        print("\n" + "=" * shutil.get_terminal_size()[0])

        # Reset epoch timer
        self.phase_time_start = time.time()

    @property
    def hyperparameter_logger(self) -> HyperparameterLogger:
        """
        Retrieve the HyperparameterLogger from the list of registered loggers.

        :return: An instance of the HyperparameterLogger.
        :raise RuntimeError: If a HyperparameterLogger is not found.
        """
        for logger in self.logger_list:
            if isinstance(logger, HyperparameterLogger):
                return logger

        raise RuntimeError("HyperparameterLogger logger not found")
