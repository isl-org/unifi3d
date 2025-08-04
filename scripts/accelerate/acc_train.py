from accelerate import Accelerator, ProfileKwargs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import rootutils
from typing import Any, Dict, List, Optional, Tuple
import torch
from unifi3d.utils.logger import Logger
import torch.multiprocessing as mp
import sys


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
torch.autograd.set_detect_anomaly(True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from unifi3d.utils import (
    extras,
    get_metric_value,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from unifi3d.utils.model.model_utils import overwrite_cfg_from_ckpt


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains the model. Can additionally evaluate on a testset, using last weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("overwrite_model_cfg", False):
        cfg = overwrite_cfg_from_ckpt(cfg)

    logging.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    ################################################################################
    # Setup profiler
    profile_kwargs = []
    profiling = cfg.profiling
    # Profiling uses the same default output_dir as logger.
    profiling_dir = os.path.join(cfg.paths.output_dir, "profiling")

    if profiling and profiling.get("activated", False):
        profile_kwargs = [
            ProfileKwargs(
                activities=profiling.get("activities", ["cpu", "cuda"]),
                record_shapes=profiling.get("record_shapes", True),
                with_flops=profiling.get("with_flops", True),
                profile_memory=profiling.get("profile_memory", True),
                with_modules=profiling.get("with_modules", True),
                output_trace_dir=profiling_dir,
            )
        ]

    # Needs to be executed for the first time inside the training function spawned on each GPU device.
    process_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=5400))]
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=profile_kwargs + process_kwargs,
    )
    if accelerator.is_main_process:
        logging.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    else:
        logger = []
    logging.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, accelerator=accelerator, logger=logger
    )

    # save config to model-run dir
    OmegaConf.save(cfg, os.path.join(trainer.default_root_dir, "config.yaml"))
    logging.info(f"Exporting model cfg to {trainer.default_root_dir}")

    logging.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model, device=trainer.accelerator.device)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        logging.info("Logging hyperparameters!")
        log_hyperparameters(
            object_dict, is_main_process=trainer.accelerator.is_main_process
        )

    if cfg.get("train"):
        logging.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            resume_training=cfg.get("resume_training", False),
        )

    train_metrics = trainer.updated_metrics

    if cfg.get("test"):
        logging.info("Starting testing!")
        ckpt_path = trainer.last_model_path
        if ckpt_path == "":
            logging.warning("Last ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logging.info(f"Last ckpt path: {ckpt_path}")

    test_metrics = trainer.updated_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
