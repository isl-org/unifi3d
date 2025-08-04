from accelerate.logging import get_logger
from typing import Any, Dict
from omegaconf import OmegaConf
import os

log = get_logger(__name__, log_level="INFO")


def unroll_dict(d, prefix=""):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(unroll_dict(v, prefix + k + "/"))
        else:
            flat_dict[prefix + k] = v
    return flat_dict


def log_hyperparameters(
    object_dict: Dict[str, Any],
    is_main_process: bool = False,
) -> None:
    """
    Controls which config parts are saved by loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The model.
        - `"trainer"`: The trainer.
    """
    if is_main_process:
        hparams = {}

        cfg = OmegaConf.to_container(object_dict["cfg"])
        model = object_dict["model"]
        trainer = object_dict["trainer"]

        if not trainer.logger:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            return

        hparams["model"] = cfg["model"]

        # save number of model parameters
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
        hparams["data"] = cfg["data"]
        hparams["trainer"] = cfg["trainer"]
        hparams["extras"] = cfg.get("extras")
        hparams["task_name"] = cfg.get("task_name")
        hparams["tags"] = cfg.get("tags")
        hparams["ckpt_path"] = cfg.get("ckpt_path")
        hparams["seed"] = cfg.get("seed")

        # Get the SLURM job ID from the environment variable
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if slurm_job_id:
            hparams["slurm_job_id"] = slurm_job_id

        hparams["cfg"] = unroll_dict(object_dict["cfg"])

        # Send hparams to all loggers
        trainer.logger.log_hyperparams(hparams)
