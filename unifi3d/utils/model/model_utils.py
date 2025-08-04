import os
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np
import os
import glob
import warnings
import shutil
from safetensors.torch import load_file


def overwrite_cfg_from_ckpt(cfg):
    """Overwrite model cfg with the one from the checkpoint
    - for encoderDecoder, we want to replace model.net config
    - for diffusion model, we want to replace model.diff_model and model.encoder_decoder
    """
    # check whether ckpt_path exists
    if (
        "ckpt_path" in cfg
        and cfg.ckpt_path is not None
        and os.path.exists(cfg.ckpt_path)
    ) or ("ckpt" in cfg and cfg.ckpt is not None and os.path.exists(cfg.ckpt)):
        ckpt_path = cfg.ckpt_path if "ckpt_path" in cfg else cfg.ckpt
        if "safetensors" in ckpt_path:
            ckpt_path = os.path.dirname(ckpt_path)
        # we usually save the config.yaml in the parent directory of the ckpt
        ckpt_config_dir = os.path.join(ckpt_path, "..", "..", "config.yaml")
        if not os.path.exists(ckpt_config_dir):
            logging.info(
                "config.yaml not found in parent directory of ckpt, checking ckpt dir"
            )
            ckpt_config_dir = os.path.join(ckpt_path, "config.yaml")
        if os.path.exists(ckpt_config_dir):
            # Workaround to deal with symbolic links -> .. works with os but not with
            # omegaconf, so that's why we make a temporary file
            ts = round(np.random.rand() * 1000000)
            temp_file = f"temp_config_{ts}.yaml"
            shutil.copy(ckpt_config_dir, temp_file)
            ckpt_cfg = OmegaConf.load(temp_file)
            os.remove(temp_file)

            # for encoder decoder model we overwrite net
            # otherwise we overwrite net_encode and net_denoise
            if "net" in cfg.model:
                cfg.net_encode = ckpt_cfg.net_encode
                cfg.model.net = ckpt_cfg.model.net
            elif "diff_model" in cfg.model:
                ckpt_1 = cfg.model.encoder_decoder.ckpt_path
                ckpt_2 = ckpt_cfg.model.encoder_decoder.ckpt_path
                if ckpt_1 != ckpt_2:
                    warnings.warn(
                        f"Using other encoder ckpt than intended, old {ckpt_1} vs new {ckpt_2}"
                    )
                cfg.model.diff_model = ckpt_cfg.model.diff_model
                cfg.model.encoder_decoder = ckpt_cfg.model.encoder_decoder
                cfg.diffusion_sampler = ckpt_cfg.diffusion_sampler
                cfg.net_encode = ckpt_cfg.net_encode
                cfg.net_denoise = ckpt_cfg.net_denoise
            else:
                raise ValueError(f"Config does not contain required keys: {cfg.model}")
            logging.info(
                f"Overwriting model config with config from ckpt: {ckpt_cfg.model}"
            )
        else:
            raise ValueError(
                "Error in overwriting config with the loaded model config:\
                `overwrite_model_cfg` is set to True, but config.yaml not found in ckpt\
                directory or its parent. Set `overwrite_model_cfg` to False or move\
                             config file."
            )
    else:
        raise ValueError(
            "Error in overwriting config: No ckpt path specified, or ckpt path does not\
            exist. Set `overwrite_model_cfg` to False or spefic a correct ckpt path."
        )
    return cfg


def load_net_from_safetensors(ckpt_path):
    """Load only the net from an EncoderDecoderTrainer checkpoint"""
    state_dict = load_file(ckpt_path)
    # filter for net keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("net.")}
    # adjust keys
    adjusted_state_dict = {
        k.replace("net.", "", 1): v for k, v in filtered_state_dict.items()
    }
    return adjusted_state_dict


def get_latest_checkpoint_file(parent_dir):
    directories = glob.glob(f"{parent_dir}/*")
    directories.sort(key=os.path.getmtime, reverse=True)

    checkpoint_file = None

    for directory in directories:
        potential_checkpoint_file = os.path.join(
            directory, "checkpoints", "last", "model.safetensors"
        )
        if os.path.exists(potential_checkpoint_file):
            checkpoint_file = potential_checkpoint_file
            ckpt_path = os.path.join(directory, "checkpoints", "last")
            break

    if checkpoint_file is None:
        raise FileNotFoundError(
            f"No valid checkpoint file model.safetensors found in {parent_dir}!"
        )

    return checkpoint_file, ckpt_path
