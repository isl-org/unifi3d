# This test can be run by running
# accelerate launch tests/trainer/acc_checkpoint_reloading.py
import os
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from accelerate import Accelerator
import logging
from typing import List
from unifi3d.utils.logger import Logger
from unifi3d.utils import (
    extras,
    get_metric_value,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from unifi3d.trainers.acc_trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

data_module_cfg = OmegaConf.load(
    os.path.join("tests", "configs", "doctree_shapenet_datamodule.yaml")
)
autoencoder_model_cfg = OmegaConf.load(
    os.path.join("tests", "configs", "doctree_autoencoder_trainer.yaml")
)
diffusion_model_cfg = OmegaConf.load(
    os.path.join("tests", "configs", "doctree_diffusion_trainer.yaml")
)

# last linking does not address relative path.
autoencoder_save_path_1 = os.path.abspath(
    "./tmp/multi_gpu_checkpoint_test_autoencoder_first"
)
autoencoder_save_path_2 = os.path.abspath(
    "./tmp/multi_gpu_checkpoint_test_autoencoder_resume"
)

diffusion_save_path_1 = os.path.abspath(
    "./tmp/multi_gpu_checkpoint_test_diffusion_first"
)
diffusion_save_path_2 = os.path.abspath(
    "./tmp/multi_gpu_checkpoint_test_diffusion_resume"
)


def main():
    ################################################################################
    # Setup profiler
    profile_kwargs = []
    profiling = False
    # Profiling uses the same default output_dir as logger.
    profiling_dir = os.path.join("./tmp", "profiling")

    # Needs to be executed for the first time inside the training function spawned on each GPU device.
    process_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=5400))]
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=profile_kwargs + process_kwargs,
    )

    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=autoencoder_save_path_1,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )

    datamodule = hydra.utils.instantiate(data_module_cfg)
    model = hydra.utils.instantiate(autoencoder_model_cfg)
    trainer.fit(model=model, datamodule=datamodule, resume_training=False)

    ckpt_path = os.path.join(autoencoder_save_path_1, "checkpoints", "00002")
    assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist!"

    del accelerator
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=profile_kwargs + process_kwargs,
    )

    trainer2 = Trainer(
        accelerator=accelerator,
        default_root_dir=autoencoder_save_path_2,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )
    datamodule2 = hydra.utils.instantiate(data_module_cfg)
    model2 = hydra.utils.instantiate(autoencoder_model_cfg)

    print("Resume training.")
    trainer2.fit(
        model=model2, datamodule=datamodule2, ckpt_path=ckpt_path, resume_training=True
    )

    # Diffusion training
    del accelerator
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=profile_kwargs + process_kwargs,
    )
    diffusion_trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=diffusion_save_path_1,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )

    datamodule = hydra.utils.instantiate(data_module_cfg)
    model = hydra.utils.instantiate(diffusion_model_cfg)
    encoder_decoder_ckpt_path = os.path.join(
        autoencoder_save_path_1, "checkpoints", "00002", "model.safetensors"
    )

    diffusion_model_cfg.encoder_decoder.ckpt_path = encoder_decoder_ckpt_path
    diffusion_trainer.fit(model=model, datamodule=datamodule, resume_training=False)

    diffusion_ckpt_path = os.path.join(diffusion_save_path_1, "checkpoints", "00002")
    assert os.path.exists(diffusion_ckpt_path), f"{diffusion_ckpt_path} does not exist!"

    del accelerator
    torch.cuda.empty_cache()
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=profile_kwargs + process_kwargs,
    )
    diffusion_trainer2 = Trainer(
        accelerator=accelerator,
        default_root_dir=diffusion_save_path_2,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )
    datamodule2 = hydra.utils.instantiate(data_module_cfg)
    model2 = hydra.utils.instantiate(diffusion_model_cfg)

    print("Resume training.")
    diffusion_trainer2.fit(
        model=model2,
        datamodule=datamodule2,
        ckpt_path=diffusion_ckpt_path,
        resume_training=True,
    )


if __name__ == "__main__":
    main()
