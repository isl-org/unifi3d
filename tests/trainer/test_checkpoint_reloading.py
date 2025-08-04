import os
import hydra
from omegaconf import DictConfig, OmegaConf
from unifi3d.trainers.acc_trainer import Trainer
from unifi3d.data.dualoctree_dataset import DualOctreeDataModule


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
autoencoder_save_path_1 = os.path.abspath("./tmp/checkpoint_test_autoencoder_first")
autoencoder_save_path_2 = os.path.abspath("./tmp/checkpoint_test_autoencoder_resume")

diffusion_save_path_1 = os.path.abspath("./tmp/checkpoint_test_diffusion_first")
diffusion_save_path_2 = os.path.abspath("./tmp/checkpoint_test_diffusion_resume")


def test_autoencoder_checkpoints():
    trainer = Trainer(
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

    trainer2 = Trainer(
        default_root_dir=autoencoder_save_path_2,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )
    datamodule2 = hydra.utils.instantiate(data_module_cfg)
    autoencoder_model_cfg.net.add_kl_loss = True
    model2 = hydra.utils.instantiate(autoencoder_model_cfg)

    print("Resume training.")
    trainer2.fit(
        model=model2, datamodule=datamodule2, ckpt_path=ckpt_path, resume_training=True
    )
    print("Done trainer 2 fit.")


def test_diffusion_checkpoints():
    trainer = Trainer(
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
    trainer.fit(model=model, datamodule=datamodule, resume_training=False)

    diffusion_ckpt_path = os.path.join(diffusion_save_path_1, "checkpoints", "00002")
    assert os.path.exists(diffusion_ckpt_path), f"{diffusion_ckpt_path} does not exist!"

    trainer2 = Trainer(
        default_root_dir=diffusion_save_path_2,
        max_epochs=3,
        min_epochs=3,
        save_checkpoint_every_n_epoch=1,
    )
    datamodule2 = hydra.utils.instantiate(data_module_cfg)
    model2 = hydra.utils.instantiate(diffusion_model_cfg)

    print("Resume training.")
    trainer2.fit(
        model=model2,
        datamodule=datamodule2,
        ckpt_path=diffusion_ckpt_path,
        resume_training=True,
    )
