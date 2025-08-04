"""
A wrapper class for autoencoder models containing a bunch of useful routines.

Inspired by: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
             https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
"""

import torch
from typing import Any, Dict, Tuple


class EncoderDecoderTrainer(torch.nn.Module):
    """
    Module for encoder-decoder model training.
    """

    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        loss,
        loss_requires_grad,
        compile,
        device=None,
    ) -> None:
        """
        Initialize a auto encoder module named EncoderDecoderTrainer

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param loss: The loss function to use for training.
        :param loss_requires_grad: Whether the loss function requires grad to be calculated even during eval.
        :param compile: compile model for faster training with pytorch 2.0
        :param device: Device on which the model is supposed to be loaded.
        """
        super(EncoderDecoderTrainer, self).__init__()
        self.optimizer = optimizer
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        if hasattr(net, "is_transmitter") and net.is_transmitter:
            self.net.build(self.device)

        # We use scheduler_config from init, because self.hparams does not have attribute scheduler for some reason.
        self.scheduler = scheduler
        self.loss = loss
        self.loss_requires_grad = loss_requires_grad
        self.compile = compile
        self._count_parameters()
        self.net._compute_model_size()

    def setup(
        self,
        stage: str,
    ) -> None:
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        learnable_parameters = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = self.optimizer(params=learnable_parameters)
        return [self.optimizer]

    def configure_scheduler(self) -> None:
        """
        Configure the scheduler for the optimizer.

        This method initializes the scheduler using the current optimizer.
        The scheduler controls the learning rate during training.
        """
        self.scheduler = self.scheduler(optimizer=self.optimizer)

    def forward(self, x, inference=False) -> Any:
        """
        Perform a forward pass through the model `self.net.encode_wrapper`.

        :param x: Input tensor.
        :param inference: if True, calls inference mode.
        :return: A dictionary containing the outputs and the computed loss.
        """
        if inference:
            return self._inference(x)
        else:
            if self.training:
                return self._training_step(x)
            else:
                return self._eval_step(x)

    def _training_step(
        self,
        batch,
    ):
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: Input batch.
        :return: A dictionary containing the outputs and the computed loss.
        """
        return self._shared_eval(batch, mode="train")

    def _eval_step(
        self,
        batch,
    ):
        """
        Perform a single eval step on a batch of data from the eval set.

        :param batch: Input batch.
        :return: A dictionary containing the outputs and the computed loss.
        """
        return self._shared_eval(batch, "eval")

    def _shared_eval(
        self,
        batch,
        mode="train",
    ):
        """
        Perform a single encoder decoder step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param mode: "train" or "eval", determines if gradients are required.
        :return: A dictionary containing the outputs and the computed loss.
        """
        # Determine if gradients are required
        requires_grad = mode == "train" or self.loss_requires_grad

        # Choose the appropriate context manager
        if requires_grad:
            context_manager = torch.enable_grad()
        else:
            # Use inference mode for better performance when gradients are not needed
            context_manager = torch.inference_mode()

        if hasattr(self.loss, "update_extra_loss_weight"):
            self.loss.update_extra_loss_weight(self.scheduler.last_epoch)

        with context_manager:
            output = self.net(batch)
            loss = self.loss(batch, output)

        # If the loss is a dictionary, we aggregate the loss terms
        if isinstance(loss, dict):
            # If the loss is a personalized loss, we aggregate loss as long
            # as it is passed back with a "loss" term in their key.
            losses = [val for key, val in loss.items() if "loss" in key]
            output["loss"] = torch.sum(torch.stack(losses))
        elif loss is not None:
            # If the loss is not a dictionary, we return it directly.
            output["loss"] = loss

        if hasattr(self.loss, "get_loss_statistics"):
            output["loss_statistics"] = self.loss.get_loss_statistics()
        return output

    def _inference(self, batch: torch.Tensor):
        """
        Perform inference on a batch of data.

        :param batch: A batch of data (a tensor) containing the input

        :return: the reconstructed representation (sdf/pointcloud/octree)
        """
        self.net.eval()

        # encode and decode with wrapper functions
        with torch.no_grad():
            encoded = self.net.encode_wrapper(batch)
            decoded = self.net.decode_wrapper(encoded)

        batch["reconstructed"] = decoded

        self.net.train()

        return batch

    def _count_parameters(self) -> None:
        """
        Computes the number of parameters and the number of trainable parameters.
        """
        # Count the total number of parameters in the component
        self.num_params = sum(p.numel() for p in self.net.parameters())
        # Count the number of trainable parameters in the component
        self.num_trainable_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
