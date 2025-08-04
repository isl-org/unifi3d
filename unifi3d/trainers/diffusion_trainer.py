"""
A wrapper class for diffusion models containing a bunch of useful routines.

Inspired by: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
             https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
             and by SDFusion
"""

import torch
from typing import Any, Dict, Tuple


class DiffusionTrainer(torch.nn.Module):
    """
    Module for diffusion model training.
    """

    def __init__(
        self,
        diff_model,
        encoder_decoder,
        optimizer,
        scheduler,
        loss_fn,
        sampling_scheduler,
        loss_requires_grad,
        compile,
        device=None,
    ) -> None:
        """
        Initialize a auto encoder module named EncoderDecoderTrainer

        :param diff_model: The model to train.
        :param encoder_decoder: The encoder decoder model to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param loss_fn: The loss function to use for training.
        :param sampling_scheduler: The scheduler to use for sampling timesteps.
        :param loss_requires_grad: Whether the loss should require gradients.
        :param compile: compile model for faster training with pytorch 2.0
        :param device: Device on which the model is supposed to be loaded.
        """
        super(DiffusionTrainer, self).__init__()

        self.optimizer = optimizer
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diff_model = diff_model
        self.encoder_decoder = encoder_decoder

        if (
            hasattr(encoder_decoder, "is_transmitter")
            and encoder_decoder.is_transmitter
        ):
            self.encoder_decoder.build(self.device)

        for param in self.encoder_decoder.parameters():
            param.requires_grad = False

        self.sampling_scheduler = sampling_scheduler
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.loss_requires_grad = loss_requires_grad
        self.compile = compile
        self._count_parameters()
        self._compute_model_size()

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
            self.diff_model = torch.compile(self.diff_model)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        learnable_parameters = [
            p for p in self.diff_model.parameters() if p.requires_grad
        ]
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
        Perform a forward pass through the model `self.diff_model.encode_wrapper`.

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
        return self._shared_eval(batch, mode="eval")

    def _shared_eval(
        self,
        batch,
        noise=None,
        mode="train",
    ):
        """
        Perform a single encoder decoder step on a batch of data.

        :param batch: The input batch of data.
        :param c: Optional condition input, default is None.
        :param noise: Optional noise input, default is None.
        :param get_mesh: Flag indicating whether to generate and save meshes. Default is False.
        :param mode: "train" or "eval", determines if gradients are required.
        :return: A dictionary containing loss and statistics of noise and code.
        """
        # Encode the batch to latent space without gradient tracking
        with torch.no_grad():
            z = self.encoder_decoder.encode_wrapper(batch)

        # Generate noise if not provided
        noise = torch.randn_like(z, device=z.device) if noise is None else noise
        bs = z.shape[0]

        # Sample random timesteps for each element in the batch
        timesteps = torch.randint(
            0,
            self.sampling_scheduler.config.num_train_timesteps,
            (bs,),
            device=z.device,
            dtype=torch.int64,
        )

        # Perform the forward diffusion process by adding noise
        noisy_latent = self.sampling_scheduler.add_noise(z, noise, timesteps)

        # make sure to pass all necessary context
        context = {}
        for cond_mode in self.diff_model.conditioning_modes:
            context[cond_mode] = batch[cond_mode]

        # Determine if we need gradients for encoding
        requires_grad = mode == "train"
        diff_model_context = (
            torch.enable_grad() if requires_grad else torch.inference_mode()
        )

        with diff_model_context:
            # Predict the noise residual or sample using the diffusion model
            model_output = self.diff_model(noisy_latent, timesteps, context)
            # Compute the loss between predicted and actual noise
            if self.sampling_scheduler.prediction_type == "sample":
                loss = self.loss_fn(model_output, z)
            elif self.sampling_scheduler.prediction_type == "epsilon":
                loss = self.loss_fn(model_output, noise)
            else:
                raise NotImplementedError("only sample or epsilon models implemented")

        # Compute the loss between predicted and actual noise
        loss_dict = {"loss": loss}
        code_shaped = z.reshape(bs, -1)
        code_min = torch.min(code_shaped, dim=1)
        code_max = torch.max(code_shaped, dim=1)
        code_mean = torch.mean(code_shaped, dim=1)
        noise_shaped = noise.reshape(bs, -1)
        noise_min = torch.min(noise_shaped, dim=1)
        noise_max = torch.max(noise_shaped, dim=1)
        noise_mean = torch.mean(noise_shaped, dim=1)
        loss_dict["noise_min"] = noise_min.values
        loss_dict["noise_max"] = noise_max.values
        loss_dict["noise_mean"] = noise_mean
        loss_dict["code_min"] = code_min.values
        loss_dict["code_max"] = code_max.values
        loss_dict["code_mean"] = code_mean
        return loss_dict

    @torch.no_grad()
    def _inference(
        self,
        batch={},
        size=None,
        context={},
        num_sampling_steps=100,
        verbose=False,
    ):
        """
        Inference for diffusion: sample noise and apply model to denoise.

        :param batch: Dictionary containing input data. Default is empty.
        :param size: Size for the random noise tensor. If provided, overrides batch. Default is None.
        :param context: Dictionary containing conditioning data, e.g.
                {"text": ["airplane", "chair"], "image": [...]}. Default is empty.
        :param num_sampling_steps: Number of sampling steps for the diffusion process. Default is 100.
        :param verbose: If True, prints detailed logs during denoising. Default is False.
        :return: A dictionary with reconstructed samples.
        """
        if size is not None:
            # Generate random noise of the specified size on GPU
            samples = torch.randn(size).cuda()
        elif len(batch) > 0:
            # If batch is provided, encode it to latent space to determine size
            with torch.no_grad():
                z = self.encoder_decoder.encode_wrapper(batch)
            # pass context taken from the batch
            for cond_mode in self.diff_model.conditioning_modes:
                context[cond_mode] = batch[cond_mode]

            # Generate random noise with the same size as the latent code
            samples = torch.randn_like(z).cuda()
        else:
            # If neither batch nor size is provided, raise an error
            raise ValueError("one of batch or size must be provided")

        # Configure the number of timesteps in the sampling scheduler
        self.sampling_scheduler.set_timesteps(num_sampling_steps)

        # Denoise the samples over the specified timesteps
        for t in self.sampling_scheduler.timesteps:
            with torch.no_grad():
                # Prepare the current timestep tensor and run the diffusion model
                t_in = torch.tensor([t for _ in range(samples.shape[0])]).cuda().long()
                noisy_residual = self.diff_model(samples, t_in, context)

                # Compute the previous noisy sample
                prev_noisy_sample = self.sampling_scheduler.step(
                    noisy_residual, t, samples
                ).prev_sample
                samples = prev_noisy_sample

                # Optionally print statistics of the samples every 100 steps
                if verbose and t % 100 == 0:
                    print(
                        "Sample stats at step",
                        t,
                        torch.mean(samples).item(),
                        torch.std(samples).item(),
                    )

        # Decode the final denoised samples and add them to the batch dictionary
        batch["samples"] = samples
        batch["reconstructed"] = self.encoder_decoder.decode_wrapper(samples)
        return batch

    def _count_parameters(self) -> None:
        """
        Computes the number of parameters and the number of trainable parameters.
        """
        # Count the total number of parameters in the component
        self.num_params = sum(
            p.numel() for p in self.encoder_decoder.parameters()
        ) + sum(p.numel() for p in self.diff_model.parameters())
        # Count the number of trainable parameters in the component
        self.num_trainable_params = sum(
            p.numel() for p in self.encoder_decoder.parameters() if p.requires_grad
        ) + sum(p.numel() for p in self.diff_model.parameters() if p.requires_grad)

    def _compute_model_size(self):
        """
        Computes and stores the total size (in bytes) of the encoder-decoder model (self.encoder_decoder)
        and the diffusion model (self.diff_model), including both their parameters and buffers.

        The computed sizes are stored in the following instance variables:
        - self.total_size_ae: Total size of the encoder-decoder model.
        - self.total_size_diff: Total size of the diffusion model.
        - self.total_size: Combined total size of both models.
        """
        param_size_ae = sum(
            p.element_size() * p.numel() for p in self.encoder_decoder.parameters()
        )
        buffer_size_ae = sum(
            b.element_size() * b.numel() for b in self.encoder_decoder.buffers()
        )
        param_size_diff = sum(
            p.element_size() * p.numel() for p in self.diff_model.parameters()
        )
        buffer_size_diff = sum(
            b.element_size() * b.numel() for b in self.diff_model.buffers()
        )
        self.total_size_ae = param_size_ae + buffer_size_ae  # in bytes
        self.total_size_diff = param_size_diff + buffer_size_diff  # in bytes
        self.total_size = self.total_size_ae + self.total_size_diff
