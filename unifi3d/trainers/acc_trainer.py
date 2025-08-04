from accelerate import Accelerator, ProfileKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datetime import timedelta
import gc
import math
import numpy as np
import os
import glob
from rich.console import Console
import shutil
import torch
from tqdm import tqdm
import trimesh
from torch.amp import autocast
import time
from unifi3d.trainers.diffusion_trainer import DiffusionTrainer
from unifi3d.trainers.encoder_decoder_trainer import EncoderDecoderTrainer
from unifi3d.utils.visualization_utils import PyrenderRenderer
from unifi3d.utils.data.mesh_utils import load_normalize_mesh, load_npz_mesh
from unifi3d.utils.model.model_utils import get_latest_checkpoint_file


class Trainer:
    """
    Trainer main class designed to automate the training and evaluation process.
    """

    def __init__(
        self,
        accelerator=None,
        logger=[],
        max_epochs: int = -1,
        min_epochs: int = -1,
        max_steps: int = -1,
        min_steps: int = -1,
        mixed_precision: str = "no",  # TODO: Remember to remove this once all code is migrated.
        check_val_every_n_epoch: int = -1,
        log_every_n_iter: int = -1,
        log_image_every_n_iters: int = -1,
        save_checkpoint_every_n_epoch: int = 100,
        ckpt_num: int = 5,
        loss_clip_val: float = 0.0,
        gradient_clip_val: float = 0.0,
        detect_anomaly: bool = False,
        default_root_dir="",
        batch_post_process=None,
        profiling=None,
        seed=12345,
    ) -> None:
        """
        Customize every aspect of training via flags.

        :param logger: Logger (or iterable collection of loggers) for experiment tracking. Default: None.
        :param max_epochs: Stop training once this number of epochs is reached. Disabled by default (-1).
        :param min_epochs: Force training for at least these many epochs. Disabled by default (-1).
        :param max_steps: Stop training after this number of steps. Disabled by default (-1).
        :param min_steps: Force training for at least these number of steps. Disabled by default (-1).
        :param mixed_precision: Mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'. Default: 'no'.
        :param check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. Disabled by default (-1).
        :param log_every_n_iter: How often to log within steps. Default: 1000.
        :param log_image_every_n_iters: Log images every N iterations. Disabled by default (-1).
        :param save_checkpoint_every_n_epoch: Log checkpoints after every `N` training epochs. Default: 100.
        :param ckpt_num: Keep the latest N checkpoint to avoid oversaving checkpoints. Default: 5.
        :param loss_clip_val: The value at which to clip the loss. Default: 0.0.
        :param gradient_clip_val: The value at which to clip gradients. Default: 0.0.
        :param detect_anomaly: Enable anomaly detection for the autograd engine. Default: False
        :param default_root_dir: Default path for logs and weights when no logger passed.
        :param batch_post_process: Optional extra postprocessing step
        :param profiling: Dictionary containing the profiler settings.
        :param seed: Reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
        """
        # Logging & Loggers setup
        self.log = get_logger(__name__, log_level="INFO")
        self._loggers = logger
        # Default log directories
        if (
            logger == None or len(logger) == 0
        ):  # Mostly in case of hyoperparameter optimization:
            self.default_root_dir = default_root_dir
            self.ckpt_dir = os.path.join(self.default_root_dir, "checkpoints")
            self.log_dir = os.path.join(self.default_root_dir, "log")
            self.tracked_variables = {"metric": ["loss"]}
        else:  # General case:
            self.default_root_dir = logger[0].root_dir
            self.ckpt_dir = logger[0].ckpt_dir
            self.log_dir = logger[0].log_dir
            self.tracked_variables = self.logger.tracked_variables
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Up to data metrics dictionary for hyperparameter optimization
        self._metrics = {"val_loss": float("inf")}

        # Profiling configuration
        self.profiling = profiling

        if accelerator is not None:
            self.accelerator = accelerator
        else:
            ################################################################################
            profiling_dir = os.path.join(self.default_root_dir, "profiling")
            process_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=5400))]
            if self.profiling and self.profiling.get("activated", False):
                # Setup profiler
                process_kwargs += [
                    ProfileKwargs(
                        activities=self.profiling.get("activities", ["cpu", "cuda"]),
                        record_shapes=self.profiling.get("record_shapes", True),
                        with_flops=self.profiling.get("with_flops", True),
                        profile_memory=self.profiling.get("profile_memory", True),
                        with_modules=self.profiling.get("with_modules", True),
                        output_trace_dir=profiling_dir,
                    )
                ]

            # Needs to be executed for the first time inside the training function spawned on each GPU device.
            self.accelerator = Accelerator(
                mixed_precision=mixed_precision,
                kwargs_handlers=process_kwargs,
            )
        # For reproducible behavior
        set_seed(seed)
        ################################################################################

        assert max_epochs >= min_epochs and max_steps >= min_steps
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.log_every_n_iter = log_every_n_iter
        self.log_image_every_n_iters = log_image_every_n_iters
        self.save_checkpoint_every_n_epoch = save_checkpoint_every_n_epoch
        self.ckpt_num = ckpt_num
        self.last_model_path = ""
        self.best_model_path = ""
        self.best_val_loss = float("inf")
        self.loss_clip_val = loss_clip_val
        self.gradient_clip_val = gradient_clip_val
        self.detect_anomaly = detect_anomaly
        self.total_iters = 0
        self.total_train_iters = 0
        self.total_val_iters = 0
        self.total_test_iters = 0
        self.batch_post_process = batch_post_process
        self.batch_size = -1
        self.lr = -1.0
        self.render_config = {
            "cam_angle_yaw": np.pi / 6.0,
            "yfov": np.pi / 2.0,
            "cam_dist": 0.75,
            "shape": (640, 480),
        }
        self.renderer = PyrenderRenderer(self.render_config)

        # Low-pass filtered validation loss delta
        self.early_terminate = {
            "alpha": 0.999,  # Exponential filter for the validation loss
            "val_loss": 10000.0,  # Raw validation loss
            "val_loss_old_f": 1e20,  # Old value of the filtered validation loss
            "threshold": 1e-5,  # Minimum improvement of the filtered validation loss
        }

    def fit(self, model, datamodule, ckpt_path=None, resume_training=True) -> None:
        """
        Runs the full optimization routine.

        :param model: Model to fit.
        :param datamodule: A DataModule that defines the train_dataloader hook.
        :param ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.
        :param resume_training: If True, tries to load the checkpoint in ckpt_path
        """
        # Print status
        console = Console()
        if self.accelerator.is_main_process:
            console.rule("[bold]Setting up Trainer.")
            start_time = time.time()

        # Configure dataloader
        self.log.info(f"Setup dataloader.")
        datamodule.setup()
        if hasattr(datamodule, "batch_size"):
            batch_size = datamodule.batch_size
        elif hasattr(datamodule, "train_batch_size"):
            batch_size = datamodule.train_batch_size
        else:
            raise Exception(
                f"Cannot find batch_size or train_batch_size in {datamodule}"
            )
        self.batch_size = batch_size

        # Module setup
        self.log.info(f"Setup datamodules.")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Configure model
        self.log.info(f"Setup model.")
        model.configure_optimizers()
        model.configure_scheduler()
        self.log.info(
            "The loaded model has {:.3f} millions of parameters, from which {:.3f} millions are trainable.".format(
                model.num_params / 1e6, model.num_trainable_params / 1e6
            )
        )

        # Accelerate should not prepare scheduler so that scheduler is always
        # properly updated regardless how many GPUs are used.
        scheduler = model.scheduler
        self.lr = scheduler.get_last_lr()

        # Setup modules for accelerator
        self.log.info(f"Setup modules for accelerator.")
        (
            model,
            model.optimizer,
            train_loader,
            val_loader,
        ) = self.accelerator.prepare(
            model,
            model.optimizer,
            train_loader,
            val_loader,
        )

        # Register model for checkpointing
        # TODO: Uncomment for backward compatibility. Remove if adapted completely.
        # self.log.info(f"Register model for checkpointing")
        # self.accelerator.register_for_checkpointing(model)

        # Load checkpoint
        resume_info = self._load_checkpoint(
            ckpt_path=ckpt_path,
            resume_training=resume_training,
            is_training=True,
            model=model,
        )
        # Profiler step
        if self.profiling and self.profiling.get("activated", False):
            self._run_profiling(model, train_loader)
            torch.cuda.empty_cache()  # freeing memory used during running the profiler

        start_epoch = 1
        # Main training / evaluation loop

        if resume_training:
            console.rule("[bold]Training...")
            if "epoch" in resume_info:
                start_epoch = resume_info["epoch"] + 1
                print("start_epoch:", start_epoch)
            if "scheduler_state_dict" in resume_info:
                scheduler.load_state_dict(resume_info["scheduler_state_dict"])
                print("Loaded scheduler state dict.")
        for epoch in tqdm(
            range(start_epoch, self.max_epochs + 1),
            ncols=8,
            disable=not self.accelerator.is_local_main_process or not self.logger,
        ):
            # Training routine
            self._train_epoch(
                model=model,
                scheduler=scheduler,
                train_loader=train_loader,
                epoch=epoch,
            )
            # Evaluation routine
            if (
                self.check_val_every_n_epoch > 0
                and epoch % self.check_val_every_n_epoch == 0
            ):
                self._validate_epoch(
                    model=model,
                    val_loader=val_loader,
                    epoch=epoch,
                )

                # Early terminate condition check
                if self._epoch_should_terminate_early(
                    val_loss=self.early_terminate["val_loss"],
                    epoch=epoch,
                ):
                    break

            # Save checkpoint periodically
            if (
                self.save_checkpoint_every_n_epoch > 0
                and epoch % self.save_checkpoint_every_n_epoch == 0
            ):
                self._save_checkpoint(model=model, scheduler=scheduler, epoch=epoch)
            if self.accelerator.is_main_process:
                end_time = time.time()
                duration = end_time - start_time
                hours, rem = divmod(duration, 3600)
                minutes, seconds = divmod(rem, 60)
                formatted_duration = "{:02}:{:02}:{:02}".format(
                    int(hours), int(minutes), int(seconds)
                )
                self.log.info(f"Running time:{formatted_duration}")
            # Garbage collector
            gc.collect()

        # Save final checkpoint
        self._save_checkpoint(model=model, scheduler=scheduler, epoch=epoch)

        # End the logging process
        self._end_log()

        # Print status
        if self.accelerator.is_main_process:
            console.rule("[bold]Done!")

    def test(self, model, datamodule, ckpt_path=None) -> None:
        """
        Runs the full test routine.

        :param model: Model to test.
        :param datamodule: A DataModule that defines the train_dataloader hook.
        :param ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.
        """
        # Print status
        console = Console()
        if self.accelerator.is_main_process:
            console.rule("[bold]Setting up Test routine.")

        # Configure dataloader
        self.log.info(f"Setup dataloader.")
        datamodule.setup()
        if hasattr(datamodule, "batch_size"):
            batch_size = datamodule.batch_size
        elif hasattr(datamodule, "test_batch_size"):
            batch_size = datamodule.test_batch_size
        else:
            raise Exception(
                f"Cannot find batch_size or test batch size in {datamodule}"
            )
        self.batch_size = batch_size

        # Module setup
        self.log.info(f"Setup datamodules.")
        test_loader = datamodule.test_dataloader()

        # Configure model
        self.log.info(f"Setup model.")

        # Setup modules for accelerator
        self.log.info(f"Setup modules for accelerator.")
        (model, test_loader) = self.accelerator.prepare(model, test_loader)

        # Load checkpoint
        self._load_checkpoint(
            ckpt_path=ckpt_path, resume_training=False, is_training=False
        )

        # Profiler step
        if self.profiling and self.profiling.get("activated", False):
            self._run_profiling(model, test_loader)

        # Main test routine
        self._test_epoch(model=model, test_loader=test_loader)

        # Print status
        if self.accelerator.is_main_process:
            console.rule("[bold]Done!")

    ###################################################################
    ##########          TRAIN - VAL - TEST ROUTINES          ##########
    ###################################################################

    def _train_epoch(
        self,
        model,
        scheduler,
        train_loader,
        epoch,
    ) -> None:
        """
        Execute one training epoch over a batch of data.

        :param model: Either a DiffusionTrainer or an EncoderDecoderTrainer model
        :param scheduler: The train scheduler
        :param train_loader: The train dataloader
        :param epoch: Current epoch
        """
        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            # Set training mode
            model.train()

            # Base log data
            log_kwargs = {
                "epoch": epoch,  # Current epoch
                # Number of data batches provided at the current epoch by the data loader.
                "batch_total_num": len(train_loader),
                "phase": "train",  # "train", "val" or "test"
            }

            # Batch loop
            print(f"There are total {len(train_loader)} training steps.")
            for step, batch in tqdm(
                enumerate(train_loader),
                total=None,
                ncols=80,
                leave=False,
                disable=not self.accelerator.is_local_main_process or not self.logger,
            ):
                # print(f"Running step {step}/{len(train_loader)}")
                # Reset gradients of all model parameters.
                model.optimizer.zero_grad()

                # Forward
                output = model(batch)

                for key, val in output.items():
                    batch[key] = val

                # Clip loss
                batch["loss"] = self._clip_loss(batch["loss"])

                # Backward pass to compute gradients
                self.accelerator.backward(batch["loss"])

                # Clip gradient
                if self.gradient_clip_val > 0:
                    self.accelerator.clip_grad_norm_(
                        model.parameters(),
                        self.gradient_clip_val,
                    )

                # Update model parameters
                model.optimizer.step()

                # Log iteration data
                self._log_iteration_data(
                    batch=batch,
                    model=model,
                    data_loader=train_loader,
                    step=step,
                    **log_kwargs,
                )

                # Early terminate condition check
                if self._step_should_terminate_early(step):
                    break

                self.total_train_iters += 1
                self.total_iters += 1
            # Log epoch update function
            self._update_log()

            # Adjust learning rate
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                scheduler.step()
                self.lr = scheduler.get_last_lr()

    def _validate_epoch(
        self,
        model,
        val_loader,
        epoch,
    ) -> None:
        """
        Execute one validation epoch over a batch of data.

        :param model: Either a DiffusionTrainer or an EncoderDecoderTrainer model
        :param val_loader: The validation dataloader
        :param epoch: Current epoch
        """
        # Set inference mode
        model.eval()

        # Base log data
        log_kwargs = {
            "epoch": epoch,  # Current epoch
            # Number of data batches provided at the current epoch by the data loader.
            "batch_total_num": len(val_loader),
            "phase": "val",  # "train", "val" or "test"
        }
        print(
            f"Running validation epoch. There are {len(val_loader)} steps in validation"
        )
        # Batch loop
        for step, batch in tqdm(
            enumerate(val_loader),
            total=None,
            ncols=80,
            leave=False,
            disable=not self.accelerator.is_local_main_process or not self.logger,
        ):
            # Forward
            # print(f"Running validation step {step}/{len(val_loader)}")
            with torch.no_grad():
                output = model(batch)

            for key, val in output.items():
                batch[key] = val

            # Record validation loss
            batch["val_loss"] = output["loss"]
            self._metrics["val_loss"] = output["loss"].mean()
            self.early_terminate["val_loss"] = self._metrics["val_loss"].detach()

            # Log iteration data
            self._log_iteration_data(
                batch=batch,
                model=model,
                data_loader=val_loader,
                step=step,
                **log_kwargs,
            )

            # Early terminate condition check
            if self._step_should_terminate_early(step):
                break

            self.total_val_iters += 1
            self.total_iters += 1

        # Log epoch update function
        self._update_log()

    def _test_epoch(
        self,
        model,
        test_loader,
    ) -> None:
        """
        Execute one test epoch over a batch of data.

        :param model: Either a DiffusionTrainer or an EncoderDecoderTrainer model
        :param test_loader: The test dataloader
        """
        # Set inference mode
        model.eval()

        # Base log data
        log_kwargs = {
            "epoch": 1,  # Current epoch
            # Number of data batches provided at the current epoch by the data loader.
            "batch_total_num": len(test_loader),
            "phase": "test",  # "train", "val" or "test"
        }

        # Batch loop
        for step, batch in tqdm(
            enumerate(test_loader),
            total=None,
            ncols=80,
            leave=False,
            disable=not self.accelerator.is_local_main_process or not self.logger,
        ):
            # Forward
            with torch.no_grad():
                output = model(batch)
            for key, val in output.items():
                if key == "loss":
                    batch["test_loss"] = batch["loss"]
                else:
                    batch[key] = val
            # Log iteration data
            self._log_iteration_data(
                batch=batch,
                model=model,
                data_loader=test_loader,
                step=step,
                **log_kwargs,
            )

            # Optional extra postprocessing step
            if self.batch_post_process is not None:
                self.batch_post_process(batch, output)

            # Early terminate condition check
            if self._step_should_terminate_early(step):
                break

            self.total_test_iters += 1
            self.total_iters += 1

        # Log epoch update function
        self._update_log()

    ###################################################################
    ####################           UTILS           ####################
    ###################################################################

    def _epoch_should_terminate_early(
        self,
        val_loss,
        epoch,
    ):
        """
        Check whether the epoch loop should be interrupted or not.

        :param val_loss: validation loss.
        :param epoch: the number of training epochs.
        :return: True if the loop should be interrupted, False otherwise.
        """
        met_min_epochs = epoch >= self.min_epochs if self.min_epochs > 0 else True

        # Exponential filter over the validation loss: x = ax + (1-a)y
        params = self.early_terminate
        alpha = params["alpha"]

        # Get the previous filtered loss
        val_loss_old_f = params["val_loss_old_f"]

        # Compute the new filtered loss
        val_loss_f = alpha * val_loss_old_f + (1 - alpha) * val_loss

        # Calculate the change in the filtered loss
        delta_val_loss = val_loss_old_f - val_loss_f

        # Update the filtered loss in parameters
        params["val_loss_old_f"] = val_loss_f

        # Termination condition
        if met_min_epochs and delta_val_loss < params["threshold"]:
            self.log.warning(f"Early termination criteria met at epoch {epoch}.")
            return True

        return False

    def _step_should_terminate_early(self, step):
        """
        Check whether the step loop should be interrupted or not.

        :param step: the number of iterations of the loop.
        :return: True if the loop should be interrupted, False otherwise.
        """
        met_min_steps = step >= self.min_steps if self.min_steps > 0 else True
        met_max_steps = step >= self.max_steps if self.max_steps > 0 else False
        if met_min_steps and met_max_steps:
            self.log.warning(
                f"Early terminate at step {step} > {self.max_steps} (max_steps)."
            )
            return True
        return False

    def _clip_loss(self, loss):
        """
        Clip the value of the loss between -self.loss_clip_val and self.loss_clip_val

        :param loss: the loss value
        :return: the clipped loss value
        """
        if self.loss_clip_val > 0.0:
            if abs(loss) > self.loss_clip_val:
                self.log.warning(
                    f"Loss Clipped from {abs(loss)} to {self.loss_clip_val}"
                )
                loss = torch.clamp(loss, -self.loss_clip_val, self.loss_clip_val)
        return loss

    def _save_checkpoint(
        self,
        model,
        scheduler,
        epoch=-1,
    ) -> None:
        """
        Saves a checkpoint of the model state and updates symbolic links to the latest and best checkpoints.

        :param model: the module whose state should be saved.
        :param scheduler: the scheduler state.
        :param epoch: the number of training epochs.
        """
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Construct checkpoint path
            ckpt_name = f"{epoch:05d}"
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            self.last_model_path = os.path.join(self.ckpt_dir, "last")
            self.best_model_path = os.path.join(self.ckpt_dir, "best")

            # Unwrap model (for parallel training)
            # model = self.accelerator.unwrap_model(model)

            # Save checkpoint
            self.accelerator.save_state(ckpt_path)

            # Save additional info for resuming training
            resume_info = {
                "epoch": epoch,
                "scheduler_state_dict": scheduler.state_dict(),
            }
            resume_info_path = os.path.join(ckpt_path, "resume_info.pth")
            with open(resume_info_path, "wb") as f:
                torch.save(resume_info, f)
            print(f"Saved resume info to {resume_info_path}")

            # Clean up old checkpoints, excluding 'best' and 'last'
            ckpts = sorted(os.listdir(self.ckpt_dir))
            # Exclude 'best' and 'last' directories from cleanup
            ckpts = [ckpt for ckpt in ckpts if ckpt not in ["best", "last"]]
            if len(ckpts) > self.ckpt_num:
                for ckpt in ckpts[: -self.ckpt_num]:
                    shutil.rmtree(os.path.join(self.ckpt_dir, ckpt))

            # Update the 'last' checkpoint by copying files
            if os.path.exists(self.last_model_path):
                shutil.rmtree(self.last_model_path)
            shutil.copytree(ckpt_path, self.last_model_path)

            # Update symbolic link to the "best" checkpoint if current val_loss is better
            current_val_loss = self._metrics["val_loss"]

            # Check if val_loss has been updated from its initial value
            if not math.isinf(current_val_loss):
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    # Update the 'best' checkpoint by copying files
                    if os.path.exists(self.best_model_path):
                        shutil.rmtree(self.best_model_path)
                    shutil.copytree(ckpt_path, self.best_model_path)
                    print(
                        f"New best model found with val_loss {current_val_loss:.4f}, saved to {self.best_model_path}"
                    )
            else:
                print(
                    "Validation loss not updated yet; skipping best checkpoint update."
                )

            # Use accelerator.print to print only on the main process. Otherwise, it will
            # slow down running speed.
            print(f"Saving states to {ckpt_path}")

    def _load_checkpoint(
        self, ckpt_path=None, resume_training=False, is_training=True, model=None
    ) -> None:
        """
        Load an existing model checkpoint.

        :param ckpt_path: Path of the checkpoint to be loaded. If None, attempts to load the 'best' or 'last' checkpoint.
        :param resume_training: Whether to resume training (True) or not (False).
        :param is_training: Whether the model is in training mode.
        :return: Dictionary containing resume information (e.g., epoch number).
        """
        if resume_training:
            assert is_training, "Cannot resume training if not in training mode."
        self.accelerator.wait_for_everyone()

        resume_info = {}
        # All process should load models.
        ckpt_found = False
        checkpoint_file = None

        if ckpt_path is None:
            # Attempt to load the "best" checkpoint first
            best_ckpt_path = os.path.join(self.ckpt_dir, "best")
            best_checkpoint_file = os.path.join(best_ckpt_path, "model.safetensors")

            if os.path.exists(best_checkpoint_file):
                ckpt_path = best_ckpt_path
                checkpoint_file = best_checkpoint_file
                ckpt_found = True
                print(f"Loading best checkpoint from {ckpt_path}")
            else:
                # If "best" checkpoint doesn't exist, try the "last" checkpoint
                last_ckpt_path = os.path.join(self.ckpt_dir, "last")
                last_checkpoint_file = os.path.join(last_ckpt_path, "model.safetensors")

                if os.path.exists(last_checkpoint_file):
                    ckpt_path = last_ckpt_path
                    checkpoint_file = last_checkpoint_file
                    ckpt_found = True
                    print(f"Loading last checkpoint from {ckpt_path}")
                else:
                    # Neither "best" nor "last" checkpoints exist
                    print(f"No best or last checkpoints found in {self.ckpt_dir}.")
        else:
            # If a specific checkpoint path is provided, verify it exists
            checkpoint_file = os.path.join(ckpt_path, "model.safetensors")
            if os.path.exists(checkpoint_file):
                ckpt_found = True
                print(f"Loading checkpoint from provided path: {checkpoint_file}")
            else:
                print(f"Checkpoint file {checkpoint_file} does not exist.")

        if not ckpt_found:
            if resume_training:
                # Try to find the latest checkpoint in the parent directory
                parent_dir = os.path.dirname(self.default_root_dir)
                checkpoint_file, ckpt_path = get_latest_checkpoint_file(parent_dir)

                if ckpt_path and os.path.exists(checkpoint_file):
                    ckpt_found = True
                    print(f"Found latest checkpoint in parent directory: {ckpt_path}")
                else:
                    raise FileNotFoundError("No checkpoint found to resume training.")
            else:
                if is_training:
                    # Starting training from scratch
                    print("No checkpoint found. Starting training from scratch.")
                    return resume_info
                else:
                    # During testing, a checkpoint must exist
                    raise FileNotFoundError(
                        "During testing, checkpoint file must exist."
                    )

        self.accelerator.load_state(ckpt_path)
        print(f"Loaded the checkpoint from {ckpt_path}")

        if resume_training:
            # Load additional information required to resume training
            print("Resuming training, loading resume_info.pth")
            resume_info_path = os.path.join(ckpt_path, "resume_info.pth")

            if os.path.exists(resume_info_path):
                resume_info = torch.load(
                    resume_info_path, map_location=self.accelerator.device
                )
                epoch = resume_info.get("epoch")
                if epoch is not None:
                    print(f"Loaded scheduler state. Resuming from epoch {epoch}.")
                else:
                    print("Epoch information not found in resume_info.pth.")
            else:
                print(f"resume_info.pth not found in {ckpt_path}.")
        else:
            print("Checkpoint loaded for evaluation/testing.")
        self.best_model_path = ckpt_path
        self.last_model_path = ckpt_path
        return resume_info

    ###################################################################
    ######################         LOGGING        #####################
    ###################################################################

    @property
    def logger(self):
        """
        The first Logger being used.
        """
        return self._loggers[0] if self._loggers and len(self._loggers) > 0 else None

    @logger.setter
    def logger(self, logger) -> None:
        if not logger:
            self._loggers = []
        else:
            self._loggers = [logger]

    @property
    def loggers(self):
        """
        The list of Logger used.
        """
        return self._loggers

    @loggers.setter
    def loggers(self, loggers) -> None:
        self._loggers = loggers if loggers else []

    @property
    def updated_metrics(self):
        """
        Recover the last logged metric values (if any).
        """
        return self._detach_before_return(self._metrics)

    def _log_iteration_data(
        self,
        batch,
        model,
        data_loader,
        step,
        **log_kwargs,
    ) -> None:
        """
        Log iteration data.
        Performs rendering of mesh representation if required.

        :param batch: A dict containing the data to be logged.
        :param model: Neural Module.
        :param data_loader: The dataloader object.
        :param step: Number of batches processed in the current epoch (reset to 0 at every epoch).
        :param log_kwargs: Additional logging arguments.
        """
        self.accelerator.wait_for_everyone()
        # Add general batch information
        phase = log_kwargs["phase"]

        # Logs for training based on log_every_n_iter, and log all validations to
        # avoid cases where when log_every_n_iter always skips validation
        # when total_iters is a even number, and validation happens to be one odd turn.
        if (
            self.accelerator.is_main_process
            and self.log_every_n_iter > 0
            and (self.total_train_iters % self.log_every_n_iter == 0 or phase == "val")
        ):
            # If required, renders image from the mesh representation
            if self.log_image_every_n_iters > 0 and (
                self.total_train_iters % self.log_image_every_n_iters == 0
                or phase == "val"
            ):
                batch = model(batch, inference=True)
                if "logits" in batch.keys() and isinstance(
                    batch["logits"], torch.Tensor
                ):
                    logits = batch["logits"].unsqueeze(-1)
                    logit_min, logit_max = logits.min(), logits.max()
                    batch["logits_min"] = logit_min
                    batch["logits_max"] = logit_max

                batch = self._log_rendered_meshes(
                    batch, data_loader=data_loader, model=model, **log_kwargs
                )

            # This field does not need to be updated during validation.
            if "scheduler" in log_kwargs:
                batch.update(
                    {"lr": torch.Tensor(log_kwargs["scheduler"].get_last_lr())}
                )

            batch.update(
                {
                    "lr": torch.Tensor(self.lr),
                    "phase": phase,
                    "epoch": log_kwargs["epoch"],
                    "max_epochs": self.max_epochs,
                    "batch_size": self.batch_size,
                    # Number of batches processed in the current epoch (reset to 1 at every epoch).
                    "batch_current": step + 1,  # +1 to start from 1
                    # Total -- maximum -- number of batches to be processed in the current epoch.
                    "batch_total": (
                        min(self.max_steps, log_kwargs["batch_total_num"])
                        if self._step_should_terminate_early(step)
                        else log_kwargs["batch_total_num"]
                    ),
                    # Dict of logged quantities
                    "output_parser": self.tracked_variables,
                }
            )

            # Add phase-specific information
            if phase == "train":
                batch["total_iters"] = self.total_train_iters
            elif phase == "val":
                batch["total_iters"] = self.total_val_iters
            elif phase == "test":
                batch["total_iters"] = self.total_test_iters
            else:
                batch["total_iters"] = self.total_iters

            # Track the averaged tensors
            tracked_metrics = self.tracked_variables["metric"]
            for key in tracked_metrics:
                # Average the metric
                if key in batch:
                    if isinstance(batch[key], dict):
                        assert (
                            key == "loss_statistics"
                        ), "Other behaviour not guaranteed to work yet. Manually check"
                        continue
                        # if the value is a dict (loss_statistics)
                    batch[key] = batch[key].mean()
                    # Store the metric if in the training phase
                    if phase == "train":
                        self._metrics[key] = batch[key]
                else:
                    print(f"Cannot find log {key}, it cannot be found in the batch.")
            # Detach the tensor to save GPU memory
            batch = self._detach_before_return(batch)

            # Execute logging
            for logger in self._loggers:
                logger.log_iter(batch)

    def _detach_before_return(self, batch):
        """
        Detach tensors in the batch to remove them from the computation graph and save GPU memory.

        :param batch: A dict containing the data to be logged.
        :return: The updated batch.
        """
        for key, val in batch.items():
            if isinstance(val, dict):
                # Recursively detach tensors within nested dictionaries
                self._detach_before_return(val)
            elif isinstance(val, torch.Tensor):
                # Only detach if the tensor requires grad, reducing unnecessary detachment
                if val.requires_grad:
                    batch[key] = val.detach()
        return batch

    def _update_log(self) -> None:
        """
        Update epoch log data.
        """
        if self.accelerator.is_main_process:
            for logger in self._loggers:
                logger.log_epoch()

    def _end_log(self) -> None:
        """
        Cleanly stops the logger.
        """
        if self.accelerator.is_main_process:
            for logger in self._loggers:
                logger.end_log()

    def _log_rendered_meshes(self, batch, data_loader, model=None, **log_kwargs):
        """
        Renders meshes and stores the resulting images into the input batch.

        :param batch: A dict containing the data to be logged.
        :param data_loader: The dataloader object.
        :param model: Neural Module.
        :param log_kwargs: Additional logging arguments.
        :return: The updated batch.
        """

        def meshes_to_trimesh(meshes):
            result = []
            for mesh in meshes:
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)

                if len(vertices) < 3 or len(faces) < 1:
                    # Not enough vertices or faces to compute normals, handle accordingly
                    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    result.append(tri_mesh)
                    continue

                try:
                    # Recompute normals
                    # mesh.compute_vertex_normals()
                    # mesh.compute_triangle_normals()

                    vertex_normals = np.asarray(mesh.vertex_normals)
                    triangle_normals = np.asarray(mesh.triangle_normals)

                    # Create a Trimesh object including normals
                    tri_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        face_normals=triangle_normals,
                        vertex_normals=vertex_normals,
                    )
                except Exception as e:
                    print(f"Failed to compute normals for the mesh: {e}")
                    # Create Trimesh without normals in case of exception
                    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                result.append(tri_mesh)

            return result

        def load_mesh_from_path(paths, is_batch, is_npz=False):
            mesh_scale = 1 - data_loader.dataset.padding
            if is_npz:
                normalized_meshes = [load_npz_mesh(path) for path in paths]
            else:
                normalized_meshes = (
                    [load_normalize_mesh(path, mesh_scale) for path in paths]
                    if is_batch
                    else [load_normalize_mesh(paths[0], mesh_scale)]
                )
            return meshes_to_trimesh(normalized_meshes)

        def reconstruct_meshes(decoded_output, is_batch):
            decoded = decoded_output if is_batch else [decoded_output[0]]
            try:
                reconstructed_meshes = [
                    data_loader.dataset.representation_to_mesh(dec) for dec in decoded
                ]
            except AttributeError:
                unwraped_model = self.accelerator.unwrap_model(model)
                if isinstance(unwraped_model, DiffusionTrainer):
                    reconstructed_meshes = [
                        unwraped_model.encoder_decoder.representation_to_mesh(dec)
                        for dec in decoded
                    ]
                elif isinstance(unwraped_model, EncoderDecoderTrainer):
                    reconstructed_meshes = [
                        unwraped_model.net.representation_to_mesh(dec)
                        for dec in decoded
                    ]
                else:
                    raise NotImplementedError(
                        f"Reconstruction for {type(model)} is not implemented."
                    )

            return meshes_to_trimesh(reconstructed_meshes)

        def reconstruct_meshes_wo_encoding(batch, is_batch):
            decoded_output = data_loader.dataset.get_representation_from_batch(batch)
            decoded = decoded_output if is_batch else [decoded_output[0]]
            reconstructed_meshes = [
                data_loader.dataset.representation_to_mesh(dec) for dec in decoded
            ]

            return meshes_to_trimesh(reconstructed_meshes)

        def render_meshes(meshes):
            rendered = []
            for mesh in meshes:
                try:
                    rendered.append(
                        self.renderer.render(meshes=[mesh]).transpose(2, 0, 1)
                    )
                except:
                    rendered.append(
                        np.zeros(
                            (
                                3,
                                self.render_config["shape"][1],
                                self.render_config["shape"][0],
                            )
                        )
                    )
            return torch.Tensor(np.stack(rendered, axis=0))

        if len(self._loggers) > 0:
            render_batch = False  # If True, renders the whole batch, else render only the first element
            red = torch.tensor([255, 0, 0], dtype=torch.float32)
            blue = torch.tensor([0, 0, 255], dtype=torch.float32)

            def process_meshes(log_type, batch, render_batch):
                if log_type in ["mesh_gt", "render_gt"]:
                    if "mesh_path" in batch:
                        return load_mesh_from_path(
                            batch["mesh_path"], render_batch, is_npz=True
                        )
                    elif "file_path" in batch:
                        return load_mesh_from_path(
                            batch["file_path"], render_batch, is_npz=False
                        )
                    elif "sdf" in batch:
                        return reconstruct_meshes(batch["sdf"], render_batch)
                elif log_type in ["mesh_rec", "render_rec"]:
                    if "reconstructed" in batch:
                        return reconstruct_meshes(batch["reconstructed"], render_batch)
                elif log_type in ["mesh_rec_gt", "render_rec_gt"]:
                    return reconstruct_meshes_wo_encoding(batch, render_batch)
                elif log_type in ["surface_points"]:
                    return batch["pcl"]
                elif log_type in ["occupancy_sample_points"]:
                    query_points = batch["query_points"]
                    if "occupancy" in batch:
                        occupancy = batch["occupancy"]
                    else:
                        assert (
                            "occu" in batch
                        ), "occupancy_sample_points must have occu or occupancy as keys in data loader."
                        occupancy = batch["occu"]
                    occupancy_rgb = torch.where(
                        occupancy.unsqueeze(-1) == 1,
                        red.to(occupancy),  # maps 1 to red
                        blue.to(occupancy),  # maps 0 to blue
                    )
                    return torch.cat((query_points, occupancy_rgb), dim=-1)
                elif log_type in ["sdf_sample_points"]:
                    query_points = batch["query_points"]
                    sdf = batch["sdf"].unsqueeze(-1)
                    sdf_min, sdf_max = sdf.min(), sdf.max()
                    sdf_norm = (sdf - sdf_min) / (sdf_max - sdf_min)
                    sdf_colors = sdf_norm * red.to(sdf) + (1 - sdf_norm) * blue.to(sdf)
                    return torch.cat((query_points, sdf_colors), dim=-1)
                elif log_type in ["pred_occupancy_sample_points"]:
                    query_points = batch["query_points"]
                    logits = batch["logits"].unsqueeze(-1)
                    logit_colors = logits * red.to(logits) + (1 - logits) * blue.to(
                        logits
                    )
                    return torch.cat((query_points, logit_colors), dim=-1)
                else:
                    raise NotImplementedError(
                        f"Logging for {log_type} is not implemented."
                    )

            # Process meshes
            for var_type in ["mesh", "image"]:
                if var_type in self.tracked_variables:
                    for item in self.tracked_variables[var_type]:
                        meshes = process_meshes(item, batch, render_batch)
                        if var_type == "mesh":
                            batch[item] = meshes
                        elif var_type == "image":
                            batch[item] = render_meshes(meshes)

        # Remove large SDFs to free memory
        del batch["reconstructed"]

        return batch

    ###################################################################
    ######################        PROFILING       #####################
    ###################################################################

    def _run_profiling(self, model, data_loader) -> None:
        """
        Profile the training process.

        :param model: Neural Module.
        :param batch: A dict containing the data to be fed to the model.
        """
        if (
            self.profiling
            and self.profiling.get("activated", False)
            and self.logger is not None
        ):
            # Synchronize all devices before starting profiling
            print("wait for everyone in _run_profiling")
            self.accelerator.wait_for_everyone()
            print("done waiting in _run_profiling")

            # Get data
            batch = next(iter(data_loader))

            # Starts the profiling session.
            with self.accelerator.profile() as prof:
                model(batch)

            # Print profiler summary
            self.log.warning(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            )
