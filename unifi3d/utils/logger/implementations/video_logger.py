import imageio
import numpy as np
import os
from PIL import Image
import torch

from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint


class VideoLogger(BaseLogger):
    """
    VideoLogger logs videos at desired epochs.
    """

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "video"
        self.iter_counter = 0

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)

    def log_iter(self, batch) -> None:
        """
        Logs video data from a batch to TensorBoard and as .gif files.

        :param batch: A dictionary containing data and metadata for the batch to log.
        """
        if not self.NAME in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if len(keys_list) == 0:
            return

        # Update logger state
        self.phase = batch["phase"]
        self.epoch = batch["epoch"]

        # Create an epoch-specific directory for the log files
        epoch_dir = os.path.join(self.log_path, f"epoch_{self.epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for video_key in keys_list:
            if video_key not in batch:
                continue

            kdata = batch[video_key]

            # Ensure the right shape for batch video data (batch_size, channels, frames, height, width)
            assert len(kdata.shape) == 5, "Video data must have 5 dimensions"

            # Detach from GPU and convert to numpy if it's a torch Tensor
            nbatch = kdata.shape[0]
            if isinstance(kdata, torch.Tensor):
                kdata = kdata.detach().cpu().numpy()

            # Process each video in the batch
            for batch_id, video in enumerate(kdata):
                # Convert grayscale videos (1 channel) to RGB
                if video.shape[1] == 1:
                    video = np.concatenate([video] * 3, axis=1)

                # Rescale and clip video pixel values
                video = video * (255.0 / video.max()) if video.max() < 200 else video
                video = np.clip(video, 0, 255).astype(np.uint8)

                # Add video to BaseLogint
                self.logger_interface.log_video(
                    tag=f"{video_key}/{batch['phase']}",
                    video=torch.from_numpy(video).unsqueeze(0) / 255.0,
                    epoch=self.epoch,
                    context={"subset": self.phase},
                )

                # Save video frames as .gif files
                frames = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in video]
                filename = os.path.join(
                    epoch_dir, f"{self.iter_counter}_{video_key}_{batch_id}.gif"
                )
                imageio.mimsave(filename, frames, fps=10)

        self.iter_counter += 1

    def log_epoch(self) -> None:
        """
        Placeholder for epoch-logging logic, should it be needed in the future.
        """
        self.iter_counter = 0

    def print_iter_log(self) -> None:
        """
        Prints out relevant logged quantity for the current iteration.
        """
        pass

    def print_epoch_log(self) -> None:
        """
        Prints out relevant logged quantity for the current epoch.
        """
        pass
