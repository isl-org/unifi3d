import cv2
from datetime import datetime
import imageio
import matplotlib
from matplotlib import cm
import numpy as np
import os
import torch

from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint

# Force matplotlib to not use any Xwindows backend
matplotlib.use("Agg")


class ImageLogger(BaseLogger):
    """
    ImageLogger logs images.
    """

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "image"
        self.iter_counter = 0

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)

    def log_iter(self, batch) -> None:
        """
        Logs image data from the batch to BaseLogint and saves it to files.

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        if not self.NAME in batch["output_parser"].keys():
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:
            return

        # Update logger state
        self.phase = batch["phase"]
        self.epoch = batch["epoch"]

        epoch_dir = os.path.join(self.log_path, f"epoch_{self.epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for img_key in keys_list:
            if img_key not in batch.keys():
                continue

            kdata = self._prepare_data(batch[img_key])
            nbatch = kdata[0].shape[0]  # Get the batch size from the first view

            for batch_id in range(nbatch):
                for view_id, img in enumerate(kdata):
                    self._process_and_log_image(
                        img[batch_id], img_key, epoch_dir, batch_id
                    )

        self.iter_counter += 1

    def _prepare_data(self, kdata):
        """
        Prepares image data for logging by ensuring it is in list form and detaches any Torch Tensors.

        :param kdata: Raw image data from the batch.
        :return: List of NumPy arrays containing image data.
        """
        if isinstance(kdata, list):
            assert (
                kdata and len(kdata[0].shape) == 4
            ), f"Invalid shape for images: {kdata[0].shape}"
        else:
            assert len(kdata.shape) == 4, f"Invalid shape for images: {kdata.shape}"
            if isinstance(kdata, torch.Tensor):
                kdata = [kdata.detach().cpu().numpy()]
            else:
                kdata = [kdata]

        # Convert to ndarray
        if isinstance(kdata[0], torch.Tensor):
            for i, tensor in enumerate(kdata):
                kdata[i] = tensor.detach().cpu().numpy()

        return kdata

    def _process_and_log_image(self, img, img_key, epoch_dir, batch_id):
        """
        Processes an individual image, logs it to the logger interface, and saves it to a file.

        :param img: The image tensor or array to process.
        :param img_key: Key name associated with the image.
        :param epoch_dir: Directory to save the image.
        :param batch_id: Batch index of the image.
        """
        assert img.ndim == 3, f"Expected image with 3 dimensions, got {img.ndim}"
        color_flag = img.shape[0] == 1  # Grayscale images
        img = self._process_image(img, color_flag)

        # Log image to the interface
        self.logger_interface.log_image(
            tag=f"{img_key}/{self.phase}",
            image=img if color_flag else img[[0, 1, 2], ...],
            epoch=self.epoch,
            context={"subset": self.phase},
        )

        # Export .png files
        img = img.transpose(1, 2, 0)  # Convert to H*W*C
        if color_flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.join(
            epoch_dir, f"{self.iter_counter}_{img_key}_{batch_id}.png"
        )
        imageio.imsave(filename, img)

    def _process_image(self, img, color_flag):
        """
        Applies color mapping and normalization to the image.

        :param img: Raw image data (3D tensor or array).
        :param color_flag: Flag to indicate if the image is grayscale.
        :return: Processed image as a NumPy array.
        """
        if color_flag:
            # Apply colormap to grayscale image
            img = cm.get_cmap("magma")(img.squeeze(0))[
                ..., :3
            ]  # Apply colormap and take RGB
            img = img.transpose(2, 0, 1)  # Convert to C*H*W format
            img *= 255.0
        else:
            img *= 255.0 if img.max() < 200 else 1

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def log_epoch(self) -> None:
        """
        Placeholder for phase logging logic.
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
