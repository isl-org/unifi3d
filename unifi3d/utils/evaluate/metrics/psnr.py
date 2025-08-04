from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchvision import transforms

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import EXT_PNG


class PSNR(Base3dMetric):
    """
    Computes the average peak signal to noise ratio between the image prompt
    and the object renders.
    """

    def _initialize(self) -> None:
        # Load PSNR function
        self.loss_fn = peak_signal_noise_ratio
        self.transform = transforms.PILToTensor()

    def __call__(self, image_prompt, renders_path: str):
        """
        Args:
            Image prompt: PIL image
            renders_path: Path to object renders folder
        Returns:
            Float with the PSNR of the object given the prompt
        """
        prompt = self.transform(image_prompt.convert("RGB"))

        # select all the .png files from the given root directory
        # files are sorted in alphabetical order
        imagefiles = sorted(
            filter(
                # predicate: has ".png" extension (case-insensitive)
                lambda filepath: filepath.name.lower().endswith(EXT_PNG),
                # get files from root directory
                Path(renders_path).iterdir(),
            )
        )

        # In sequence over images, compute cosine similarities between embeddings
        per_image_metrics = []
        for imf in imagefiles:
            render = Image.open(imf).convert("RGB")
            image = self.transform(render)
            distance = self.loss_fn(image, prompt)
            per_image_metrics.append(float(distance.numpy()))
        per_image_metrics = np.asarray(per_image_metrics)

        return np.mean(per_image_metrics[np.isfinite(per_image_metrics)])
