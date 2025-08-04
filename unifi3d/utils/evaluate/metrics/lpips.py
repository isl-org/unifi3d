from pathlib import Path

import lpips
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import EXT_PNG


class LPIPS(Base3dMetric):
    """
    Computes the average LPIPS distance between the image prompt
    and the object renders.
    """

    def _initialize(self, device: str = "cuda") -> None:
        """
        Args:
            device: The device on which to place the model, cuda or cpu
        """
        self.device = device

        # Load LPIPS model
        self.loss_fn = lpips.LPIPS(net="alex").to(device)
        self.transform = transforms.PILToTensor()

    def __call__(self, image_prompt, renders_path: str):
        """
        Args:
            Image prompt: PIL image
            renders_path: Path to object renders folder
        Returns:
            Float with the LPIPS of the object given the prompt

        Note: LPIPS requires the inputs to be scaled to [-1, 1]
        """
        prompt = self.transform(image_prompt.convert("RGB"))
        prompt = torch.unsqueeze(prompt / 127.5 - 1, 0).to(self.device)

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
            image = torch.unsqueeze(self.transform(render) / 127.5 - 1, 0).to(
                self.device
            )
            distance = self.loss_fn(image, prompt)
            per_image_metrics.append(float(distance.cpu().detach().numpy()))

        return np.mean(np.asarray(per_image_metrics))
