from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

from unifi3d.utils.evaluate.metrics.utils import EXT_PNG
from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric


class CLIPSimilarityText(Base3dMetric):
    """
    Computes the average cosine similarity between the CLIP embeddings of the
    text prompt and renders of an object.
    """

    def _initialize(self, clip_model: str = "vit-l-14", device: str = "cuda") -> None:
        """
        Args:
            clip_model: The version of the CLIP model used to compute embeddings
            device: The device on which to place the model, cuda or cpu
        """
        self.clip_model = clip_model
        self.device = device

        if clip_model == "vit-l-14":
            self.model = SentenceTransformer("clip-ViT-L-14")
        else:
            raise NotImplementedError

        self.model.to(device)
        self.loss_fn = util.cos_sim

    def __call__(self, text_prompt: str, renders_path: str):
        """
        Args:
            Text prompt: A string text prompt
            renders_path: Path to object renders
        Returns:
            Float with the CLIP score of the object given the prompt
        """
        # Encode text prompt
        text_emb = self.model.encode([text_prompt])

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
            img_emb = self.model.encode(render)
            cos_scores = self.loss_fn(text_emb, img_emb)
            per_image_metrics.append(float(cos_scores[0][0].cpu().numpy()))

        return np.mean(np.asarray(per_image_metrics))
