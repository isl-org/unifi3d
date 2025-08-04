import wandb
from .base_logint import BaseLogint
import numpy as np
import torch
from typing import Any, Dict, MutableMapping, Optional, Union


class WandbLogint(BaseLogint):
    """
    Abstraction layer for Weights and Biases (wandb) logger.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        A logint should be a singleton
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, logint_config: dict) -> None:
        if not hasattr(self, "initialized"):  # To prevent reinitialization
            self.run = wandb.init(project=logint_config["name"], config=logint_config)
            self.initialized = True

    def log_scalar(
        self,
        tag: str,
        value,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add scalar data to summary.

        :param tag: Data identifier.
        :param value: Value to save.
        :param epoch: Global step value to record.
        """
        self.run.log({tag: value}, step=epoch)

    def log_scalars(
        self,
        tag: str,
        values_dict,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add many scalar data to summary.

        :param tag: The parent name for the tags.
        :param values_dict: Key-value pair storing the tag and corresponding values.
        :param epoch: Global step value to record.
        """
        for k, v in values_dict.items():
            self.run.log({f"{tag}/{k}": v}, step=epoch)

    def log_tensor(
        self,
        tag: str,
        tensor,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add tensor data to summary.

        :param tag: The parent name for the tags.
        :param tensor: tensor to save.
        :param epoch: Global step value to record.
        """
        self.run.log({tag: wandb.Histogram(tensor)}, step=epoch)

    def log_metrics(self, metrics, epoch: int = 0, context: dict = {}) -> None:
        """
        A more general function to add metric data to summary.

        :param metrics: Dictionary of metric data to save.
        :param epoch: Global step value to record.
        """
        for k, v in metrics.items():
            self.run.log({k: v}, step=epoch)

    def log_image(self, tag: str, image, epoch: int = 0, context: dict = {}) -> None:
        """
        Add image data to summary.

        :param tag: Data identifier.
        :param image: Image data.
        :param epoch: Global step value to record.
        """
        self.run.log({tag: wandb.Image(image)}, step=epoch)

    def log_images(self, tag: str, images, epoch: int = 0, context: dict = {}) -> None:
        """
        Add batched image data to summary.

        :param tag: Data identifier.
        :param images: Image data.
        :param epoch: Global step value to record.
        """
        for idx, img in enumerate(images):
            self.run.log({f"{tag}/{idx}": wandb.Image(img)}, step=epoch)

    def log_histogram(self, tag: str, hist, epoch: int = 0, context: dict = {}) -> None:
        """
        Add histogram data to summary.

        :param tag: Data identifier.
        :param hist: Values to build histogram.
        :param epoch: Global step value to record.
        """
        if hist < 1e6:
            self.run.log({tag: wandb.Histogram(hist)}, step=epoch)
        else:
            print(f"Skipping logging due to value for {tag} > 1e6")

    def log_mesh(
        self,
        tag: str,
        vertices,
        faces=None,
        colors=None,
        epoch: int = 0,
        config_dict: dict = {},
        context: dict = {},
    ) -> None:
        """
        Add meshes or 3D point clouds to summary.

        :param tag: Data identifier.
        :param vertices: List of the 3D coordinates of vertices.
        :param faces: Indices of vertices within each triangle.
        :param colors: Colors for each vertex.
        :param epoch: Global step value to record.
        :param config_dict: Dictionary with ThreeJS classes names and configuration.
        """
        if not isinstance(vertices, np.ndarray):
            vertices = vertices.cpu().numpy()
        if colors is not None and not isinstance(colors, np.ndarray):
            colors = colors.cpu().numpy()

        if colors is None:
            pcd = vertices
        else:
            pcd = np.concatenate((vertices, colors), axis=-1)
        pcd = np.squeeze(pcd)

        self.run.log(
            {tag: wandb.Object3D(pcd)},
            step=epoch,
        )

    def log_video(
        self, tag: str, video, epoch: int = 0, fps: int = 4, context: dict = {}
    ) -> None:
        """
        Add video data to summary.

        :param tag: Data identifier.
        :param video: Video data.
        :param epoch: Global step value to record.
        :param fps: Frames per second.
        """
        video_data = (
            [wandb.Video(v, fps=fps) for v in video]
            if len(video.shape) == 5
            else wandb.Video(video, fps=fps)
        )
        self.run.log({tag: video_data}, step=epoch)

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: a dictionary-like container with the hyperparameters.
        :param metrics: Dictionary with metric names as keys and measured quantities as values.
        """
        self.run.config.update(params)

        if metrics:
            self.log_metrics(metrics, 0)

    def finalize(self) -> None:
        """
        Free resources.
        """
        if self.run is not None:
            self.run.finish()
