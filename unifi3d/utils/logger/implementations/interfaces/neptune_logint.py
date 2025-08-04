import neptune.new as neptune
from .base_logint import BaseLogint
import numpy as np
import torch
from typing import Any, Dict, MutableMapping, Optional, Union


class NeptuneLogint(BaseLogint):
    """
    Abstraction layer for Neptune logger.
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
            self.run = neptune.init(
                api_token=logint_config["api_key"],
                project=logint_config["project_name"],
                name=logint_config["run_name"],
                tags=logint_config["tags"],
            )
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
        self.run[tag].log(value, step=epoch)

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
            self.run[f"{tag}/{k}"].log(v, step=epoch)

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
        self.run[tag].log(neptune.types.File.as_html(tensor.numpy()), step=epoch)

    def log_metrics(
        self,
        metrics,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        A more general function to add metric data to summary.

        :param metrics: Dictionary of metric data to save.
        :param epoch: Global step value to record.
        """
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    self.run[f"{k}/{sub_k}"].log(sub_v, step=epoch)
            else:
                self.run[k].log(v, step=epoch)

    def log_image(
        self,
        tag: str,
        image,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add image data to summary.

        :param tag: Data identifier.
        :param image: Image data.
        :param epoch: Global step value to record.
        """
        self.run[tag].log(neptune.types.File.as_image(image), step=epoch)

    def log_images(
        self,
        tag: str,
        images,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add batched image data to summary.

        :param tag: Data identifier.
        :param images: Image data.
        :param epoch: Global step value to record.
        """
        for idx, img in enumerate(images):
            self.run[f"{tag}/{idx}"].log(neptune.types.File.as_image(img), step=epoch)

    def log_histogram(
        self,
        tag: str,
        hist,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add histogram data to summary.

        :param tag: Data identifier.
        :param hist: Values to build histogram.
        :param epoch: Global step value to record.
        """
        self.run[tag].log(neptune.types.File.as_html(hist), step=epoch)

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
        mesh_data = {
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
            "config": config_dict,
        }
        self.run[tag].log(neptune.types.File.as_html(str(mesh_data)), step=epoch)

    def log_video(
        self,
        tag: str,
        video,
        epoch: int = 0,
        fps: int = 4,
        context: dict = {},
    ) -> None:
        """
        Add video data to summary.

        :param tag: Data identifier.
        :param video: Video data.
        :param epoch: Global step value to record.
        :param fps: Frames per second.
        """
        # Convert videos to Neptune's supported format
        video_temp_path = "temp_video.mp4"
        import cv2

        height, width, _ = video[0].shape
        writer = cv2.VideoWriter(
            video_temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        for frame in video:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        self.run[tag].log(neptune.types.File(video_temp_path), step=epoch)

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: a dictionary-like container with the hyperparameters.
        :param metrics: Dictionary with metric names as keys and measured quantities as values.
        """
        self.run["hyperparameters"] = params

        if metrics:
            self.log_metrics(metrics, 0)

    def finalize(self) -> None:
        """
        Free resources.
        """
        if self.run is not None:
            self.run.stop()
