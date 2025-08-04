import mlflow
from mlflow import log_metric, log_param, log_artifact
from .base_logint import BaseLogint
import numpy as np
import torch
from typing import Any, Dict, MutableMapping, Optional, Union


class MLFlowLogint(BaseLogint):
    """
    Abstraction layer for MLflow logger.
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
            mlflow.set_experiment(logint_config["experiment"])
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
        log_metric(tag, value, step=epoch)

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
            log_metric(f"{tag}/{k}", v, step=epoch)

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
        npy_path = f"{tag}_tensor_epoch_{epoch}.npy"
        np.save(npy_path, tensor.detach().numpy())
        log_artifact(npy_path)

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
                    log_metric(f"{k}/{sub_k}", sub_v, step=epoch)
            else:
                log_metric(k, v, step=epoch)

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
        import matplotlib.pyplot as plt

        plt.imshow(image)
        img_path = f"{tag}_epoch_{epoch}.png"
        plt.savefig(img_path)
        log_artifact(img_path)

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
        import matplotlib.pyplot as plt

        for idx, img in enumerate(images):
            plt.imshow(img)
            img_path = f"{tag}_{idx}_epoch_{epoch}.png"
            plt.savefig(img_path)
            log_artifact(img_path)

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
        import matplotlib.pyplot as plt

        plt.hist(hist.cpu().numpy() if isinstance(hist, torch.Tensor) else hist)
        hist_path = f"{tag}_hist_epoch_{epoch}.png"
        plt.savefig(hist_path)
        log_artifact(hist_path)

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
            "vertices": (
                vertices.tolist() if isinstance(vertices, np.ndarray) else vertices
            ),
            "faces": (
                faces.tolist()
                if faces is not None and isinstance(faces, np.ndarray)
                else faces
            ),
            "colors": (
                colors.tolist()
                if colors is not None and isinstance(colors, np.ndarray)
                else colors
            ),
            "config": config_dict,
        }
        mesh_path = f"{tag}_mesh_epoch_{epoch}.json"
        import json

        with open(mesh_path, "w") as f:
            json.dump(mesh_data, f)
        log_artifact(mesh_path)

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
        video_temp_path = f"{tag}_video_epoch_{epoch}.mp4"
        import cv2

        height, width, _ = video[0].shape
        writer = cv2.VideoWriter(
            video_temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        for frame in video:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        log_artifact(video_temp_path)

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: a dictionary-like container with the hyperparameters.
        :param metrics: Dictionary with metric names as keys and measured quantities as values.
        """
        for k, v in params.items():
            log_param(k, v)

        if metrics:
            self.log_metrics(metrics, 0)

    def finalize(self) -> None:
        """
        Free resources.
        """
        if mlflow.active_run():
            mlflow.end_run()
