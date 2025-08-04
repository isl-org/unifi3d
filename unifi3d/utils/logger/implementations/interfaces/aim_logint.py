from aim import Run, Distribution, Image
from .base_logint import BaseLogint
import torch
import numpy as np


class AimLogint(BaseLogint):
    """
    Abstraction layer for AIM logger.
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
            self.run = Run(
                experiment=logint_config["experiment"],
                repo=logint_config["repo"],
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
        self.run.track(
            value=value,  # (float or string/blobname): Value to save
            name=tag,  # (str): Data identifier
            epoch=epoch,  # (int): Global step value to record
            context=context,  # (dict) Sequence tracking context.
        )

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
        :param value_dict: Key-value pair storing the tag and corresponding values.
        :param epoch: Global step value to record.
        """
        self.run.track(
            value=values_dict,  # (float or string/blobname): Value to save
            name=None,  # (str): Data identifier
            epoch=epoch,  # (int): Global step value to record
            context=context,  # (dict) Sequence tracking context.
        )

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
        self.run.track(
            value=tensor,  # (float or string/blobname): Value to save
            name=tag,  # (str): Data identifier
            epoch=epoch,  # (int): Global step value to record
            context=context,  # (dict) Sequence tracking context.
        )

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
                self.run.track(
                    value=v,  # (float or string/blobname): Value to save
                    name=None,  # (str): Data identifier
                    epoch=epoch,  # (int): Global step value to record
                    context=context,  # (dict) Sequence tracking context.
                )
            else:
                try:
                    self.run.track(
                        name=k,  # (str): Data identifier
                        value=v,  # (float or string/blobname): Value to save
                        epoch=epoch,  # (int): Global step value to record
                        context=context,  # (dict) Sequence tracking context.
                    )
                except Exception as ex:
                    print(f"Exception: {ex}")
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    ) from ex

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
        :param images: Image data.
        :param epoch: Global step value to record.
        """
        if image.shape[0] <= 3:
            image = image.transpose(1, 2, 0)
        context["type"] = "image"
        self.run.track(
            value=Image(image),  # (float or string/blobname): Value to save
            name=tag,  # (str): Data identifier
            epoch=epoch,  # (int): Global step value to record
            context=context,  # (dict) Sequence tracking context.
        )

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
        context["type"] = "image"
        for idx, img_array in enumerate(images):
            context["index"] = idx
            self.run.track(
                value=Image(img_array),  # (float or string/blobname): Value to save
                name=tag + "/" + str(idx),  # (str): Data identifier
                epoch=epoch,  # (int): Global step value to record
                context=context,  # (dict) Sequence tracking context.
            )

    def log_histogram(
        self,
        tag: str,
        hist,
        epoch: int = 0,
        context: dict = {},
    ) -> None:
        """
        Add batched image data to summary.

        :param tag: Data identifier.
        :param hist: Values to build histogram.
        :param epoch: Global step value to record.
        """
        context["type"] = "histogram"
        dtype = hist.dtype
        if dtype == torch.float32:
            min_val, max_val = np.finfo(np.float32).min, np.finfo(np.float32).max
        elif dtype == torch.float64:
            min_val, max_val = np.finfo(np.float64).min, np.finfo(np.float64).max
        elif dtype == torch.int32:
            min_val, max_val = np.finfo(np.int32).min, np.finfo(np.int32).max
        elif dtype == torch.int64:
            min_val, max_val = np.finfo(np.int64).min, np.finfo(np.in64).max
        else:
            min_val = None
            max_val = None
        if min_val is not None:
            hist = torch.clamp(hist, min=min_val, max=max_val)
        try:
            value = Distribution(hist)
        except:
            value = Distribution(torch.zeros_like(hist))
        self.run.track(
            value=value,  # (float or string/blobname): Value to save
            name=tag,  # (str): Data identifier
            epoch=epoch,  # (int): Global step value to record
            context=context,  # (dict) Sequence tracking context.
        )

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
            "colors": colors,
            "faces": faces,
            "config": config_dict,
        }
        # self.run.track(
        #     value=mesh_data,  # (float or string/blobname): Value to save
        #     name=tag,  # (str): Data identifier
        #     epoch=epoch,  # (int): Global step value to record
        #     context=context,  # (dict) Sequence tracking context.
        # )

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
        """
        aim_images = []
        context["type"] = "video"
        context["fps"] = fps
        if len(video.shape) == 5:  # Batch of videos
            for seq_idx in range(video.shape[0]):
                for frame_idx in range(video.shape[-1]):
                    frame = video[seq_idx, ..., frame_idx]

                context["sequence"] = seq_idx
                self.run.track(
                    aim_images,  # (float or string/blobname): Value to save
                    name=f"{tag}/{seq_idx}",  # (str): Data identifier
                    epoch=epoch,  # (int): Global step value to record
                    context=context,  # (dict) Sequence tracking context.
                )
        elif len(video.shape) == 4:  # Single video
            for frame_idx in range(video.shape[-1]):
                frame = video[..., frame_idx]
                aim_images.append(Image(frame))
            self.run.track(
                aim_images,  # (float or string/blobname): Value to save
                name=tag,  # (str): Data identifier
                epoch=epoch,  # (int): Global step value to record
                context=context,  # (dict) Sequence tracking context.
            )

        else:
            return

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: a dictionary-like container with the hyperparameters.
        :param metrics: Dictionary with metric names as keys and measured quantities as values.
        """

        if metrics:
            self.log_metrics(metrics, 0)

            for key, value in params.items():
                self.run.set(("hparams", key), value, strict=False)

    def finalize(self) -> None:
        """
        Free resources.
        """
        if self.run is not None:
            self.run.close()
