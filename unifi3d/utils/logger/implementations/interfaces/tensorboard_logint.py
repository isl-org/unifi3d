from argparse import Namespace
from dataclasses import asdict, is_dataclass
from torch.utils.tensorboard import SummaryWriter
from .base_logint import BaseLogint
import numpy as np
import torch
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union


class TensorBoardLogint(BaseLogint):
    """
    Abstraction layer for TensorBoard logger.
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
            self.writer = SummaryWriter(log_dir=logint_config["log_dir"])
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
        self.writer.add_scalar(
            tag=tag,  # (str): Data identifier
            scalar_value=value,  # (float or string/blobname): Value to save
            global_step=epoch,  # (int): Global step value to record
            new_style=False,  # (boolean): Whether to use (tensor field) or old (simple_value field)
            double_precision=False,  # (boolean): Whether to use (folat) or (double)
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
        self.writer.add_scalars(
            main_tag=tag,  # (str): The parent name for the tags
            tag_scalar_dict=values_dict,  # (dict): Tag and corresponding values
            global_step=epoch,  # (int): Global step value to record
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
        self.writer.add_tensor(
            tag=tag,  # (str): Data identifier
            tensor=tensor,  # (torch.Tensor): tensor to save
            global_step=epoch,  # (int): Global step value to record
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
            if isinstance(v, torch.Tensor):
                self.writer.add_tensor(
                    tag=k,  # (str): Data identifier
                    tensor=v,  # (torch.Tensor): tensor to save
                    global_step=epoch,  # (int): Global step value to record
                )

            elif isinstance(v, dict):
                self.writer.add_scalars(
                    main_tag=k,  # (str): The parent name for the tags
                    tag_scalar_dict=v,  # (dict): Tag and corresponding values
                    global_step=epoch,  # (int): Global step value to record
                )
            else:
                try:
                    self.writer.add_scalar(
                        tag=k,  # (str): Data identifier
                        scalar_value=v,  # (float or string/blobname): Value to save
                        global_step=epoch,  # (int): Global step value to record
                    )
                except Exception as ex:
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
        self.writer.add_image(
            tag=tag,  # (str): Data identifier
            img_tensor=image,  # (torch.Tensor, numpy.ndarray, or string/blobname): Image data.
            dataformats="CHW",  # (str): Image data format specification of the form CHW, HWC, HW, WH, etc.
            global_step=epoch,  # (int): Global step value to record
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
        self.writer.add_images(
            tag=tag,  # (str): Data identifier
            img_tensor=images,  # (torch.Tensor, numpy.ndarray, or string/blobname): Image data.
            dataformats="CHW",  # (str): Image data format specification of the form CHW, HWC, HW, WH, etc.
            global_step=epoch,  # (int): Global step value to record.
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
        self.writer.add_histogram(
            tag=tag,  # (str): Data identifier
            values=hist,  # (torch.Tensor, numpy.ndarray, or string/blobname): Values to build histogram
            global_step=epoch,  # (int): Global step value to record.
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
        self.writer.add_mesh(
            tag=tag,  # (str): Data identifier
            vertices=vertices,  # (torch.Tensor): List of the 3D coordinates of vertices.
            faces=faces,  # (torch.Tensor): Indices of vertices within each triangle.
            colors=colors,  # (torch.Tensor): Colors for each vertex.
            config_dict=config_dict,  # Dictionary with ThreeJS classes names and configuration.
            global_step=epoch,  # (int): Global step value to record.
        )

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
        self.writer.add_video(
            tag=tag,  # (str): Data identifier
            vid_tensor=video,  # (torch.Tensor): Video data
            fps=fps,  # (int): Frames per second
            global_step=epoch,  # (int): Global step value to record.
        )

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: a dictionary-like container with the hyperparameters.
        :param metrics: Dictionary with metric names as keys and measured quantities as values.
        """
        """Record hyperparameters. TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the previously saved logs to
        display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values

        """
        # Format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        self.writer.add_hparams(hparam_dict=params, metric_dict=metrics)

    def finalize(self) -> None:
        """
        Free resources.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _flatten_dict(self, params, delimiter: str = "/", parent_key: str = "") -> dict:
        """
        Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

        Args:
            params: Dictionary containing the hyperparameters
            delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

        Returns:
            Flattened dict.

        Examples:
            >>> _flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> _flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
            >>> _flatten_dict({5: {'a': 123}})
            {'5/a': 123}

        """
        result = {}
        for k, v in params.items():
            new_key = parent_key + delimiter + str(k) if parent_key else str(k)
            if is_dataclass(v):
                v = asdict(v)
            elif isinstance(v, Namespace):
                v = vars(v)

            if isinstance(v, MutableMapping):
                result = {
                    **result,
                    **self._flatten_dict(v, parent_key=new_key, delimiter=delimiter),
                }
            else:
                result[new_key] = v
        return result

    def _sanitize_params(self, params):
        """
        Returns params with non-primitvies converted to strings for logging.

        >>> import torch
        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
            'float': 0.3,
            'int': 1,
            'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
            'list': '[1, 2, 3]',
            'namespace': 'Namespace(foo=3)',
            'string': 'abc'}

        """
        for k in params:
            # convert relevant np scalars to python types first (instead of str)
            if isinstance(params[k], (np.bool_, np.integer, np.floating)):
                params[k] = params[k].item()
            elif type(params[k]) not in [bool, int, float, str, torch.Tensor]:
                params[k] = str(params[k])
        return params
