from abc import ABC, abstractmethod


class BaseLogint(ABC):
    """
    Abstraction layer for logging interface.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def log_hyperparams(self, params, metrics=None) -> None:
        """
        Record hyperparameters.

        :param param: The hyperparameters.
        :param metrics: Any metric.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Free resources.
        """
        pass
