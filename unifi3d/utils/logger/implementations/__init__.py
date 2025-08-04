from .video_logger import VideoLogger
from .metric_logger import MetricLogger
from .image_logger import ImageLogger
from .mesh_logger import MeshLogger
from .histogram_logger import HistogramLogger
from .hyperparameter_logger import HyperparameterLogger


REGISTED_LOGGER = {
    "metric": MetricLogger,
    "image": ImageLogger,
    "mesh": MeshLogger,
    "video": VideoLogger,
    "hist": HistogramLogger,
    "hparams": HyperparameterLogger,
}
