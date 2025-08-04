from abc import ABC, abstractmethod
from typing import Any


class NamedObject(ABC):
    """
    An object that is named by its classname and the arguments it received as input.
    This is used for giving dynamically loaded objects human-readable string names.
    We use this, e.g., upon evaluation to have a human-readable column name with the metric and its parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

        self._initialize(*args, **kwargs)

    @abstractmethod
    def _initialize(self, *args, **kwargs) -> None: ...

    @property
    def name(self):
        args_repr = "__".join(map(str, self.args))
        kwargs_repr = "__".join(
            map(lambda kwarg: f"{kwarg[0]}={kwarg[1]}", self.kwargs.items())
        )
        return f"{self.__class__.__name__}--args--{args_repr}--kwargs--{kwargs_repr}"

    def __str__(self) -> str:
        return self.name


class Base3dMetric(NamedObject):
    """
    Abstract base class for metric over 3D assets. This is to enforce a unified
    metric definition for evaluation

    This class is intended to be subclassed by specific metric implementations
    that operates on 3D assets via the `__call__` method. The arguments should
    match the expected output of the dataset being acted upon.
    """

    def _initialize(self, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
