from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class Camera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def image_coords(self) -> np.ndarray:
        """
        :return: ([self.height, self.width, 2]).reshape(self.height * self.width, 2) image coordinates
        """

    @abstractmethod
    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        """
        For every (x, y) coordinate in a rendered image, get the direction that
        corresponds to "depth" in an RGBD rendering.

        This may raise an exception if there is no "D" channel in the
        corresponding ViewData.

        :param coords: an [N x 2] integer array of 2D image coordinates.
        :return: an [N x 3] array of normalized depth directions.
        """
        _ = coords
        raise NotImplementedError

    @abstractmethod
    def center_crop(self) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with a center crop to a square of the smaller dimension.
        """

    @abstractmethod
    def resize_image(self, width: int, height: int) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with resized image dimensions.
        """

    @abstractmethod
    def scale_scene(self, factor: float) -> "Camera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the scene rescaled by the given factor.
        """


@dataclass
class ProjectiveCamera(Camera):
    """
    A Camera implementation for a standard pinhole camera.

    The camera rays shoot away from the origin in the z direction, with the x
    and y directions corresponding to the positive horizontal and vertical axes
    in image space.
    """

    origin: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    width: int
    height: int
    x_fov: float
    y_fov: float

    def image_coords(self) -> np.ndarray:
        ind = np.arange(self.width * self.height)
        coords = np.stack([ind % self.width, ind // self.width], axis=1).astype(
            np.float32
        )
        return coords

    def camera_rays(self, coords: np.ndarray) -> np.ndarray:
        fracs = (
            coords / (np.array([self.width, self.height], dtype=np.float32) - 1)
        ) * 2 - 1
        fracs = fracs * np.tan(np.array([self.x_fov, self.y_fov]) / 2)
        directions = self.z + self.x * fracs[:, :1] + self.y * fracs[:, 1:]
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        return np.stack(
            [np.broadcast_to(self.origin, directions.shape), directions], axis=1
        )

    def depth_directions(self, coords: np.ndarray) -> np.ndarray:
        return np.tile((self.z / np.linalg.norm(self.z))[None], [len(coords), 1])

    def resize_image(self, width: int, height: int) -> "ProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert (
            width * self.height == height * self.width
        ), "The aspect ratio should not change."
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )

    def center_crop(self) -> "ProjectiveCamera":
        """
        Creates a new camera for the center-cropped view
        """
        size = min(self.width, self.height)
        fov = min(self.x_fov, self.y_fov)
        return ProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=size,
            height=size,
            x_fov=fov,
            y_fov=fov,
        )

    def scale_scene(self, factor: float) -> "ProjectiveCamera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with the camera frame rescaled by the given factor.
        """
        return ProjectiveCamera(
            origin=self.origin * factor,
            x=self.x,
            y=self.y,
            z=self.z,
            width=self.width,
            height=self.height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


@dataclass
class DifferentiableCamera(ABC):
    """
    An object describing how a camera corresponds to pixels in an image.
    """

    @abstractmethod
    def camera_rays(self, coords: torch.Tensor) -> torch.Tensor:
        """
        For every (x, y) coordinate in a rendered image, compute the ray of the
        corresponding pixel.

        :param coords: an [N x ... x 2] integer array of 2D image coordinates.
        :return: an [N x ... x 2 x 3] array of [2 x 3] (origin, direction) tuples.
                 The direction should always be unit length.
        """

    @abstractmethod
    def resize_image(self, width: int, height: int) -> "DifferentiableCamera":
        """
        Creates a new camera with the same intrinsics and direction as this one,
        but with resized image dimensions.
        """


@dataclass
class DifferentiableProjectiveCamera(DifferentiableCamera):
    """
    Implements a batch, differentiable, standard pinhole camera
    """

    origin: torch.Tensor  # [batch_size x 3]
    x: torch.Tensor  # [batch_size x 3]
    y: torch.Tensor  # [batch_size x 3]
    z: torch.Tensor  # [batch_size x 3]
    width: int
    height: int
    x_fov: float
    y_fov: float

    def __post_init__(self):
        assert (
            self.x.shape[0]
            == self.y.shape[0]
            == self.z.shape[0]
            == self.origin.shape[0]
        )
        assert (
            self.x.shape[1]
            == self.y.shape[1]
            == self.z.shape[1]
            == self.origin.shape[1]
            == 3
        )
        assert (
            len(self.x.shape)
            == len(self.y.shape)
            == len(self.z.shape)
            == len(self.origin.shape)
            == 2
        )

    def resolution(self):
        return torch.from_numpy(np.array([self.width, self.height], dtype=np.float32))

    def fov(self):
        return torch.from_numpy(np.array([self.x_fov, self.y_fov], dtype=np.float32))

    def image_coords(self) -> torch.Tensor:
        """
        :return: coords of shape (width * height, 2)
        """
        pixel_indices = torch.arange(self.height * self.width)
        coords = torch.stack(
            [
                pixel_indices % self.width,
                torch.div(pixel_indices, self.width, rounding_mode="trunc"),
            ],
            axis=1,
        )
        return coords

    def camera_rays(self, coords: torch.Tensor) -> torch.Tensor:
        batch_size, *shape, n_coords = coords.shape
        assert n_coords == 2
        assert batch_size == self.origin.shape[0]
        flat = coords.view(batch_size, -1, 2)

        res = self.resolution().to(flat.device)
        fov = self.fov().to(flat.device)

        fracs = (flat.float() / (res - 1)) * 2 - 1
        fracs = fracs * torch.tan(fov / 2)

        fracs = fracs.view(batch_size, -1, 2)
        directions = (
            self.z.view(batch_size, 1, 3)
            + self.x.view(batch_size, 1, 3) * fracs[:, :, :1]
            + self.y.view(batch_size, 1, 3) * fracs[:, :, 1:]
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        rays = torch.stack(
            [
                torch.broadcast_to(
                    self.origin.view(batch_size, 1, 3),
                    [batch_size, directions.shape[1], 3],
                ),
                directions,
            ],
            dim=2,
        )
        return rays.view(batch_size, *shape, 2, 3)

    def resize_image(self, width: int, height: int) -> "DifferentiableProjectiveCamera":
        """
        Creates a new camera for the resized view assuming the aspect ratio does not change.
        """
        assert (
            width * self.height == height * self.width
        ), "The aspect ratio should not change."
        return DifferentiableProjectiveCamera(
            origin=self.origin,
            x=self.x,
            y=self.y,
            z=self.z,
            width=width,
            height=height,
            x_fov=self.x_fov,
            y_fov=self.y_fov,
        )


@dataclass
class DifferentiableCameraBatch(ABC):
    """
    Annotate a differentiable camera with a multi-dimensional batch shape.
    """

    shape: Tuple[int]
    flat_camera: DifferentiableCamera


def sample_data_cameras(
    batch_camera, output_dim, views_per_obj, device: torch.device
) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    num_cameras = len(batch_camera)
    sampled_cameras = np.random.choice(num_cameras, size=views_per_obj)
    for j in sampled_cameras:
        origin = batch_camera[j].origin
        x = batch_camera[j].x
        y = batch_camera[j].y
        z = batch_camera[j].z
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return sampled_cameras, DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=output_dim,
            height=output_dim,
            x_fov=batch_camera[0].x_fov,
            y_fov=batch_camera[0].y_fov,
        ),
    )


@torch.no_grad()
def get_image_coords(width, height) -> torch.Tensor:
    pixel_indices = torch.arange(height * width)
    # torch throws warnings for pixel_indices // width
    pixel_indices_div = torch.div(pixel_indices, width, rounding_mode="trunc")
    coords = torch.stack([pixel_indices % width, pixel_indices_div], dim=1)
    return coords
