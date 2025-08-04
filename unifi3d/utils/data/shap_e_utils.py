from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import os
import platform
import random
import subprocess
import tempfile
import zipfile
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple, Union

import blobfile as bf
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from unifi3d.utils.model.shap_e_utils import AttrDict
from unifi3d.utils.rendering.camera import Camera, ProjectiveCamera

COLORS = frozenset(["R", "G", "B", "A"])
UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]
BASIC_AMBIENT_COLOR = 0.3
BASIC_DIFFUSE_COLOR = 0.7

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "shap_e_blender.py"
)


class ViewData(ABC):
    """
    A collection of rendered camera views of a scene or object.

    This is a generalization of a NeRF dataset, since NeRF datasets only encode
    RGB or RGBA data, whereas this dataset supports arbitrary channels.
    """

    @property
    @abstractmethod
    def num_views(self) -> int:
        """
        The number of rendered views.
        """

    @property
    @abstractmethod
    def channel_names(self) -> List[str]:
        """
        Get all of the supported channels available for the views.

        This can be arbitrary, but there are some standard names:
        "R", "G", "B", "A" (alpha), and "D" (depth).
        """

    @abstractmethod
    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        """
        Load the given channels from the view at the given index.

        :return: a tuple (camera_view, data), where data is a float array of
                 shape [height x width x num_channels].
        """


class BlenderViewData(ViewData):
    """
    Interact with a dataset zipfile exported by view_data.py.
    """

    def __init__(self, f_obj: BinaryIO):
        self.zipfile = zipfile.ZipFile(f_obj, mode="r")
        self.infos = []
        with self.zipfile.open("info.json", "r") as f:
            self.info = json.load(f)
        self.channels = list(self.info.get("channels", "RGBAD"))
        assert set("RGBA").issubset(
            set(self.channels)
        ), "The blender output should at least have RGBA images."
        names = set(x.filename for x in self.zipfile.infolist())
        for i in itertools.count():
            name = f"{i:05}.json"
            if name not in names:
                break
            with self.zipfile.open(name, "r") as f:
                self.infos.append(json.load(f))

    @property
    def num_views(self) -> int:
        return len(self.infos)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels)

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        for ch in channels:
            if ch not in self.channel_names:
                raise ValueError(f"unsupported channel: {ch}")

        # Gather (a superset of) the requested channels.
        channel_map = {}
        if any(x in channels for x in "RGBA"):
            with self.zipfile.open(f"{index:05}.png", "r") as f:
                rgba = np.array(Image.open(f)).astype(np.float32) / 255.0
                channel_map.update(zip("RGBA", rgba.transpose([2, 0, 1])))
        if "D" in channels:
            with self.zipfile.open(f"{index:05}_depth.png", "r") as f:
                # Decode a 16-bit fixed-point number.
                fp = np.array(Image.open(f))
                inf_dist = fp == 0xFFFF
                channel_map["D"] = np.where(
                    inf_dist,
                    np.inf,
                    self.infos[index]["max_depth"] * (fp.astype(np.float32) / 65536),
                )
        if "MatAlpha" in channels:
            with self.zipfile.open(f"{index:05}_MatAlpha.png", "r") as f:
                channel_map["MatAlpha"] = (
                    np.array(Image.open(f)).astype(np.float32) / 65536
                )

        # The order of channels is user-specified.
        combined = np.stack([channel_map[k] for k in channels], axis=-1)

        h, w, _ = combined.shape
        return self.camera(index, w, h), combined

    def camera(self, index: int, width: int, height: int) -> ProjectiveCamera:
        info = self.infos[index]
        return ProjectiveCamera(
            origin=np.array(info["origin"], dtype=np.float32),
            x=np.array(info["x"], dtype=np.float32),
            y=np.array(info["y"], dtype=np.float32),
            z=np.array(info["z"], dtype=np.float32),
            width=width,
            height=height,
            x_fov=info["x_fov"],
            y_fov=info["y_fov"],
        )


def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data


@dataclass
class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]

    @classmethod
    def from_rgbd(cls, vd: ViewData, num_views: Optional[int] = None) -> "PointCloud":
        """
        Construct a point cloud from the given view data.

        The data must have a depth channel. All other channels will be stored
        in the `channels` attribute of the result.

        Pixels in the rendered views are not converted into points in the cloud
        if they have infinite depth or less than 1.0 alpha.
        """
        channel_names = vd.channel_names
        if "D" not in channel_names:
            raise ValueError(f"view data must have depth channel")
        depth_index = channel_names.index("D")

        all_coords = []
        all_channels = defaultdict(list)

        if num_views is None:
            num_views = vd.num_views
        for i in range(num_views):
            camera, channel_values = vd.load_view(i, channel_names)
            flat_values = channel_values.reshape([-1, len(channel_names)])

            # Create an array of integer (x, y) image coordinates for Camera methods.
            image_coords = camera.image_coords()

            # Select subset of pixels that have meaningful depth/color.
            image_mask = np.isfinite(flat_values[:, depth_index])
            if "A" in channel_names:
                image_mask = image_mask & (
                    flat_values[:, channel_names.index("A")] >= 1 - 1e-5
                )
            image_coords = image_coords[image_mask]
            flat_values = flat_values[image_mask]

            # Use the depth and camera information to compute the coordinates
            # corresponding to every visible pixel.
            camera_rays = camera.camera_rays(image_coords)
            camera_origins = camera_rays[:, 0]
            camera_directions = camera_rays[:, 1]
            depth_dirs = camera.depth_directions(image_coords)
            ray_scales = flat_values[:, depth_index] / np.sum(
                camera_directions * depth_dirs, axis=-1
            )
            coords = camera_origins + camera_directions * ray_scales[:, None]

            all_coords.append(coords)
            for j, name in enumerate(channel_names):
                if name != "D":
                    all_channels[name].append(flat_values[:, j])

        if len(all_coords) == 0:
            return cls(coords=np.zeros([0, 3], dtype=np.float32), channels={})

        return cls(
            coords=np.concatenate(all_coords, axis=0),
            channels={k: np.concatenate(v, axis=0) for k, v in all_channels.items()},
        )

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx

            # Without this line, we may duplicate an index more than once if
            # there are duplicate points, due to rounding errors.
            cur_dists[idx] = -1

            cur_dists = np.minimum(cur_dists, compute_dists(idx))

        return self.subsample(indices, **subsample_kwargs)

    def subsample(
        self, indices: np.ndarray, average_neighbors: bool = False
    ) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(
            self.coords
        )

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack(
            [preprocess(self.channels[name], name) for name in channel_names], axis=-1
        )
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = (
                norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            )
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0)
                for k, v in self.channels.items()
            },
        )


def load_or_create_multimodal_batch(
    device: torch.device,
    *,
    mesh_path: Optional[str] = None,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    point_count: int = 2**14,
    random_sample_count: int = 2**19,
    pc_num_views: int = 40,
    mv_light_mode: Optional[str] = None,
    mv_num_views: int = 20,
    mv_image_size: int = 512,
    mv_alpha_removal: str = "black",
    verbose: bool = False,
) -> AttrDict:
    if verbose:
        print("creating point cloud...")
    pc = load_or_create_pc(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        random_sample_count=random_sample_count,
        point_count=point_count,
        num_views=pc_num_views,
        verbose=verbose,
    )
    raw_pc = np.concatenate([pc.coords, pc.select_channels(["R", "G", "B"])], axis=-1)
    encode_me = torch.from_numpy(raw_pc).float().to(device)
    batch = AttrDict(points=encode_me.t()[None])
    if mv_light_mode:
        if verbose:
            print("creating multiview...")
        with load_or_create_multiview(
            mesh_path=mesh_path,
            model_path=model_path,
            cache_dir=cache_dir,
            num_views=mv_num_views,
            extract_material=False,
            light_mode=mv_light_mode,
            verbose=verbose,
        ) as mv:
            cameras, views, view_alphas, depths = [], [], [], []
            for view_idx in range(mv.num_views):
                camera, view = mv.load_view(
                    view_idx,
                    (
                        ["R", "G", "B", "A"]
                        if "A" in mv.channel_names
                        else ["R", "G", "B"]
                    ),
                )
                depth = None
                if "D" in mv.channel_names:
                    _, depth = mv.load_view(view_idx, ["D"])
                    depth = process_depth(depth, mv_image_size)
                view, alpha = process_image(
                    np.round(view * 255.0).astype(np.uint8),
                    mv_alpha_removal,
                    mv_image_size,
                )
                camera = camera.center_crop().resize_image(mv_image_size, mv_image_size)
                cameras.append(camera)
                views.append(view)
                view_alphas.append(alpha)
                depths.append(depth)
            batch.depths = [depths]
            batch.views = [views]
            batch.view_alphas = [view_alphas]
            batch.cameras = [cameras]
    return normalize_input_batch(batch, pc_scale=2.0, color_scale=1.0 / 255.0)


def load_or_create_pc(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    random_sample_count: int,
    point_count: int,
    num_views: int,
    verbose: bool = False,
) -> PointCloud:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if cache_dir is not None:
        cache_path = bf.join(
            cache_dir,
            f"pc_{bf.basename(path)}_mat_{num_views}_{random_sample_count}_{point_count}.npz",
        )
        if bf.exists(cache_path):
            return PointCloud.load(cache_path)
    else:
        cache_path = None

    with load_or_create_multiview(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        num_views=num_views,
        verbose=verbose,
    ) as mv:
        if verbose:
            print("extracting point cloud from multiview...")
        pc = mv_to_pc(
            multiview=mv,
            random_sample_count=random_sample_count,
            point_count=point_count,
        )
        if cache_path is not None:
            pc.save(cache_path)
        return pc


@contextmanager
def load_or_create_multiview(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    num_views: int = 20,
    extract_material: bool = True,
    light_mode: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[BlenderViewData]:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if extract_material:
        assert light_mode is None, "light_mode is ignored when extract_material=True"
    else:
        assert (
            light_mode is not None
        ), "must specify light_mode when extract_material=False"

    if cache_dir is not None:
        if extract_material:
            cache_path = bf.join(
                cache_dir, f"mv_{bf.basename(path)}_mat_{num_views}.zip"
            )
        else:
            cache_path = bf.join(
                cache_dir, f"mv_{bf.basename(path)}_{light_mode}_{num_views}.zip"
            )
        if bf.exists(cache_path):
            with bf.BlobFile(cache_path, "rb") as f:
                yield BlenderViewData(f)
                return
    else:
        cache_path = None

    common_kwargs = dict(
        fast_mode=True,
        extract_material=extract_material,
        camera_pose="random",
        light_mode=light_mode or "uniform",
        verbose=verbose,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = bf.join(tmp_dir, "out.zip")
        if mesh_path is not None:
            mesh = TriMesh.load(mesh_path)
            render_mesh(
                mesh=mesh,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        elif model_path is not None:
            render_model(
                model_path,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        if cache_path is not None:
            bf.copy(tmp_path, cache_path)
        with bf.BlobFile(tmp_path, "rb") as f:
            yield BlenderViewData(f)


def render_model(
    model_path: str,
    output_path: str,
    num_images: int,
    backend: str = "BLENDER_EEVEE",
    light_mode: str = "random",
    camera_pose: str = "random",
    camera_dist_min: float = 2.0,
    camera_dist_max: float = 2.0,
    fast_mode: bool = False,
    extract_material: bool = False,
    delete_material: bool = False,
    verbose: bool = False,
    timeout: float = 15 * 60,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = model_path
        tmp_out = os.path.join(tmp_dir, "out")
        zip_out = tmp_out + ".zip"
        os.mkdir(tmp_out)
        args = []
        if platform.system() == "Linux":
            # Needed to enable Eevee backend on headless linux.
            args = ["xvfb-run", "-a"]
        args.extend(
            [
                "blender",
                "-b",
                "-P",
                SCRIPT_PATH,
                "--",
                "--input_path",
                tmp_in,
                "--output_path",
                tmp_out,
                "--num_images",
                str(num_images),
                "--backend",
                backend,
                "--light_mode",
                light_mode,
                "--camera_pose",
                camera_pose,
                "--camera_dist_min",
                str(camera_dist_min),
                "--camera_dist_max",
                str(camera_dist_max),
                "--uniform_light_direction",
                *[str(x) for x in UNIFORM_LIGHT_DIRECTION],
                "--basic_ambient",
                str(BASIC_AMBIENT_COLOR),
                "--basic_diffuse",
                str(BASIC_DIFFUSE_COLOR),
            ]
        )
        if fast_mode:
            args.append("--fast_mode")
        if extract_material:
            args.append("--extract_material")
        if delete_material:
            args.append("--delete_material")
        if verbose:
            subprocess.check_call(args)
        else:
            try:
                output = subprocess.check_output(
                    args, stderr=subprocess.STDOUT, timeout=timeout
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{exc}: {exc.output}") from exc
        if not os.path.exists(os.path.join(tmp_out, "info.json")):
            if verbose:
                # There is no output available, since it was
                # logged directly to stdout/stderr.
                raise RuntimeError(f"render failed: output file missing")
            else:
                raise RuntimeError(
                    f"render failed: output file missing. Output: {output}"
                )
        _combine_rgba(tmp_out)
        with zipfile.ZipFile(zip_out, mode="w") as zf:
            for name in os.listdir(tmp_out):
                zf.write(os.path.join(tmp_out, name), name)
        bf.copy(zip_out, output_path, overwrite=True)


def _combine_rgba(out_dir: str):
    i = 0
    while True:
        paths = [os.path.join(out_dir, f"{i:05}_{ch}.png") for ch in "rgba"]
        if not os.path.exists(paths[0]):
            break
        joined = np.stack(
            [(np.array(Image.open(path)) >> 8).astype(np.uint8) for path in paths],
            axis=-1,
        )
        Image.fromarray(joined).save(os.path.join(out_dir, f"{i:05}.png"))
        for path in paths:
            os.remove(path)
        i += 1


def mv_to_pc(
    multiview: ViewData, random_sample_count: int, point_count: int
) -> PointCloud:
    pc = PointCloud.from_rgbd(multiview)

    # Handle empty samples.
    if len(pc.coords) == 0:
        pc = PointCloud(
            coords=np.zeros([1, 3]),
            channels=dict(zip("RGB", np.zeros([3, 1]))),
        )
    while len(pc.coords) < point_count:
        pc = pc.combine(pc)
        # Prevent duplicate points; some models may not like it.
        pc.coords += np.random.normal(size=pc.coords.shape) * 1e-4

    pc = pc.random_sample(random_sample_count)
    pc = pc.farthest_point_sample(point_count, average_neighbors=True)

    return pc


def normalize_input_batch(
    batch: AttrDict, *, pc_scale: float, color_scale: float
) -> AttrDict:
    res = batch.copy()
    scale_vec = torch.tensor(
        [*([pc_scale] * 3), *([color_scale] * 3)], device=batch.points.device
    )
    res.points = res.points * scale_vec[:, None]

    if "cameras" in res:
        res.cameras = [
            [cam.scale_scene(pc_scale) for cam in cams] for cams in res.cameras
        ]

    if "depths" in res:
        res.depths = [[depth * pc_scale for depth in depths] for depths in res.depths]

    return res


def process_depth(depth_img: np.ndarray, image_size: int) -> np.ndarray:
    depth_img = center_crop(depth_img)
    depth_img = resize(depth_img, width=image_size, height=image_size)
    return np.squeeze(depth_img)


def process_image(
    img_or_img_arr: Union[Image.Image, np.ndarray], alpha_removal: str, image_size: int
):
    if isinstance(img_or_img_arr, np.ndarray):
        img = Image.fromarray(img_or_img_arr)
        img_arr = img_or_img_arr
    else:
        img = img_or_img_arr
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            # Grayscale
            rgb = Image.new("RGB", img.size)
            rgb.paste(img)
            img = rgb
            img_arr = np.array(img)

    img = center_crop(img)
    alpha = get_alpha(img)
    img = remove_alpha(img, mode=alpha_removal)
    alpha = alpha.resize((image_size,) * 2, resample=Image.BILINEAR)
    img = img.resize((image_size,) * 2, resample=Image.BILINEAR)
    return img, alpha


def center_crop(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
) -> Union[Image.Image, torch.Tensor, np.ndarray]:
    """
    Center crops an image.
    """
    if isinstance(img, (np.ndarray, torch.Tensor)):
        height, width = img.shape[:2]
    else:
        width, height = img.size
    size = min(width, height)
    left, top = (width - size) // 2, (height - size) // 2
    right, bottom = left + size, top + size
    if isinstance(img, (np.ndarray, torch.Tensor)):
        img = img[top:bottom, left:right]
    else:
        img = img.crop((left, top, right, bottom))
    return img


def resize(
    img: Union[Image.Image, torch.Tensor, np.ndarray],
    *,
    height: int,
    width: int,
    min_value: Optional[Any] = None,
    max_value: Optional[Any] = None,
) -> Union[Image.Image, torch.Tensor, np.ndarray]:
    """
    :param: img: image in HWC order
    :return: currently written for downsampling
    """

    orig, cls = img, type(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    dtype = img.dtype
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    ndim = img.ndim
    if img.ndim == 2:
        img = img.unsqueeze(-1)

    if min_value is None and max_value is None:
        # .clamp throws an error when both are None
        min_value = -np.inf

    img = img.permute(2, 0, 1)
    size = (height, width)
    img = (
        F.interpolate(img[None].float(), size=size, mode="area")[0]
        .clamp(min_value, max_value)
        .to(img.dtype)
        .permute(1, 2, 0)
    )

    if ndim < img.ndim:
        img = img.squeeze(-1)
    if not isinstance(orig, torch.Tensor):
        img = img.numpy()
    img = img.astype(dtype)
    if isinstance(orig, Image.Image):
        img = Image.fromarray(img)

    return img


def get_alpha(img: Image.Image) -> Image.Image:
    """
    :return: the alpha channel separated out as a grayscale image
    """
    img_arr = np.asarray(img)
    if img_arr.shape[2] == 4:
        alpha = img_arr[:, :, 3]
    else:
        alpha = np.full(img_arr.shape[:2], 255, dtype=np.uint8)
    alpha = Image.fromarray(alpha)
    return alpha


def remove_alpha(img: Image.Image, mode: str = "random") -> Image.Image:
    """
    No op if the image doesn't have an alpha channel.

    :param: mode: Defaults to "random" but has an option to use a "black" or
        "white" background

    :return: image with alpha removed
    """
    img_arr = np.asarray(img)
    if img_arr.shape[2] == 4:
        # Add bg to get rid of alpha channel
        if mode == "random":
            height, width = img_arr.shape[:2]
            bg = Image.fromarray(
                random.choice([_black_bg, _gray_bg, _checker_bg, _noise_bg])(
                    height, width
                )
            )
            bg.paste(img, mask=img)
            img = bg
        elif mode == "black" or mode == "white":
            img_arr = img_arr.astype(float)
            rgb, alpha = img_arr[:, :, :3], img_arr[:, :, -1:] / 255
            background = (
                np.zeros((1, 1, 3)) if mode == "black" else np.full((1, 1, 3), 255)
            )
            rgb = rgb * alpha + background * (1 - alpha)
            img = Image.fromarray(np.round(rgb).astype(np.uint8))
    return img
