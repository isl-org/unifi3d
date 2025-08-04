from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import rootutils

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)

import numpy as np
import torch
import open3d as o3d

from unifi3d.models.modules.shap_e.nerstf import Model, NeRFModel
from unifi3d.models.modules.shap_e.nn import MetaModule, subdict
from unifi3d.utils.model.shap_e_utils import (
    ArrayType,
    AttrDict,
    Query,
    append_tensor,
    safe_divide,
    sample_pmf,
    to_torch,
)
from unifi3d.utils.rendering.camera import (
    DifferentiableCamera,
    DifferentiableProjectiveCamera,
    get_image_coords,
)

from unifi3d.utils.data.sdf_utils import volume_to_mesh

BASIC_AMBIENT_COLOR = 0.3
BASIC_DIFFUSE_COLOR = 0.7


@dataclass
class VolumeRange:
    t0: torch.Tensor
    t1: torch.Tensor
    intersected: torch.Tensor

    def __post_init__(self):
        assert self.t0.shape == self.t1.shape == self.intersected.shape

    def next_t0(self):
        """
        Given convex volume1 and volume2, where volume1 is contained in
        volume2, this function returns the t0 at which rays leave volume1 and
        intersect with volume2 \\ volume1.
        """
        return self.t1 * self.intersected.float()

    def extend(self, another: "VolumeRange") -> "VolumeRange":
        """
        The ranges at which rays intersect with either one, or both, or none of
        the self and another are merged together.
        """
        return VolumeRange(
            t0=torch.where(self.intersected, self.t0, another.t0),
            t1=torch.where(another.intersected, another.t1, self.t1),
            intersected=torch.logical_or(self.intersected, another.intersected),
        )

    def partition(self, ts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Partitions t0 and t1 into n_samples intervals.

        :param ts: [batch_size, *shape, n_samples, 1]
        :return: a tuple of (
            lower: [batch_size, *shape, n_samples, 1]
            upper: [batch_size, *shape, n_samples, 1]
            delta: [batch_size, *shape, n_samples, 1]
        ) where

            ts \\in [lower, upper]
            deltas = upper - lower
        """
        mids = (ts[..., 1:, :] + ts[..., :-1, :]) * 0.5
        lower = torch.cat([self.t0[..., None, :], mids], dim=-2)
        upper = torch.cat([mids, self.t1[..., None, :]], dim=-2)
        delta = upper - lower
        assert lower.shape == upper.shape == delta.shape == ts.shape
        return lower, upper, delta


class Volume(ABC):
    """
    An abstraction of rendering volume.
    """

    @abstractmethod
    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        epsilon: float = 1e-6,
    ) -> VolumeRange:
        """
        :param origin: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
        :param params: Optional meta parameters in case Volume is parametric
        :param epsilon: to stabilize calculations

        :return: A tuple of (t0, t1, intersected) where each has a shape
            [batch_size, *shape, 1]. If a ray intersects with the volume, `o + td` is
            in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed
            to be on the boundary of the volume.
        """


class BoundingBoxVolume(MetaModule, Volume):
    """
    Axis-aligned bounding box defined by the two opposite corners.
    """

    def __init__(
        self,
        *,
        bbox_min: ArrayType,
        bbox_max: ArrayType,
        min_dist: float = 0.0,
        min_t_range: float = 1e-3,
        device: torch.device = torch.device("cuda"),
    ):
        """
        :param bbox_min: the left/bottommost corner of the bounding box
        :param bbox_max: the other corner of the bounding box
        :param min_dist: all rays should start at least this distance away from the origin.
        """
        super().__init__()

        self.bbox_min = to_torch(bbox_min).to(device)
        self.bbox_max = to_torch(bbox_max).to(device)
        self.min_dist = min_dist
        self.min_t_range = min_t_range
        self.bbox = torch.stack([self.bbox_min, self.bbox_max])
        assert self.bbox.shape == (2, 3)
        assert self.min_dist >= 0.0
        assert self.min_t_range > 0.0
        self.device = device

    def intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0_lower: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        epsilon=1e-6,
    ) -> VolumeRange:
        """
        :param origin: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0_lower: Optional [batch_size, *shape, 1] lower bound of t0 when intersecting this volume.
        :param params: Optional meta parameters in case Volume is parametric
        :param epsilon: to stabilize calculations

        :return: A tuple of (t0, t1, intersected) where each has a shape
            [batch_size, *shape, 1]. If a ray intersects with the volume, `o + td` is
            in the volume for all t in [t0, t1]. If the volume is bounded, t1 is guaranteed
            to be on the boundary of the volume.
        """

        batch_size, *shape, _ = origin.shape
        ones = [1] * len(shape)
        bbox = self.bbox.view(1, *ones, 2, 3)
        ts = safe_divide(
            bbox - origin[..., None, :], direction[..., None, :], epsilon=epsilon
        )

        # Cases to think about:
        #
        #   1. t1 <= t0: the ray does not pass through the AABB.
        #   2. t0 < t1 <= 0: the ray intersects but the BB is behind the origin.
        #   3. t0 <= 0 <= t1: the ray starts from inside the BB
        #   4. 0 <= t0 < t1: the ray is not inside and intersects with the BB twice.
        #
        # 1 and 4 are clearly handled from t0 < t1 below.
        # Making t0 at least min_dist (>= 0) takes care of 2 and 3.
        t0 = ts.min(dim=-2).values.max(dim=-1, keepdim=True).values.clamp(self.min_dist)
        t1 = ts.max(dim=-2).values.min(dim=-1, keepdim=True).values
        assert t0.shape == t1.shape == (batch_size, *shape, 1)
        if t0_lower is not None:
            assert t0.shape == t0_lower.shape
            t0 = torch.maximum(t0, t0_lower)

        intersected = t0 + self.min_t_range < t1
        t0 = torch.where(intersected, t0, torch.zeros_like(t0))
        t1 = torch.where(intersected, t1, torch.ones_like(t1))

        return VolumeRange(t0=t0, t1=t1, intersected=intersected)


class Renderer(MetaModule):
    """
    A rendering abstraction that can render rays and views by calling the
    appropriate models. The models are instantiated outside but registered in
    this module.
    """

    @abstractmethod
    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        Returns a backproppable rendering of a view

        :param batch: contains
            - height: Optional[int]
            - width: Optional[int]
            - inner_batch_size or ray_batch_size: Optional[int] defaults to 4096 rays

            And additionally, to specify poses with a default up direction:
            - poses: [batch_size x *shape x 2 x 3] where poses[:, ..., 0, :] are the camera
                positions, and poses[:, ..., 1, :] are the z-axis (toward the object) of
                the camera frame.
            - camera: DifferentiableCamera. Assumes the same camera position
                across batch for simplicity.  Could eventually support
                batched cameras.

            or to specify a batch of arbitrary poses:
            - cameras: DifferentiableCameraBatch of shape [batch_size x *shape].

        :param params: Meta parameters
        :param options: Optional[Dict]
        """


class RayRenderer(Renderer):
    """
    A rendering abstraction that can render rays and views by calling the
    appropriate models. The models are instantiated outside but registered in
    this module.
    """

    @abstractmethod
    def render_rays(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        :param batch: has
            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.
            - radii (optional): [batch_size x ... x 1] the "thickness" of each ray.
        :param options: Optional[Dict]
        """

    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
    ) -> AttrDict:
        output = render_views_from_rays(
            self.render_rays,
            batch,
            params=params,
            options=options,
            device=self.device,
        )
        return output

    def forward(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[Dict] = None,
    ) -> AttrDict:
        """
        :param batch: must contain either

            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.

            or

            - poses: [batch_size x 2 x 3] where poses[:, 0] are the camera
                positions, and poses[:, 1] are the z-axis (toward the object) of
                the camera frame.
            - camera: an instance of Camera that implements camera_rays

            or

            - cameras: DifferentiableCameraBatch of shape [batch_size x *shape].

            For both of the above two options, these may be specified.
            - height: Optional[int]
            - width: Optional[int]
            - ray_batch_size or inner_batch_size: Optional[int] defaults to 4096 rays

        :param params: a dictionary of optional meta parameters.
        :param options: A Dict of other hyperparameters that could be
            related to rendering or debugging

        :return: a dictionary containing

            - channels: [batch_size, *shape, n_channels]
            - distances: [batch_size, *shape, 1]
            - transmittance: [batch_size, *shape, 1]
            - aux_losses: Dict[str, torch.Tensor]
        """

        if "rays" in batch:
            for key in ["poses", "camera", "height", "width"]:
                assert key not in batch
            return self.render_rays(batch, params=params, options=options)
        elif "poses" in batch or "cameras" in batch:
            assert "rays" not in batch
            if "poses" in batch:
                assert "camera" in batch
            else:
                assert "camera" not in batch
            return self.render_views(batch, params=params, options=options)

        raise NotImplementedError


def get_camera_from_batch(
    batch: AttrDict,
) -> Tuple[DifferentiableCamera, int, Tuple[int]]:
    if "poses" in batch:
        assert not "cameras" in batch
        batch_size, *inner_shape, n_vecs, spatial_dim = batch.poses.shape
        assert n_vecs == 2 and spatial_dim == 3
        inner_batch_size = int(np.prod(inner_shape))
        poses = batch.poses.view(batch_size * inner_batch_size, 2, 3)
        position, direction = poses[:, 0], poses[:, 1]
        camera = projective_camera_frame(position, direction, batch.camera)
    elif "cameras" in batch:
        assert not "camera" in batch
        batch_size, *inner_shape = batch.cameras.shape
        camera = batch.cameras.flat_camera
    else:
        raise ValueError(f'neither "poses" nor "cameras" found in keys: {batch.keys()}')
    if "height" in batch and "width" in batch:
        camera = camera.resize_image(batch.width, batch.height)
    return camera, batch_size, inner_shape


def render_views_from_rays(
    render_rays_fn: Callable[[AttrDict, AttrDict, AttrDict], AttrDict],
    batch: AttrDict,
    params: Optional[Dict] = None,
    options: Optional[Dict] = None,
    device: torch.device = torch.device("cuda"),
) -> AttrDict:
    camera, batch_size, inner_shape = get_camera_from_batch(batch)
    inner_batch_size = int(np.prod(inner_shape))

    coords = get_image_coords(camera.width, camera.height).to(device)
    coords = torch.broadcast_to(
        coords.unsqueeze(0), [batch_size * inner_batch_size, *coords.shape]
    )
    rays = camera.camera_rays(coords)

    # mip-NeRF radii calculation from: https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/datasets.py#L193-L200
    directions = rays.view(
        batch_size, inner_batch_size, camera.height, camera.width, 2, 3
    )[..., 1, :]
    neighbor_dists = torch.linalg.norm(
        directions[:, :, :, 1:] - directions[:, :, :, :-1], dim=-1
    )
    neighbor_dists = torch.cat([neighbor_dists, neighbor_dists[:, :, :, -2:-1]], dim=3)
    radii = (neighbor_dists * 2 / np.sqrt(12)).view(batch_size, -1, 1)

    rays = rays.view(batch_size, inner_batch_size * camera.height * camera.width, 2, 3)

    if isinstance(camera, DifferentiableProjectiveCamera):
        # Compute the camera z direction corresponding to every ray's pixel.
        # Used for depth computations below.
        z_directions = (
            (camera.z / torch.linalg.norm(camera.z, dim=-1, keepdim=True))
            .reshape([batch_size, inner_batch_size, 1, 3])
            .repeat(1, 1, camera.width * camera.height, 1)
            .reshape(1, inner_batch_size * camera.height * camera.width, 3)
        )

    ray_batch_size = batch.get("ray_batch_size", batch.get("inner_batch_size", 4096))
    assert rays.shape[1] % ray_batch_size == 0
    n_batches = rays.shape[1] // ray_batch_size

    output_list = AttrDict(aux_losses=dict())

    for idx in range(n_batches):
        rays_batch = AttrDict(
            rays=rays[:, idx * ray_batch_size : (idx + 1) * ray_batch_size],
            radii=radii[:, idx * ray_batch_size : (idx + 1) * ray_batch_size],
        )
        output = render_rays_fn(rays_batch, params=params, options=options)

        if isinstance(camera, DifferentiableProjectiveCamera):
            z_batch = z_directions[:, idx * ray_batch_size : (idx + 1) * ray_batch_size]
            ray_directions = rays_batch.rays[:, :, 1]
            z_dots = (ray_directions * z_batch).sum(-1, keepdim=True)
            output.depth = output.distances * z_dots

        output_list = output_list.combine(output, append_tensor)

    def _resize(val_list: List[torch.Tensor]):
        val = torch.cat(val_list, dim=1)
        assert val.shape[1] == inner_batch_size * camera.height * camera.width
        return val.view(batch_size, *inner_shape, camera.height, camera.width, -1)

    def _avg(_key: str, loss_list: List[torch.Tensor]):
        return sum(loss_list) / n_batches

    output = AttrDict(
        {
            name: _resize(val_list)
            for name, val_list in output_list.items()
            if name != "aux_losses"
        }
    )
    output.aux_losses = output_list.aux_losses.map(_avg)

    return output


def render_rays(
    rays: torch.Tensor,
    parts: List["RayVolumeIntegral"],
    void_model: NeRFModel,
    shared: bool = False,
    prev_raw_outputs: Optional[List[AttrDict]] = None,
    render_with_direction: bool = True,
    importance_sampling_options: Optional[Dict[str, Any]] = None,
) -> Tuple["RayVolumeIntegralResults", List["RaySampler"], List[AttrDict]]:
    """
    Perform volumetric rendering over a partition of possible t's in the union
    of rendering volumes (written below with some abuse of notations)

        C(r) := sum(
            transmittance(t[i]) *
            integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t[i], t[i + 1]],
            )
            for i in range(len(parts))
        ) + transmittance(t[-1]) * void_model(t[-1]).channels

    where

    1) transmittance(s) := exp(-integrate(density, [t[0], s])) calculates the
       probability of light passing through the volume specified by [t[0], s].
       (transmittance of 1 means light can pass freely)
    2) density and channels are obtained by evaluating the appropriate
       part.model at time t.
    3) [t[i], t[i + 1]] is defined as the range of t where the ray intersects
       (parts[i].volume \\ union(part.volume for part in parts[:i])) at the surface
       of the shell (if bounded). If the ray does not intersect, the integral over
       this segment is evaluated as 0 and transmittance(t[i + 1]) :=
       transmittance(t[i]).
    4) The last term is integration to infinity (e.g. [t[-1], math.inf]) that
       is evaluated by the void_model (i.e. we consider this space to be empty).

    :param rays: [batch_size x ... x 2 x 3] origin and direction.
    :param parts: disjoint volume integrals.
    :param void_model: use this model to integrate over the empty space
    :param shared: All RayVolumeIntegrals are calculated with the same model.
    :param prev_raw_outputs: Raw outputs from the previous rendering step

    :return: A tuple of
        - AttrDict containing the rendered `channels`, `distances`, and the `aux_losses`
        - A list of importance samplers for additional fine-grained rendering
        - A list of raw output for each interval
    """
    if importance_sampling_options is None:
        importance_sampling_options = {}

    origin, direc = rays[..., 0, :], rays[..., 1, :]

    if prev_raw_outputs is None:
        prev_raw_outputs = [None] * len(parts)

    samplers = []
    raw_outputs = []
    t0 = None
    results = None

    for part_i, prev_raw_i in zip(parts, prev_raw_outputs):
        # Integrate over [t[i], t[i + 1]]
        results_i = part_i.render_rays(
            origin,
            direc,
            t0=t0,
            prev_raw=prev_raw_i,
            shared=shared,
            render_with_direction=render_with_direction,
        )

        # Create an importance sampler for (optional) fine rendering
        samplers.append(
            ImportanceRaySampler(
                results_i.volume_range, results_i.raw, **importance_sampling_options
            )
        )
        raw_outputs.append(results_i.raw)

        # Pass t[i + 1] as the start of integration for the next interval.
        t0 = results_i.volume_range.next_t0()

        # Combine the results from [t[0], t[i]] and [t[i], t[i+1]]
        results = results_i if results is None else results.combine(results_i)

    # While integrating out [t[-1], math.inf] is the correct thing to do, this
    # erases a lot of useful information. Also, void_model is meant to predict
    # the channels at t=math.inf.

    # # Add the void background over [t[-1], math.inf] to complete integration.
    # results = results.combine(
    #     RayVolumeIntegralResults(
    #         output=AttrDict(
    #             channels=void_model(origin, direc),
    #             distances=torch.zeros_like(t0),
    #             aux_losses=AttrDict(),
    #         ),
    #         volume_range=VolumeRange(
    #             t0=t0,
    #             t1=torch.full_like(t0, math.inf),
    #             intersected=torch.full_like(results.volume_range.intersected, True),
    #         ),
    #         # Void space extends to infinity. It is assumed that no light
    #         # passes beyond the void.
    #         transmittance=torch.zeros_like(results_i.transmittance),
    #     )
    # )

    results.output.channels = (
        results.output.channels
        + results.transmittance * void_model(Query(origin, direc))
    )

    return results, samplers, raw_outputs


@dataclass
class RayVolumeIntegralResults:
    """
    Stores the relevant state and results of

        integrate(
            lambda t: density(t) * channels(t) * transmittance(t),
            [t0, t1],
        )
    """

    # Rendered output and auxiliary losses
    # output.channels has shape [batch_size, *inner_shape, n_channels]
    output: AttrDict

    """
    Optional values
    """

    # Raw values contain the sampled `ts`, `density`, `channels`, etc.
    raw: Optional[AttrDict] = None

    # Integration
    volume_range: Optional[VolumeRange] = None

    # If a ray intersects, the transmittance from t0 to t1 (e.g. the
    # probability that the ray passes through this volume).
    # has shape [batch_size, *inner_shape, 1]
    transmittance: Optional[torch.Tensor] = None

    def combine(self, cur: "RayVolumeIntegralResults") -> "RayVolumeIntegralResults":
        """
        Combines the integration results of `self` over [t0, t1] and
        `cur` over [t1, t2] to produce a new set of results over [t0, t2] by
        using a similar equation to (4) in NeRF++:

            integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t0, t2]
            )

          = integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t0, t1]
            ) + transmittance(t1) * integrate(
                lambda t: density(t) * channels(t) * transmittance(t),
                [t1, t2]
            )
        """
        assert torch.allclose(self.volume_range.next_t0(), cur.volume_range.t0)

        def _combine_fn(
            prev_val: Optional[torch.Tensor],
            cur_val: Optional[torch.Tensor],
            *,
            prev_transmittance: torch.Tensor,
        ):
            assert prev_val is not None
            if cur_val is None:
                # cur_output.aux_losses are empty for the void_model.
                return prev_val
            return prev_val + prev_transmittance * cur_val

        output = self.output.combine(
            cur.output,
            combine_fn=partial(_combine_fn, prev_transmittance=self.transmittance),
        )

        combined = RayVolumeIntegralResults(
            output=output,
            volume_range=self.volume_range.extend(cur.volume_range),
            transmittance=self.transmittance * cur.transmittance,
        )
        return combined


@dataclass
class RayVolumeIntegral:
    model: NeRFModel
    volume: Volume
    sampler: "RaySampler"
    n_samples: int

    def render_rays(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        t0: Optional[torch.Tensor] = None,
        prev_raw: Optional[AttrDict] = None,
        shared: bool = False,
        render_with_direction: bool = True,
    ) -> "RayVolumeIntegralResults":
        """
        Perform volumetric rendering over the given volume.

        :param position: [batch_size, *shape, 3]
        :param direction: [batch_size, *shape, 3]
        :param t0: Optional [batch_size, *shape, 1]
        :param prev_raw: the raw outputs when using multiple levels with this model.
        :param shared: means the same model is used for all RayVolumeIntegral's
        :param render_with_direction: use the incoming ray direction when querying the model.

        :return: RayVolumeIntegralResults
        """
        # 1. Intersect the rays with the current volume and sample ts to
        # integrate along.
        vrange = self.volume.intersect(origin, direction, t0_lower=t0)
        ts = self.sampler.sample(vrange.t0, vrange.t1, self.n_samples)

        if prev_raw is not None and not shared:
            # Append the previous ts now before fprop because previous
            # rendering used a different model and we can't reuse the output.
            ts = torch.sort(torch.cat([ts, prev_raw.ts], dim=-2), dim=-2).values

        # Shape sanity checks
        batch_size, *_shape, _t0_dim = vrange.t0.shape
        _, *ts_shape, _ts_dim = ts.shape

        # 2. Get the points along the ray and query the model
        directions = torch.broadcast_to(
            direction.unsqueeze(-2), [batch_size, *ts_shape, 3]
        )
        positions = origin.unsqueeze(-2) + ts * directions

        optional_directions = directions if render_with_direction else None
        mids = (ts[..., 1:, :] + ts[..., :-1, :]) / 2
        raw = self.model(
            Query(
                position=positions,
                direction=optional_directions,
                t_min=torch.cat([vrange.t0[..., None, :], mids], dim=-2),
                t_max=torch.cat([mids, vrange.t1[..., None, :]], dim=-2),
            )
        )
        raw.ts = ts

        if prev_raw is not None and shared:
            # We can append the additional queries to previous raw outputs
            # before integration
            copy = prev_raw.copy()
            result = torch.sort(
                torch.cat([raw.pop("ts"), copy.pop("ts")], dim=-2), dim=-2
            )
            merge_results = partial(self._merge_results, dim=-2, indices=result.indices)
            raw = raw.combine(copy, merge_results)
            raw.ts = result.values

        # 3. Integrate the raw results
        output, transmittance = self.integrate_samples(vrange, raw)

        # 4. Clean up results that do not intersect with the volume.
        transmittance = torch.where(
            vrange.intersected, transmittance, torch.ones_like(transmittance)
        )

        def _mask_fn(_key: str, tensor: torch.Tensor):
            return torch.where(vrange.intersected, tensor, torch.zeros_like(tensor))

        def _is_tensor(_key: str, value: Any):
            return isinstance(value, torch.Tensor)

        output = output.map(map_fn=_mask_fn, should_map=_is_tensor)

        return RayVolumeIntegralResults(
            output=output,
            raw=raw,
            volume_range=vrange,
            transmittance=transmittance,
        )

    def integrate_samples(
        self,
        volume_range: VolumeRange,
        raw: AttrDict,
    ) -> Tuple[AttrDict, torch.Tensor]:
        """
        Integrate the raw.channels along with other aux_losses and values to
        produce the final output dictionary containing rendered `channels`,
        estimated `distances` and `aux_losses`.

        :param volume_range: Specifies the integral range [t0, t1]
        :param raw: Contains a dict of function evaluations at ts. Should have

            density: torch.Tensor [batch_size, *shape, n_samples, 1]
            channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
            aux_losses: {key: torch.Tensor [batch_size, *shape, n_samples, 1] for each key}
            no_weight_grad_aux_losses: an optional set of losses for which the weights
                                       should be detached before integration.

            after the call, integrate_samples populates some intermediate calculations
            for later use like

            weights: torch.Tensor [batch_size, *shape, n_samples, 1] (density *
                transmittance)[i] weight for each rgb output at [..., i, :].
        :returns: a tuple of (
            a dictionary of rendered outputs and aux_losses,
            transmittance of this volume,
        )
        """

        # 1. Calculate the weights
        _, _, dt = volume_range.partition(raw.ts)
        ddensity = raw.density * dt

        mass = torch.cumsum(ddensity, dim=-2)
        transmittance = torch.exp(-mass[..., -1, :])

        alphas = 1.0 - torch.exp(-ddensity)
        Ts = torch.exp(
            torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[..., :-1, :]], dim=-2)
        )
        # This is the probability of light hitting and reflecting off of
        # something at depth [..., i, :].
        weights = alphas * Ts

        # 2. Integrate all results
        def _integrate(key: str, samples: torch.Tensor, weights: torch.Tensor):
            if key == "density":
                # Omit integrating the density, because we don't need it
                return None
            return torch.sum(samples * weights, dim=-2)

        def _is_tensor(_key: str, value: Any):
            return isinstance(value, torch.Tensor)

        if raw.no_weight_grad_aux_losses:
            extra_aux_losses = raw.no_weight_grad_aux_losses.map(
                partial(_integrate, weights=weights.detach()), should_map=_is_tensor
            )
        else:
            extra_aux_losses = {}
        output = raw.map(partial(_integrate, weights=weights), should_map=_is_tensor)
        if "no_weight_grad_aux_losses" in output:
            del output["no_weight_grad_aux_losses"]
        output.aux_losses.update(extra_aux_losses)

        # Integrating the ts yields the distance away from the origin; rename the variable.
        output.distances = output.ts
        del output["ts"]
        del output["density"]

        assert output.distances.shape == (*output.channels.shape[:-1], 1)
        assert output.channels.shape[:-1] == raw.channels.shape[:-2]
        assert output.channels.shape[-1] == raw.channels.shape[-1]

        # 3. Reduce loss
        def _reduce_loss(_key: str, loss: torch.Tensor):
            return loss.view(loss.shape[0], -1).sum(dim=-1)

        # 4. Store other useful calculations
        raw.weights = weights

        output.aux_losses = output.aux_losses.map(_reduce_loss)

        return output, transmittance

    def _merge_results(
        self,
        a: Optional[torch.Tensor],
        b: torch.Tensor,
        dim: int,
        indices: torch.Tensor,
    ):
        """
        :param a: [..., n_a, ...]. The other dictionary containing the b's may
            contain extra tensors from earlier calculations, so a can be None.
        :param b: [..., n_b, ...]
        :param dim: dimension to merge
        :param indices: how the merged results should be sorted at the end
        :return: a concatted and sorted tensor of size [..., n_a + n_b, ...]
        """
        if a is None:
            return None

        merged = torch.cat([a, b], dim=dim)
        return torch.gather(
            merged, dim=dim, index=torch.broadcast_to(indices, merged.shape)
        )


class RaySampler(ABC):
    @abstractmethod
    def sample(
        self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """


class StratifiedRaySampler(RaySampler):
    """
    Instead of fixed intervals, a sample is drawn uniformly at random from each
    interval.
    """

    def __init__(self, depth_mode: str = "linear"):
        """
        :param depth_mode: linear samples ts linearly in depth. harmonic ensures
            closer points are sampled more densely.
        """
        self.depth_mode = depth_mode
        assert self.depth_mode in ("linear", "geometric", "harmonic")

    def sample(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        n_samples: int,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        ones = [1] * (len(t0.shape) - 1)
        ts = (
            torch.linspace(0, 1, n_samples)
            .view(*ones, n_samples)
            .to(t0.dtype)
            .to(t0.device)
        )

        if self.depth_mode == "linear":
            ts = t0 * (1.0 - ts) + t1 * ts
        elif self.depth_mode == "geometric":
            ts = (
                t0.clamp(epsilon).log() * (1.0 - ts) + t1.clamp(epsilon).log() * ts
            ).exp()
        elif self.depth_mode == "harmonic":
            # The original NeRF recommends this interpolation scheme for
            # spherical scenes, but there could be some weird edge cases when
            # the observer crosses from the inner to outer volume.
            ts = 1.0 / (
                1.0 / t0.clamp(epsilon) * (1.0 - ts) + 1.0 / t1.clamp(epsilon) * ts
            )

        mids = 0.5 * (ts[..., 1:] + ts[..., :-1])
        upper = torch.cat([mids, t1], dim=-1)
        lower = torch.cat([t0, mids], dim=-1)
        t_rand = torch.rand_like(ts)

        ts = lower + (upper - lower) * t_rand
        return ts.unsqueeze(-1)


class ImportanceRaySampler(RaySampler):
    """
    Given the initial estimate of densities, this samples more from
    regions/bins expected to have objects.
    """

    def __init__(
        self,
        volume_range: VolumeRange,
        raw: AttrDict,
        blur_pool: bool = False,
        alpha: float = 1e-5,
    ):
        """
        :param volume_range: the range in which a ray intersects the given volume.
        :param raw: dictionary of raw outputs from the NeRF models of shape
            [batch_size, *shape, n_coarse_samples, 1]. Should at least contain

            :param ts: earlier samples from the coarse rendering step
            :param weights: discretized version of density * transmittance
        :param blur_pool: if true, use 2-tap max + 2-tap blur filter from mip-NeRF.
        :param alpha: small value to add to weights.
        """
        self.volume_range = volume_range
        self.ts = raw.ts.clone().detach()
        self.weights = raw.weights.clone().detach()
        self.blur_pool = blur_pool
        self.alpha = alpha

    @torch.no_grad()
    def sample(
        self, t0: torch.Tensor, t1: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """
        :param t0: start time has shape [batch_size, *shape, 1]
        :param t1: finish time has shape [batch_size, *shape, 1]
        :param n_samples: number of ts to sample
        :return: sampled ts of shape [batch_size, *shape, n_samples, 1]
        """
        lower, upper, _ = self.volume_range.partition(self.ts)

        batch_size, *shape, n_coarse_samples, _ = self.ts.shape

        weights = self.weights
        if self.blur_pool:
            padded = torch.cat(
                [weights[..., :1, :], weights, weights[..., -1:, :]], dim=-2
            )
            maxes = torch.maximum(padded[..., :-1, :], padded[..., 1:, :])
            weights = 0.5 * (maxes[..., :-1, :] + maxes[..., 1:, :])
        weights = weights + self.alpha
        pmf = weights / weights.sum(dim=-2, keepdim=True)
        inds = sample_pmf(pmf, n_samples)
        assert inds.shape == (batch_size, *shape, n_samples, 1)
        assert (inds >= 0).all() and (inds < n_coarse_samples).all()

        t_rand = torch.rand(inds.shape, device=inds.device)
        lower_ = torch.gather(lower, -2, inds)
        upper_ = torch.gather(upper, -2, inds)

        ts = lower_ + (upper_ - lower_) * t_rand
        ts = torch.sort(ts, dim=-2).values
        return ts


class STFRendererBase(ABC):
    @abstractmethod
    def get_signed_distance(
        self,
        position: torch.Tensor,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_texture(
        self,
        position: torch.Tensor,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        pass


class NeRSTFRenderer(RayRenderer, STFRendererBase):
    def __init__(
        self,
        sdf: Optional[Model],
        tf: Optional[Model],
        nerstf: Optional[Model],
        void: NeRFModel,
        volume: Volume,
        grid_size: int,
        n_coarse_samples: int,
        n_fine_samples: int,
        importance_sampling_options: Optional[Dict[str, Any]] = None,
        separate_shared_samples: bool = False,
        texture_channels: Sequence[str] = ("R", "G", "B"),
        channel_scale: Sequence[float] = (255.0, 255.0, 255.0),
        ambient_color: Union[float, Tuple[float]] = BASIC_AMBIENT_COLOR,
        diffuse_color: Union[float, Tuple[float]] = BASIC_DIFFUSE_COLOR,
        specular_color: Union[float, Tuple[float]] = 0.0,
        output_srgb: bool = True,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(
            volume, BoundingBoxVolume
        ), "cannot sample points in unknown volume"
        assert (nerstf is not None) ^ (sdf is not None and tf is not None)
        self.sdf = sdf
        self.tf = tf
        self.nerstf = nerstf
        self.void = void
        self.volume = volume
        self.grid_size = grid_size
        self.n_coarse_samples = n_coarse_samples
        self.n_fine_samples = n_fine_samples
        self.importance_sampling_options = AttrDict(importance_sampling_options or {})
        self.separate_shared_samples = separate_shared_samples
        self.texture_channels = texture_channels
        self.channel_scale = to_torch(channel_scale).to(device)
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.output_srgb = output_srgb
        self.device = device
        self.to(device)

    def _query(
        self,
        query: Query,
        params: AttrDict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> AttrDict:
        no_dir_query = query.copy()
        no_dir_query.direction = None

        if options.get("rendering_mode", "stf") == "stf":
            assert query.direction is None

        if self.nerstf is not None:
            sdf = tf = self.nerstf(
                query,
                params=subdict(params, "nerstf"),
                options=options,
            )
        else:
            sdf = self.sdf(no_dir_query, params=subdict(params, "sdf"), options=options)
            tf = self.tf(query, params=subdict(params, "tf"), options=options)

        return AttrDict(
            density=sdf.density,
            signed_distance=sdf.signed_distance,
            channels=tf.channels,
            aux_losses=dict(),
        )

    def render_rays(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[AttrDict] = None,
    ) -> AttrDict:
        """
        :param batch: has

            - rays: [batch_size x ... x 2 x 3] specify the origin and direction of each ray.
        :param options: Optional[Dict]
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        # Necessary to tell the TF to use specific NeRF channels.
        options.rendering_mode = "nerf"

        model = partial(self._query, params=params, options=options)

        # First, render rays with coarse, stratified samples.
        options.nerf_level = "coarse"
        parts = [
            RayVolumeIntegral(
                model=model,
                volume=self.volume,
                sampler=StratifiedRaySampler(),
                n_samples=self.n_coarse_samples,
            ),
        ]
        coarse_results, samplers, coarse_raw_outputs = render_rays(
            batch.rays,
            parts,
            self.void,
            shared=not self.separate_shared_samples,
            render_with_direction=options.render_with_direction,
            importance_sampling_options=self.importance_sampling_options,
        )

        # Then, render with additional importance-weighted ray samples.
        options.nerf_level = "fine"
        parts = [
            RayVolumeIntegral(
                model=model,
                volume=self.volume,
                sampler=samplers[0],
                n_samples=self.n_fine_samples,
            ),
        ]
        fine_results, _, raw_outputs = render_rays(
            batch.rays,
            parts,
            self.void,
            shared=not self.separate_shared_samples,
            prev_raw_outputs=coarse_raw_outputs,
            render_with_direction=options.render_with_direction,
        )
        raw = raw_outputs[0]

        aux_losses = fine_results.output.aux_losses.copy()
        if self.separate_shared_samples:
            for key, val in coarse_results.output.aux_losses.items():
                aux_losses[key + "_coarse"] = val

        channels = fine_results.output.channels
        shape = [1] * (channels.ndim - 1) + [len(self.texture_channels)]
        channels = channels * self.channel_scale.view(*shape)

        res = AttrDict(
            channels=channels,
            transmittance=fine_results.transmittance,
            raw_signed_distance=raw.signed_distance,
            raw_density=raw.density,
            distances=fine_results.output.distances,
            t0=fine_results.volume_range.t0,
            t1=fine_results.volume_range.t1,
            intersected=fine_results.volume_range.intersected,
            aux_losses=aux_losses,
        )

        if self.separate_shared_samples:
            res.update(
                dict(
                    channels_coarse=(
                        coarse_results.output.channels * self.channel_scale.view(*shape)
                    ),
                    distances_coarse=coarse_results.output.distances,
                    transmittance_coarse=coarse_results.transmittance,
                )
            )

        return res

    def render_views(
        self,
        batch: AttrDict,
        params: Optional[Dict] = None,
        options: Optional[AttrDict] = None,
    ) -> AttrDict:
        """
        Returns a backproppable rendering of a view

        :param batch: contains either ["poses", "camera"], or ["cameras"]. Can
            optionally contain any of ["height", "width", "query_batch_size"]

        :param params: Meta parameters
            contains rendering_mode in ["stf", "nerf"]
        :param options: controls checkpointing, caching, and rendering.
            Can provide a `rendering_mode` in ["stf", "nerf"]
        """
        params = self.update(params)
        options = AttrDict() if options is None else AttrDict(options)

        if options.cache is None:
            created_cache = True
            options.cache = AttrDict()
        else:
            created_cache = False

        rendering_mode = options.get("rendering_mode", "stf")

        if rendering_mode == "nerf":
            output = render_views_from_rays(
                self.render_rays,
                batch,
                params=params,
                options=options,
                device=self.device,
            )

        elif rendering_mode == "stf":
            sdf_fn = tf_fn = nerstf_fn = None
            if self.nerstf is not None:
                nerstf_fn = partial(
                    self.nerstf.forward_batched,
                    params=subdict(params, "nerstf"),
                    options=options,
                )
            else:
                sdf_fn = partial(
                    self.sdf.forward_batched,
                    params=subdict(params, "sdf"),
                    options=options,
                )
                tf_fn = partial(
                    self.tf.forward_batched,
                    params=subdict(params, "tf"),
                    options=options,
                )
            output = render_views_from_stf(
                batch,
                options,
                sdf_fn=sdf_fn,
                tf_fn=tf_fn,
                nerstf_fn=nerstf_fn,
                volume=self.volume,
                grid_size=self.grid_size,
                channel_scale=self.channel_scale,
                texture_channels=self.texture_channels,
                ambient_color=self.ambient_color,
                diffuse_color=self.diffuse_color,
                specular_color=self.specular_color,
                output_srgb=self.output_srgb,
                device=self.device,
            )

        else:
            raise NotImplementedError

        if created_cache:
            del options["cache"]

        return output

    def get_signed_distance(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        if self.sdf is not None:
            return self.sdf(
                query, params=subdict(params, "sdf"), options=options
            ).signed_distance
        assert self.nerstf is not None
        return self.nerstf(
            query, params=subdict(params, "nerstf"), options=options
        ).signed_distance

    def get_texture(
        self,
        query: Query,
        params: Dict[str, torch.Tensor],
        options: AttrDict[str, Any],
    ) -> torch.Tensor:
        if self.tf is not None:
            return self.tf(
                query, params=subdict(params, "tf"), options=options
            ).channels
        assert self.nerstf is not None
        return self.nerstf(
            query, params=subdict(params, "nerstf"), options=options
        ).channels


def volume_query_points(
    volume: Volume,
    grid_size: int,
):
    assert isinstance(volume, BoundingBoxVolume)
    indices = torch.arange(grid_size**3, device=volume.bbox_min.device)
    zs = indices % grid_size
    ys = torch.div(indices, grid_size, rounding_mode="trunc") % grid_size
    xs = torch.div(indices, grid_size**2, rounding_mode="trunc") % grid_size
    combined = torch.stack([xs, ys, zs], dim=1)
    return (combined.float() / (grid_size - 1)) * (
        volume.bbox_max - volume.bbox_min
    ) + volume.bbox_min


def nerf_to_mesh(
    options: AttrDict[str, Any],
    nerstf_fn: Optional[Callable],
    volume: BoundingBoxVolume,
    grid_size: int,
    device: torch.device = torch.device("cuda"),
) -> AttrDict:
    batch_size = 1
    device_type = device.type

    query_batch_size = 4096
    query_points = volume_query_points(volume, grid_size)

    fn = nerstf_fn
    with torch.no_grad():
        field_out = fn(
            query=Query(position=query_points[None].repeat(batch_size, 1, 1)),
            query_batch_size=query_batch_size,
            options=options,
        )
    raw_density = field_out.density
    density = torch.squeeze(raw_density.reshape(batch_size, *([grid_size] * 3)))
    density = density.cpu().numpy()

    surface = volume_to_mesh(density, 0.5, -0.4, 0.4)

    # Rotate mesh to be consistent with other representations
    dir_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    R = dir_mesh.get_rotation_matrix_from_xyz((np.pi / 2, np.pi / 2, 0))
    surface.rotate(R, center=(0, 0, 0))

    return surface.to_legacy()
