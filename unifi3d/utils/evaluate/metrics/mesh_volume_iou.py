import numpy as np
import open3d as o3d

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import align_meshes


class MeshVolumeIOU(Base3dMetric):
    """
    Computes the volume IoU for the two meshes.

    The function converts both meshes to an occupancy grid and computes the IoU
    on the grid values. The grid bounding box is defined by the ground truth mesh.
    """

    def _initialize(self, resolution: int = 128, margin: float = 0.05) -> None:
        """
        Args:
            resolution: The resolution for computing the occupancy grid for each mesh.
            margin: Increase the bounding box for the grid by this percentage for each axis.
        """
        self.resolution = resolution
        self.margin = margin

    def __call__(
        self, mesh: o3d.t.geometry.TriangleMesh, gt: o3d.t.geometry.TriangleMesh
    ):
        """
        Args:
            mesh: The predicted mesh
            gt: The ground truth mesh. This mesh defines the bounding box used in
                computing the occupancy grid.

        Returns:
            Float with the IoU value
        """
        mesh = align_meshes(mesh, gt)

        # compute grid using the gt bbox
        bb_min = gt.get_min_bound().numpy()
        bb_max = gt.get_max_bound().numpy()
        bb_len = bb_max - bb_min
        bb_min -= bb_len * self.margin / 2
        bb_max += bb_len * self.margin / 2
        xyzs = [
            np.linspace(bmin, bmax, num=self.resolution, dtype=np.float32)
            for bmin, bmax in zip(bb_min, bb_max)
        ]
        grid = np.stack(np.meshgrid(*xyzs), axis=-1)

        # compute occupancy grids
        mesh_scene = o3d.t.geometry.RaycastingScene()
        mesh_scene.add_triangles(mesh)
        mesh_occupancy = mesh_scene.compute_occupancy(grid, nsamples=3).numpy()

        gt_scene = o3d.t.geometry.RaycastingScene()
        gt_scene.add_triangles(gt)
        gt_occupancy = gt_scene.compute_occupancy(grid, nsamples=3).numpy()

        intersection = np.count_nonzero(np.logical_and(mesh_occupancy, gt_occupancy))
        union = np.count_nonzero(np.logical_or(mesh_occupancy, gt_occupancy))

        return intersection / union
