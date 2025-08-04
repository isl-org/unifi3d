import open3d as o3d

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import align_meshes
from unifi3d.utils.evaluate.metrics.utils import point_to_mesh_distances


class ChamferDistance(Base3dMetric):
    """
    Computes the chamfer distance between the two meshes
    """

    def _initialize(self, num_points: int = 1000) -> None:
        """
        Args:
            num_points: The number of points used for computing distances
        """
        self.num_points = num_points

    def __call__(
        self, mesh1: o3d.t.geometry.TriangleMesh, mesh2: o3d.t.geometry.TriangleMesh
    ):
        """
        Args:
            mesh1: A mesh
            mesh2: Another mesh
        Returns:
            Float with the symmetric chamfer distance
        """
        mesh1 = align_meshes(mesh1, mesh2)

        mesh1_legacy = mesh1.to_legacy()
        mesh2_legacy = mesh2.to_legacy()
        mesh1_points = mesh1_legacy.sample_points_uniformly(self.num_points)
        mesh2_points = mesh2_legacy.sample_points_uniformly(self.num_points)
        d1 = point_to_mesh_distances(
            o3d.t.geometry.PointCloud.from_legacy(mesh1_points), mesh2
        ).mean()
        d2 = point_to_mesh_distances(
            o3d.t.geometry.PointCloud.from_legacy(mesh2_points), mesh1
        ).mean()
        return (d1 + d2).item()
