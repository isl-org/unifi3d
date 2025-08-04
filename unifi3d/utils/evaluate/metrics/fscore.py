import numpy as np
import open3d as o3d

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import align_meshes
from unifi3d.utils.evaluate.metrics.utils import point_to_mesh_distances


class FScore(Base3dMetric):
    """
    Computes the F-score for the mesh with respect to the ground truth.
    Computes the F-score as defined in the Tanks and Temples paper.
    """

    def _initialize(self, threshold: float = 0.05, num_points: int = 10000) -> None:
        """

        Args:
            threshold: The threshold as percentage of the diagonal of the axis
                aligned bounding box of the ground truth mesh.
            num_points: The number of points used for computing distances.
        """
        self.threshold = threshold
        self.num_points = num_points

    def __call__(
        self, mesh: o3d.t.geometry.TriangleMesh, gt: o3d.t.geometry.TriangleMesh
    ):
        """
        Args:
            mesh: The predicted mesh.
            gt: The ground truth mesh.
        Returns:
            The F-score as percentage in the range [0..100]
        """
        mesh = align_meshes(mesh, gt)

        bb_min = gt.get_min_bound().numpy()
        bb_max = gt.get_max_bound().numpy()
        bb_len = bb_max - bb_min
        diagonal = np.sqrt((bb_len**2).sum())
        t = self.threshold * diagonal

        mesh_legacy = mesh.to_legacy()
        gt_legacy = gt.to_legacy()
        mesh_points = mesh_legacy.sample_points_uniformly(self.num_points)
        gt_points = gt_legacy.sample_points_uniformly(self.num_points)
        d1 = point_to_mesh_distances(
            o3d.t.geometry.PointCloud.from_legacy(mesh_points), gt
        ).numpy()
        d2 = point_to_mesh_distances(
            o3d.t.geometry.PointCloud.from_legacy(gt_points), mesh
        ).numpy()
        precision = 100 * np.count_nonzero(d1 < t) / self.num_points
        recall = 100 * np.count_nonzero(d2 < t) / self.num_points
        fscore = 2 * precision * recall / (precision + recall)
        return fscore
