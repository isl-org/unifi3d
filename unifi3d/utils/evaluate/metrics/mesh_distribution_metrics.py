from typing import List
import numpy as np
import open3d as o3d
from tqdm import tqdm

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric
from unifi3d.utils.evaluate.metrics.utils import point_to_mesh_distances


class MeshDistributionMetrics(Base3dMetric):
    r"""Computes metrics between two distributions defined by two sets of meshes

    The computed metrics are
        - COV (Coverage)
        - MMD (Minimum matching distance)
        - 1-NNA (1-nearest neighbor accuracy)

    The metrics between the generated set :math:`S_g` and the reference set :math:`S_r`
    are defined as
    .. math::
        \mathrm{COV}(S_g, S_r) = \frac{|\{\mathrm{arg\,min}_{Y\in S_r} D(X,Y) | X\in S_g\}|}{|S_r|}
        \mathrm{MMD}(S_g, S_r) = \frac{1}{|S_r|} \sum_{Y\in S_r}\underset{X\in S_g}{\mathrm{min}} D(X,Y)
        \text{1-NNA}(S_g, S_r) = \frac{\sum_{X\in S_g}\mathbb{I}[N_x\in S_g] + \sum_{Y\in S_r} \mathbb{I}[N_y \in S_r]}{|S_g|+|S_r|}

    :math:`N_X` is the nearest neighbor in the set :math:`S_r \cup S_g - \{X\}`

    The metrics use the Chamfer distance defined as
    .. math::
        D(X,Y) = \frac{1}{N}\sum_i^N | x_i - n(x_i,Y) | + \frac{1}{N}\sum_i^N | y_i - n(y_i,X) |

    Args:
        num_points: The number of points used for computing distances.
        show_progress: Show a progress bar
        return_matrices: Returns the distance matrix and the Chamfer distance
            matrix for debugging. Note that the diagonal of the Chamfer
            distance matrix is filled with 'inf'. The first len(reference)
            rows/cols correspond to the reference meshes.
    """

    def _initialize(
        self, num_points: int = 10000, show_progress=False, return_matrices=False
    ) -> None:
        self.num_points = num_points
        self.show_progress = show_progress
        self.return_matrices = return_matrices

    @staticmethod
    def compute_metrics_from_cd_matrix(cd_matrix: np.ndarray, r_size: int):
        """Computes all metrics from the symmetric Chamfer ditance matrix
        Args:
            cd_matrix: The matrix with symmetric chamfer distances.
                The first r_size rows/cols correspond to the reference set and
                the matrix must be symmetric. The values on the diagonal must
                be set to 'inf'.
            r_size: The size of the reference set
        Returns:
            A dictionary with the metrics
            - COV (Coverage)
            - MMD (Minimum matching distance)
            - 1-NNA (1-nearest neighbor accuracy)
        """
        g_size = cd_matrix.shape[0] - r_size
        cd_matrix_ref_gen = cd_matrix[:r_size, r_size:]
        COV = len(set(cd_matrix_ref_gen.argmin(axis=0))) / r_size

        mmd_vector = cd_matrix_ref_gen.min(axis=1)
        MMD = mmd_vector.mean()

        nearest = cd_matrix.argmin(axis=1)
        NNA = (
            (nearest[:r_size] < r_size).sum() + (nearest[r_size:] >= r_size).sum()
        ) / (r_size + g_size)
        return {
            "COV": COV,
            "MMD": MMD,
            "1-NNA": NNA,
            "MMD_vector": mmd_vector.tolist(),
        }

    def __call__(
        self,
        *,
        generated: List[o3d.t.geometry.TriangleMesh],
        reference: List[o3d.t.geometry.TriangleMesh],
    ):
        """
        Args:
            generated: The set of generated meshes.
            reference: The set of reference meshes.
        Returns:
            A dictionary with the metrics
            - COV (Coverage)
            - MMD (Minimum matching distance)
            - 1-NNA (1-nearest neighbor accuracy)
        """
        # compute all required distances avoiding recomputation
        r_size = len(reference)
        g_size = len(generated)
        distance_matrix = np.zeros(shape=2 * (r_size + g_size,))

        pointclouds = np.empty(
            shape=(r_size + g_size, self.num_points, 3), dtype=np.float32
        )

        all_meshes = reference + generated

        tqdm_opts = {"disable": not self.show_progress, "maxinterval": None}
        for i, m in enumerate(tqdm(all_meshes, **tqdm_opts)):
            p = m.to_legacy().sample_points_uniformly(self.num_points)
            pointclouds[i] = np.asarray(p.points)
        for i, m in enumerate(tqdm(all_meshes, **tqdm_opts)):
            distance_matrix[i, :] = (
                point_to_mesh_distances(pointclouds, m).numpy().mean(axis=-1)
            )
        np.fill_diagonal(distance_matrix, 0)

        # compute symmetric chamfer distance. Fill diagonal with inf to simplify
        # computing the metrics with min operations
        cd_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        np.fill_diagonal(cd_matrix, np.inf)

        result = self.compute_metrics_from_cd_matrix(cd_matrix, r_size)
        if self.return_matrices:
            result["distance_matrix"] = distance_matrix
            result["chamfer_distance_matrix"] = cd_matrix
        return result
