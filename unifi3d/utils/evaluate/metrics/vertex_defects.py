import numpy as np
import open3d as o3d
import scipy as sp
import trimesh

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric


class VertexDefects(Base3dMetric):
    """
    Compute the angle defect for each vertex
    Computes the 2pi - (sum of adjacent angles for all faces).
    """

    def __call__(self, mesh: o3d.t.geometry.TriangleMesh):
        """
        Args:
            mesh: Mesh

        Returns:
            Vector with the angle defects for all vertices.
        """
        vertices = mesh.vertex.positions.numpy()
        faces = mesh.triangle.indices.numpy()
        m = trimesh.Trimesh(vertices=vertices, faces=faces)

        col_indices = np.tile(np.arange(faces.shape[0]), (3, 1)).T.flatten()
        row_indices = faces.flatten()
        s = sp.sparse.coo_array(
            (m.face_angles.flatten(), (row_indices, col_indices)),
            shape=(vertices.shape[0], faces.shape[0]),
        )
        angle_sum = s.sum(axis=1)
        defect = (2 * np.pi) - angle_sum
        return defect
