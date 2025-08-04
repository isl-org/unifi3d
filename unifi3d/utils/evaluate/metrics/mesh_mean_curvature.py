import igl
import numpy as np
import open3d as o3d
import scipy as sp

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric


class MeshMeanCurvature(Base3dMetric):
    """
    Computes the mean curvature of the mesh for each vertex.
    For details see https://libigl.github.io/libigl-python-bindings/tut-chapter1/#curvature-directions
    """

    def __call__(self, mesh: o3d.t.geometry.TriangleMesh):
        """
        Args:
            mesh: The mesh.

        Returns:
            The mean curvature at each vertex as vector with shape (N,)
        """
        v, f = mesh.vertex.positions.numpy(), mesh.triangle.indices.numpy()
        l = igl.cotmatrix(v, f)
        m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        minv = sp.sparse.diags(1 / m.diagonal())
        hn = -minv.dot(l.dot(v))
        h = np.linalg.norm(hn, axis=1)
        return np.mean(h[np.isfinite(h)])
