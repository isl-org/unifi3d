from typing import List, Literal
import open3d as o3d
import numpy as np


EXT_PNG = ".png"


def point_to_mesh_distances(
    points: o3d.t.geometry.PointCloud | np.ndarray, mesh: o3d.t.geometry.TriangleMesh
):
    """Computes the Euclidean distance of the point cloud to the mesh
    Args:
        points: o3d.t.geometry.PointCloud or numpy array with shape (N,3) and
            dtype np.float32
        mesh: Mesh
    Returns:
        The distance of each point to the mesh.
    """
    rt_mesh = o3d.t.geometry.RaycastingScene()
    rt_mesh.add_triangles(mesh)
    if isinstance(points, np.ndarray):
        ans = rt_mesh.compute_distance(points)
    else:
        ans = rt_mesh.compute_distance(points.point.positions)
    return ans


def align_meshes(
    source: o3d.t.geometry.TriangleMesh, target: o3d.t.geometry.TriangleMesh
):
    """Aligns the source mesh to the target using the ICP algorithm
    Args:
        source: The mesh that needs to be aligned
        target: The target mesh

    Returns:
        Returns the transformed source mesh minimizing the distance to the target.
    """
    source_legacy = source.to_legacy()
    target_legacy = target.to_legacy()

    source_points = source_legacy.sample_points_uniformly(10000)
    target_points = target_legacy.sample_points_uniformly(10000)
    init = np.eye(4)
    max_distance = 0.2 * (target.get_max_bound() - target.get_min_bound()).max().item()
    ans = o3d.pipelines.registration.registration_icp(
        source_points,
        target_points,
        max_distance,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
    )
    return source.transform(ans.transformation)
