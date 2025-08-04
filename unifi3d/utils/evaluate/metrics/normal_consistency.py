import numpy as np
import trimesh
from scipy.spatial import KDTree

from unifi3d.utils.evaluate.metrics.base3d_metric import Base3dMetric


class NormalConsistency(Base3dMetric):
    def __init__(self, num_points=10000):
        super().__init__(num_points=num_points)
        self.num_points = num_points

    def __call__(self, mesh_pred, mesh_gt):
        mesh_pred = mesh_pred.to_legacy()
        mesh_gt = mesh_gt.to_legacy()

        dist, normals_dot_product = self.normal_consistency(mesh_pred, mesh_gt)
        return np.mean(normals_dot_product)

    def sample_points_with_normals(self, mesh_o3d):
        """
        Sample points and the correspondng normals
        Problem: doing this with open3d is difficult because we cannot get the face idx
        when sampling points.
        """
        # Extract vertices and faces
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)

        # Create a Trimesh mesh using the extracted data
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # sample points
        pointcloud, idx = mesh_trimesh.sample(self.num_points, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh_trimesh.face_normals[idx]

        # # o3d way (incomplete)
        # mesh_pred_vertices = np.asarray(mesh_pred.vertices)
        # idx_pred = np.random.permutation(len(mesh_pred_vertices))[: self.num_points]
        # mesh_pred = mesh_pred.compute_triangle_normals()
        # normals_src = np.asarray(mesh_pred.triangle_normals)

        return pointcloud, normals

    def normal_consistency(self, mesh_pred, mesh_gt):
        points_src, normals_src = self.sample_points_with_normals(mesh_pred)
        points_tgt, normals_tgt = self.sample_points_with_normals(mesh_gt)

        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        dist, normals_dot_product = self.distance_p2p(
            points_src, normals_src, points_tgt, normals_tgt
        )
        return dist, normals_dot_product

    def distance_p2p(self, points_src, normals_src, points_tgt, normals_tgt):
        """Computes distances of normals between each point in points_src to
         and nearest point in points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        """
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)

        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
        return dist, normals_dot_product
