import os
import numpy as np
import torch
from einops import rearrange, reduce
from torchtyping import TensorType
from torch.nn.utils.rnn import pad_sequence
import open3d as o3d
import hydra


def get_unoriented_bb(vertices: np.ndarray):
    """compute center and extend of an unoriented bb"""
    min_verts = np.min(vertices, axis=0)
    size = (np.max(vertices, axis=0) - min_verts) / 2
    center = min_verts + size
    return center, size


def normalize_mesh(mesh: o3d.geometry.TriangleMesh, mesh_scale: float = 0.8):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    vertices = tmesh.vertex.positions.numpy()

    # rescale mesh to [-mesh_scale/2, mesh_scale/2]
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    scale = mesh_scale / (bbox_max - bbox_min).max()
    tmesh.vertex.positions = (vertices - center) * scale
    return tmesh


def load_normalize_mesh(
    mesh_path: str, mesh_scale: float = 0.8, to_legacy: bool = True
):
    """Load mesh from file and scale it to a cube with side length <mesh_scale>"""
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh_real = o3d.io.read_triangle_mesh(mesh_path)
    mesh_real = normalize_mesh(mesh_real, mesh_scale)
    if to_legacy:
        return mesh_real.to_legacy()
    return mesh_real


def load_npz_pcl(pcl_path: str):
    """Load a point cloud that is stored as an npz file"""
    pcl_npz = np.load(pcl_path)
    points = pcl_npz["points"]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def load_npz_mesh(mesh_path: str):
    """Load a mesh that is stored as an npz file"""
    mesh_npz = np.load(mesh_path)
    vertices = mesh_npz["vertices"]
    triangles = mesh_npz["triangles"]
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )


def save_o3d_mesh(mesh, path, name):
    o3d.io.write_triangle_mesh(f"{path}/{name}.obj", mesh)


def save_o3d_pointcloud(pcd, path, name):
    # Save the PointCloud as an OBJ file
    o3d.io.write_point_cloud(f"{path}/{name}.ply", pcd)


def save_np_pointcloud(points, path, name):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save the PointCloud as an OBJ file
    o3d.io.write_point_cloud(f"{path}/{name}.ply", pcd)


def scale_mesh_to_gt(
    mesh_pred: o3d.geometry.TriangleMesh, mesh_real: o3d.geometry.TriangleMesh
):
    """
    Scale predicted mesh to fit the bounding box of the real mesh
    """
    desired_center, desired_size = get_unoriented_bb(mesh_real.vertices)
    vertices = mesh_pred.vertices
    current_center, current_size = get_unoriented_bb(vertices)

    # put into canonical pose
    centered_vertices = vertices - current_center  # center
    vertices_canonical = centered_vertices / current_size  # scale

    # scale to desired pose
    vertices_desired = vertices_canonical * desired_size + desired_center

    # create new mesh
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_desired),
        o3d.utility.Vector3iVector(mesh_pred.triangles),
    )


def derive_face_edges_from_faces(
    faces: TensorType["b", "nf", 3, int],
    pad_id=-1,
    neighbor_if_share_one_vertex=False,
    include_self=True,
) -> TensorType["b", "e", 2, int]:

    is_one_face, device = faces.ndim == 2, faces.device

    if is_one_face:
        faces = rearrange(faces, "nf c -> 1 nf c")

    max_num_faces = faces.shape[1]
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2

    all_edges = torch.stack(
        torch.meshgrid(
            torch.arange(max_num_faces, device=device),
            torch.arange(max_num_faces, device=device),
            indexing="ij",
        ),
        dim=-1,
    )

    face_masks = reduce(faces != pad_id, "b nf c -> b nf", "all")
    face_edges_masks = rearrange(face_masks, "b i -> b i 1") & rearrange(
        face_masks, "b j -> b 1 j"
    )

    face_edges = []

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, "i c -> i 1 c 1") == rearrange(
            face, "j c -> 1 j 1 c"
        )
        num_shared_vertices = shared_vertices.any(dim=-1).sum(dim=-1)

        is_neighbor_face = (
            num_shared_vertices >= face_edges_vertices_threshold
        ) & face_edge_mask

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3

        face_edge = all_edges[is_neighbor_face]
        face_edges.append(face_edge)

    face_edges = pad_sequence(face_edges, padding_value=pad_id, batch_first=True)

    if is_one_face:
        face_edges = rearrange(face_edges, "1 e ij -> e ij")

    return face_edges


def load_reference_dataset(
    reference_dataset_cfg: str, num_samples: int = 100, return_files=False
):
    """Load reference dataset based on a dataset iterator configuration"""
    reference_dataset = hydra.utils.instantiate(reference_dataset_cfg)
    all_files = [f["mesh_path"] for f in reference_dataset]
    # sample <num_samples> meshes
    sampled_files = np.random.choice(all_files, size=num_samples, replace=False)
    print(
        "Loading reference set - Available:",
        len(all_files),
        ", Sampled:",
        len(sampled_files),
    )
    # load meshes and scale
    reference_meshes = [
        o3d.t.geometry.TriangleMesh.from_legacy(load_npz_mesh(path))
        for path in sampled_files
    ]
    # reference_meshes = [load_normalize_mesh(f, to_legacy=False) for f in sampled_files]
    if return_files:
        return sampled_files, reference_meshes
    return reference_meshes


def load_generated_dataset(path: str, limit=-1, from_legacy=True):
    """Load generated dataset from a directory"""
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".obj")]
    limit = len(all_files) if limit < 0 else limit
    generated_meshes = []
    for f in all_files[:limit]:
        mesh = o3d.io.read_triangle_mesh(f)
        if from_legacy:
            generated_meshes.append(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        else:
            generated_meshes.append(mesh)
    return generated_meshes


def remove_floaters(mesh: o3d.geometry.TriangleMesh):
    """Removes floaters by keeping only the largest connected component
    Args:
        mesh: The input mesh.

    Returns:
        A mesh containing only the largest connected components.
    """
    m = o3d.geometry.TriangleMesh(mesh)
    triangle_clusters, cluster_n_triangles, cluster_area = (
        m.cluster_connected_triangles()
    )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    m.remove_triangles_by_mask(triangles_to_remove)
    m.remove_unreferenced_vertices()
    return m
