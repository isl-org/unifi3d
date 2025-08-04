import torch
import numpy as np
import open3d as o3d
import scipy.ndimage
import skimage
import rootutils
from typing import Optional, Tuple, Union

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)


def coord_grid_from_bbox(
    resolution: Union[int, Tuple[int, int, int]],
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    dimension: int = 3,
):
    """Creates a grid of coordinates from a bounding box
    Args:
        resolution: The resolution of the grid for the x,y,z axis. This is
            either a scalar or a 3-tuple
        bbox_min: (3,) bounding box min values
        bbox_max: (3,) bounding box max values
    Returns:
        numpy array with the coordinates. Note that the coordinates are in the
        format (x,y,z) but the memory order is [z,y,x].
    """
    if isinstance(resolution, int):
        resolution = np.array(dimension * [resolution], dtype=np.int64)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    voxel_size = (bbox_max - bbox_min) / resolution
    bbox_min += 0.5 * voxel_size
    bbox_max -= 0.5 * voxel_size
    coords = [
        np.linspace(bbox_min[i], bbox_max[i], num=resolution[i])
        for i in range(dimension - 1, -1, -1)
    ]
    grid = np.meshgrid(
        *coords,
        indexing="ij",
    )
    grid = np.stack(grid, axis=-1)[..., ::-1].astype(np.float32)
    return grid


def volume_to_mesh(
    volume: np.ndarray,
    level: float = 0,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
    smaller_values_are_inside: bool = False,
):
    """Create a mesh from a 3D scalar field

    If a bounding box is provided along with the volume the mesh will be scaled
    and translated accordingly.

    Args:
        volume: A 3D array with memory order [z,y,x].
        level: The value at which to extract the contour
        bbox_min: Optional (3,) bounding box min values
        bbox_max: Optional (3,) bounding box max values
        smaller_values_are_inside: Set this to True if values smaller than level
            are considered to be inside the shape. For occupancy fields this is
            usually False. For SDFs with negative inside values this should be True.
    Returns:
        The mesh as o3d.t.geometry.TriangleMesh
    """
    assert not ((bbox_min is None) ^ (bbox_max is None))

    surface = o3d.t.geometry.TriangleMesh.create_isosurfaces(volume, [level])
    if surface.vertex.positions.shape[0]:
        # shift by +0.5 because the voxel center is not a corner of the bounding box
        surface.vertex.positions += 0.5
        if smaller_values_are_inside:
            surface.triangle.indices = surface.triangle.indices[:, [2, 1, 0]]
        if bbox_min is not None:
            bbox_min = np.asarray(bbox_min, dtype=np.float32)
            bbox_max = np.asarray(bbox_max, dtype=np.float32)
            resolution = np.array(volume.shape[::-1])
            voxel_size = ((bbox_max - bbox_min) / resolution).astype(np.float32)
            surface.vertex.positions = surface.vertex.positions * voxel_size + bbox_min

    return surface


def sdf_to_mesh(
    sdf,
    box_size=1,
    level=0,
    padding=0,
    smaller_values_are_inside=None,
    remove_floaters=True,
):
    """Extract meshes from SDF"""
    if smaller_values_are_inside is None:
        print(
            """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!                                                                        !!!!
!!!! Please set the smaller_values_are_inside argument for your code        !!!!
!!!! when calling                                                           !!!!
!!!!  unifi3d/utils/data/sdf_utils.py:sdf_to_mesh()                       !!!!
!!!!                                                                        !!!!
!!!! For SDFs with negative values defining the inside of a shape use       !!!!
!!!!  "smaller_values_are_inside = True"                                    !!!!
!!!!                                                                        !!!!
!!!! For density/occupancy fields use                                       !!!!
!!!!  "smaller_values_are_inside = False"                                   !!!!
!!!!                                                                        !!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        )
        # keep the original default
        smaller_values_are_inside = False

    box_size = box_size + padding
    bbox_min = 3 * (-0.5 * box_size,)
    bbox_max = 3 * (0.5 * box_size,)
    p3d_mesh = volume_to_mesh(
        sdf,
        level=level,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        smaller_values_are_inside=smaller_values_are_inside,
    ).to_legacy()

    # Optionally, compute the normals
    p3d_mesh.compute_vertex_normals()

    if remove_floaters:
        # Remove isolated triangle artifacts from the marching cube
        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = p3d_mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 50
        p3d_mesh.remove_triangles_by_mask(triangles_to_remove)

    return p3d_mesh


def generate_query_points(grid_resolution: int = 64, padding: float = 0.1, dimension=3):
    """
    Generates a grid of query points for Signed Distance Field (SDF) computation.

    This function creates a uniform 3D grid of points within a cubic volume. The grid
    is used to query the SDF values at each point. The cubic volume is defined to
    have a side length slightly larger than 1, depending on the padding, to ensure
    it encompasses the unit cube centered at the origin.

    Args:
        grid_resolution (int): The number of points along each axis of the grid.
                               Higher values result in a denser grid.
        padding (float): Additional space added to each side of the unit cube to
                         ensure the grid covers the entire region of interest.
        dim (int): Dimension of the querry points
    Returns:
        numpy.ndarray: A numpy array of shape (N, 3), where N is the total number
                       of points in the grid (grid_resolution^3). Each row contains
                       the (x, y, z) coordinates of a point.
    """
    # Calculate the total size of the box accounting for padding
    box_size = 1 + padding
    bbox_min = tuple(-0.5 * box_size for _ in range(dimension))
    bbox_max = tuple(0.5 * box_size for _ in range(dimension))

    # bbox_min = 3 * (-0.5 * box_size,)
    # bbox_max = 3 * (0.5 * box_size,)
    points = coord_grid_from_bbox(
        resolution=grid_resolution,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        dimension=dimension,
    )
    points = np.reshape(points, (-1, dimension))
    return points


def center_scale_mesh(mesh: o3d.geometry.TriangleMesh):
    """
    Takes a mesh and centers and scales it to fit into the cube spanned by [-0.5, -0.5, -0.5] and [0.5, 0.5, 0.5]
    """
    # Centering
    mesh_vertices = np.asarray(mesh.vertices)
    center = (
        np.max(mesh_vertices, axis=0) - np.min(mesh_vertices, axis=0)
    ) / 2 + np.min(mesh_vertices, axis=0)
    mesh = mesh.translate(-center, relative=True)

    # scale mesh to box between -0.5 and 0.5
    mesh_vertices = np.asarray(mesh.vertices)
    min_bound = np.min(mesh_vertices, axis=0)
    max_bound = np.max(mesh_vertices, axis=0)
    scale_factor = 1.0 / np.max(max_bound - min_bound)
    mesh.scale(scale_factor, center=np.zeros(3))
    return mesh


def compute_sdf_naive(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = 128,
    bbox_min=(-0.5, -0.5, -0.5),
    bbox_max=(0.5, 0.5, 0.5),
):
    """Computes the signed distance field.
    The sign is determined by counting ray intersections.

    Args:
        vertices: (n,3) numpy array with float type
        triangles: (n,3) numpy array with int type
        resolution: Grid resolution
        bbox_min: (3,) bounding box min values
        bbox_max: (3,) bounding box max values

    Returns:
        Tuple with (SDF, voxel size)
    """
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(
        vertices.astype(np.float32), triangles.astype(np.uint32)
    )

    grid = coord_grid_from_bbox(resolution, bbox_min, bbox_max)

    sdf = scene.compute_signed_distance(grid).numpy()
    voxel_size = grid[1, 1, 1] - grid[0, 0, 0]
    return sdf, voxel_size


def compute_sdf_fill(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = 128,
    bbox_min=(-0.5, -0.5, -0.5),
    bbox_max=(0.5, 0.5, 0.5),
    seed_point=(0, 0, 0),
    return_boundary_faces: bool = False,
):
    """Computes the signed distance field to the surface.
    The sign is determined by filling the outside volume starting from a seed
    point. This function does not need correcly oriented faces.

    Args:
        vertices: (n,3) numpy array with float type
        triangles: (n,3) numpy array with int type
        resolution: Grid resolution
        bbox_min: (3,) bounding box min values
        bbox_max: (3,) bounding box max values
        seed_point: The seed point as voxel index (3,)
        return_boundary_faces: If True, the function will return the faces that
            touch the zero level set.

    Returns:
        Tuple with (SDF, voxel size) or (SDF, voxel size, boundary_faces) if
        return_boundary_faces is True.
    """
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(
        vertices.astype(np.float32), triangles.astype(np.uint32)
    )

    grid = coord_grid_from_bbox(resolution, bbox_min, bbox_max)

    voxel_size = grid[1, 1, 1] - grid[0, 0, 0]

    closest_points = {
        k: v.numpy()
        for k, v in scene.compute_closest_points(
            grid,
        ).items()
    }
    diffvec = grid - closest_points["points"]

    vol = np.zeros(grid.shape[:-1], dtype=np.int8)
    # set voxels intersecting with a triangle to 1
    vol[np.all(np.abs(diffvec) <= 0.5 * voxel_size, axis=-1)] = 1

    # set value 2 for voxels outside
    vol = skimage.segmentation.flood_fill(
        vol, seed_point=seed_point, connectivity=1, new_value=2
    )
    # estimate a vector pointing outside by looking at the voxels labelled with '2'
    dirs = (
        np.stack(np.unravel_index(np.arange(27), (3, 3, 3)), axis=-1).reshape(
            3, 3, 3, -1
        )[..., ::-1]
        - 1
    )
    dirs = dirs.astype(np.int8)
    outside_voxels = (vol == 2).astype(np.int8)
    outside_dir = np.stack(
        [scipy.ndimage.correlate(outside_voxels, dirs[..., i]) for i in range(3)],
        axis=-1,
    )

    border = np.logical_and(
        scipy.ndimage.correlate(outside_voxels, np.ones((3, 3, 3), dtype=np.int8)) > 0,
        vol == 1,
    )
    sign = np.sign((outside_dir[border] * diffvec[border]).sum(axis=-1))
    # define which border voxels are outside by adding the sign
    vol[border] += sign.astype(np.int8)

    inside = vol != 2

    sdf = np.linalg.norm(diffvec, axis=-1)

    # compute inside distances. Consider only triangles intersecting the border voxels
    select_face_idx = list(set(closest_points["primitive_ids"][border]))
    # selected_faces = triangles[select_face_idx]
    # scene = o3d.t.geometry.RaycastingScene()
    # geom_id = scene.add_triangles(vertices.astype(np.float32), selected_faces.astype(np.uint32))
    # closest_points = {
    #     k: v.numpy()
    #     for k, v in scene.compute_closest_points(
    #         grid[inside],
    #     ).items()
    # }
    # diffvec = grid[inside] - closest_points["points"]
    # dist_inside = -np.linalg.norm(diffvec, axis=-1)

    # use the border points to compute the inside distances which is more stable
    # than using the triangles touching the border
    border_points = closest_points["points"][border]
    nns = o3d.core.nns.NearestNeighborSearch(border_points)
    nns.knn_index()
    dist_inside = -nns.knn_search(grid[inside], knn=1)[1].sqrt().numpy()[:, 0]
    sdf[inside] = dist_inside
    if return_boundary_faces:
        return sdf, voxel_size, select_face_idx
    else:
        return sdf, voxel_size


def mesh_to_sdf(
    mesh: o3d.geometry.TriangleMesh,
    sample_res: int,
    grid_resolution: int,
    center_mesh: bool = False,
):
    if center_mesh:
        mesh = center_scale_mesh(mesh)

    sdf, _ = compute_sdf_fill(
        vertices=np.asarray(mesh.vertices).astype(np.float32),
        triangles=np.asarray(mesh.triangles).astype(np.float32),
        resolution=sample_res,
    )
    sdf = torch.from_numpy(sdf).unsqueeze(0).float()
    sdf = downsample_grid_sdf(sdf, grid_resolution, sample_res)

    return sdf


def downsample_grid_sdf(sdf, grid_resolution, sample_res):
    """
    Downsample a grid sdf to a lower resolution
    """
    if sample_res == grid_resolution:
        return sdf
    # downsample to desired resolution
    sdf = torch.nn.functional.interpolate(
        torch.unsqueeze(sdf, 0), (grid_resolution, grid_resolution, grid_resolution)
    )[0]
    return sdf


def get_unoriented_bb(vertices: np.ndarray):
    """compute center and extend of an unoriented bb"""
    min_verts = np.min(vertices, axis=0)
    size = (np.max(vertices, axis=0) - min_verts) / 2
    center = min_verts + size
    return center, size


def mesh2sdf_manifold_mesh(mesh: o3d.geometry.TriangleMesh, size: int, level: float):
    """Create a new mesh that is manifold in the same way as mesh2sdf (but faster)

    This function assumes that the mesh is inside the bounds [-1,1].

    Args:
        mesh: Input mesh to fix
        size: The resolution of the volume
        level: The threshold for creating the levelsets

    Returns:
        The fixed mesh.
    """
    bbox_min = np.array([-1, -1, -1], dtype=np.float32)
    bbox_max = np.array([1, 1, 1], dtype=np.float32)
    voxel_size = (bbox_max - bbox_min) / size
    bbox_min += 0.5 * voxel_size
    bbox_max -= 0.5 * voxel_size
    grid = np.meshgrid(
        np.linspace(bbox_min[0], bbox_max[0], num=size),
        np.linspace(bbox_min[1], bbox_max[1], num=size),
        np.linspace(bbox_min[2], bbox_max[2], num=size),
        indexing="ij",
    )
    grid = np.stack(grid[::-1], axis=-1).astype(np.float32)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    distances = scene.compute_distance(grid)
    surface = o3d.t.geometry.TriangleMesh.create_isosurfaces(distances, [level])
    surface.vertex.positions = (surface.vertex.positions + 0.5) * voxel_size - 1
    surface_legacy = surface.to_legacy()

    triangle_clusters, cluster_n_triangles, cluster_area = (
        surface_legacy.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    surface_legacy.remove_triangles_by_mask(triangles_to_remove)

    tris = np.asarray(surface_legacy.triangles)
    tris[...] = tris[:, [2, 1, 0]]
    return surface_legacy
