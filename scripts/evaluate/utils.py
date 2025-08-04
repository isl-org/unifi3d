import open3d as o3d
import numpy as np


def estimate_volume_from_sdf(
    sdf, bbox_min=np.zeros(3), bbox_max=np.array([1, 1, 1]), epsilon=None
):
    """
    Estimate the volume inside a non-watertight mesh using a Signed Distance Field (SDF).
    :param sdf: A 3D numpy array of shape (128, 128, 128) with signed distances.
                Negative values are inside the mesh, positive outside, ~0 on the surface.
    :param bbox_min: Tuple (x_min, y_min, z_min) specifying the bounding box min corner.
    :param bbox_max: Tuple (x_max, y_max, z_max) specifying the bounding box max corner.
    :param epsilon: float specifying the boundary threshold
    :return: Approximate volume of the mesh in the same units as the bounding box dimensions.
    """
    # SDF shape, typically (128, 128, 128)
    nx, ny, nz = sdf.shape

    # Compute voxel size based on the bounding box
    dx = (bbox_max[0] - bbox_min[0]) / nx
    dy = (bbox_max[1] - bbox_min[1]) / ny
    dz = (bbox_max[2] - bbox_min[2]) / nz
    voxel_volume = dx * dy * dz
    # Heuristic boundary threshold
    if epsilon is None:
        epsilon = 2 * max(dx, dy, dz)

    # Occupancy function
    occupancy = np.full_like(sdf, 0.5, dtype=np.float32)

    # Strongly inside => occupancy = 1
    strongly_inside_mask = sdf < -epsilon
    occupancy[strongly_inside_mask] = 1.0

    # Strongly outside => occupancy = 0
    strongly_outside_mask = sdf > epsilon
    occupancy[strongly_outside_mask] = 0.0

    # Boundary region => linear ramp from 1 to 0 across [-epsilon, epsilon]
    boundary_mask = ~(strongly_inside_mask | strongly_outside_mask)
    # SDF = -epsilon => occupancy=1
    # SDF = 0    => occupancy=0.5
    # SDF = epsilon => occupancy=0
    sdf_boundary = sdf[boundary_mask]
    occupancy[boundary_mask] = 1.0 - (sdf_boundary + epsilon) / (2 * epsilon)

    # Clip the occupancy to [0, 1] to avoid any overshoot or negative values
    np.clip(occupancy, 0.0, 1.0, out=occupancy)

    # Sum the occupancy => total fraction of inside voxels
    inside_fraction = occupancy.sum()

    # Multiply by voxel volume to get total volume
    volume_estimate = inside_fraction * voxel_volume
    return volume_estimate


def compute_surf2vol_ratio(mesh, sdf=None, is_o3d=False):
    """
    Computes the number of faces and the (surface area / volume) ratio
    for a watertight, properly oriented triangular mesh.
    :param mesh: A dictionary containing the fields "vertices" and "triangles".
    :return: (num_faces, surface_to_volume_ratio)
    """

    # Compute surface
    if is_o3d:
        o3d_mesh = mesh
    else:
        o3d_mesh = o3d.geometry.TriangleMesh()
        v = o3d.utility.Vector3dVector(mesh["vertices"])
        t = o3d.utility.Vector3iVector(mesh["triangles"])
        o3d_mesh.vertices = v
        o3d_mesh.triangles = t

    surface_area = o3d_mesh.get_surface_area()

    volume = estimate_volume_from_sdf(
        sdf, bbox_min=np.zeros(3), bbox_max=np.array([1, 1, 1]), epsilon=None
    )

    # Check for zero or near-zero volume to avoid division by zero
    if volume <= 1e-12:
        # If volume is zero, the mesh is likely non-watertight or degenerate.
        # Return None or float('inf'), depending on your preference.
        ratio = float("inf")
    else:
        ratio = surface_area / volume
    return volume, surface_area, ratio
