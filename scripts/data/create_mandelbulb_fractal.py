#!/usr/bin/env python3
"""Create data for the mandelbulb fractal"""
from typing import List
import os
import sys
import argparse
from pathlib import Path
import shutil
import json
import numpy as np
from tqdm import tqdm
import traceback
import rootutils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create data for the mandelbulb fractal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "output_dir", type=Path, help="Target directory for the preprocessed data"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


def mandelbulb_distance_estimate(
    positions, max_iter: int = 10, bail_out: float = 2.0, power: int = 8
):
    """Mandelbulb distance estimator.
    Args:
        positions: np.ndarray of shape (N, 3). The (x, y, z) coordinates of N points.
        max_iter: Maximum number of iterations.
        bail_out: Bailout radius for escape.
        power: Power used in the Mandelbulb formula (commonly 8).

    Returns:
        Distance estimates for each input point with shape (N,)
    """
    # Ensure positions is a NumPy array
    positions = np.asarray(positions, dtype=np.float64)
    N = positions.shape[0]

    # z will be iterated, c is the constant "seed"
    z = positions.copy()
    c = positions

    # dr tracks an approximation to the derivative of the escape radius
    dr = np.ones(N, dtype=np.float64)
    # r will be the norm of z
    r = np.zeros(N, dtype=np.float64)

    for _ in range(max_iter):
        # Compute radius for each point
        r = np.linalg.norm(z, axis=1)

        # Find points that have not escaped yet
        mask = r < bail_out
        if not np.any(mask):
            # All points have escaped; no need to continue
            break

        # --- Convert to spherical for masked points ---
        # Only update those that haven't escaped
        x = z[mask, 0]
        y = z[mask, 1]
        zval = z[mask, 2]
        r_masked = r[mask]

        # theta = atan2(y, x)
        # phi   = arccos(z / r)
        theta = np.arctan2(y, x)
        phi = np.arccos(
            np.clip(zval / r_masked, -1.0, 1.0)
        )  # clip to avoid float issues

        # Update derivative
        dr[mask] = power * (r_masked ** (power - 1)) * dr[mask] + 1.0

        # r^power
        rp = r_masked**power

        # Multiply angles by power
        theta_p = theta * power
        phi_p = phi * power

        # Convert back to Cartesian
        sin_phi_p = np.sin(phi_p)
        cos_phi_p = np.cos(phi_p)
        xnew = rp * sin_phi_p * np.cos(theta_p)
        ynew = rp * sin_phi_p * np.sin(theta_p)
        znew = rp * cos_phi_p

        # Update z with c added
        z[mask, 0] = xnew + c[mask, 0]
        z[mask, 1] = ynew + c[mask, 1]
        z[mask, 2] = znew + c[mask, 2]

    # Final radius after last iteration
    r = np.linalg.norm(z, axis=1)

    # Avoid log(0) by clamping r away from zero
    r_safe = np.maximum(r, 1e-14)

    # Distance estimate: 0.5 * log(r) * r / dr
    dist_est = 0.5 * np.log(r_safe) * r_safe / dr

    return dist_est


def preprocess(max_iter, output_dir, args, rng):
    """Function doing all the data processing. Expects the dict returned from the dataiterator"""
    import torch
    import open3d as o3d
    import ocnn
    from scipy.interpolate import RegularGridInterpolator
    from unifi3d.utils.data.sdf_utils import coord_grid_from_bbox, sdf_to_mesh

    generated_files = []

    output_dir.mkdir(parents=True, exist_ok=True)

    def sample_grid_sdf(resolution: int = 128):
        grid = coord_grid_from_bbox(
            resolution=resolution,
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
            dimension=3,
        )
        # scale with 3 for computing the distances
        sdf = mandelbulb_distance_estimate(
            3 * grid.reshape(-1, 3), max_iter=max_iter
        ).reshape(3 * [resolution])

        generated_files.append(output_dir / f"sdf_grid_{resolution}.npz")
        np.savez_compressed(
            output_dir / f"sdf_grid_{resolution}.npz",
            sdf=sdf.astype(np.float16),
        )
        return sdf, None

    sample_grid_sdf(resolution=128)
    sdf_256, _ = sample_grid_sdf(resolution=256)

    lmesh = sdf_to_mesh(sdf_256, smaller_values_are_inside=True, remove_floaters=False)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(lmesh)

    mesh = {
        "vertices": tmesh.vertex.positions.numpy(),
        "triangles": tmesh.triangle.indices.numpy(),
    }

    generated_files.append(output_dir / "mesh.npz")
    np.savez_compressed(
        output_dir / "mesh.npz",
        **mesh,
    )

    # all faces are boundary faces
    sdf256_boundary_faces = np.arange(mesh["triangles"].shape[0])

    #
    # sample points from the surface following shapenet.sample_pts_from_mesh()
    #
    def sample_pts_from_mesh(mesh: o3d.geometry.TriangleMesh):
        num_samples = 40000
        filename_pts = output_dir / "pointcloud.npz"

        for i in range(1):
            # filename_pts = output_dir / "pointcloud" / f"pointcloud_{i:02d}.npz"
            pcd = mesh.sample_points_uniformly(num_samples, use_triangle_normal=True)
            points = np.asarray(pcd.points).astype(np.float32)
            normals = np.asarray(pcd.normals).astype(np.float32)

            # save points
            generated_files.append(filename_pts)
            np.savez_compressed(
                filename_pts,
                points=points.astype(np.float16),
                normals=normals.astype(np.float16),
            )
        return points

    points = sample_pts_from_mesh(tmesh.to_legacy())

    def sample_octree_sdf(
        mesh: o3d.t.geometry.TriangleMesh, points: np.ndarray, sdf_grid: np.ndarray
    ):
        depth, full_depth = 6, 4
        grid_resolution = 2 ** (depth + 1)
        sample_num = 4  # number of samples in each octree node

        points = 2 * points  # rescale points to [-1, 1]

        coords = coord_grid_from_bbox(
            resolution=sdf_grid.shape[0],
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
            dimension=3,
        )
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf_grid,
            bounds_error=False,
            fill_value=1.0,
        )

        # build octree
        points = ocnn.points_new(
            torch.from_numpy(points),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
        octree2points = ocnn.Points2Octree(depth=depth, full_depth=full_depth)
        octree = octree2points(points)

        xyzs = []
        for d in range(full_depth, depth + 1):
            xyz = ocnn.octree_property(octree, "xyz", d)
            xyz = ocnn.octree_decode_key(xyz)

            # sample k points in each octree node
            xyz = xyz[:, :3].float()  # + 0.5 -> octree node center
            xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
            xyz = xyz.view(-1, 3)  # (N, 3)
            xyz = xyz * (grid_resolution / 2**d)  # normalize to [0, 2^sdf_depth]
            xyz = xyz[
                (xyz < (grid_resolution - 1)).all(dim=1)
            ]  # remove out-of-bound points
            xyzs.append(xyz)

        # concat xyzs
        xyzs = torch.cat(xyzs, dim=0).numpy()
        points = (xyzs / (grid_resolution / 2) - 1) * 0.5

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        sdf = scene.compute_distance(points).numpy()
        sdf_sign = grid_interp(points[:, [2, 1, 0]])
        # copy sign from the grid which is more reliable
        sdf = np.copysign(sdf, sdf_sign)

        grad = points - scene.compute_closest_points(points)["points"].numpy()
        grad /= np.linalg.norm(grad, axis=-1, keepdims=True)

        generated_files.append(output_dir / f"sdf_octree_depth_{depth}.npz")
        np.savez_compressed(
            output_dir / f"sdf_octree_depth_{depth}.npz",
            points=points.astype(np.float16),
            grad=grad.astype(np.float16),
            sdf=sdf.astype(np.float16),
        )

    sample_octree_sdf(tmesh, points, sdf_256)

    def sample_occu(sdf: np.ndarray, min_inside_ratio: float = 0.0):
        num_samples = 100000
        if min_inside_ratio > 0:
            filename_occu = output_dir / f"points_iou_{min_inside_ratio}.npz"
        else:
            filename_occu = output_dir / "points_iou.npz"

        coords = coord_grid_from_bbox(
            resolution=sdf.shape[0],
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
            dimension=3,
        )
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf,
            bounds_error=False,
            fill_value=1.0,
        )

        outside_points = []
        inside_points = []
        num_occupied_points = 0
        count = 0
        while num_occupied_points < num_samples * min_inside_ratio + 1 and count < 1000:
            points = rng.uniform(low=-0.5, high=0.5, size=(num_samples, 3)).astype(
                np.float32
            )
            sdf_values = grid_interp(points[:, [2, 1, 0]])
            occu = sdf_values < 0
            inside_points.append(points[occu])
            if sum([x.shape[0] for x in outside_points]) < num_samples:
                outside_points.append(points[~occu])
            num_occupied_points += inside_points[-1].shape[0]
            count += 1

        points = np.concatenate(inside_points + outside_points, axis=0)[:num_samples]
        idx = np.arange(points.shape[0])
        occu = idx < num_occupied_points
        # shuffle points and occu
        idx = rng.permutation(points.shape[0])
        occu = occu[idx]
        points = points[idx]

        occu = np.packbits(occu)

        generated_files.append(filename_occu)
        np.savez(filename_occu, points=points.astype(np.float16), occupancies=occu)

    sample_occu(sdf_256)
    sample_occu(sdf_256, min_inside_ratio=0.1)

    def sample_pts_from_boundary(
        mesh: dict, sdf: np.ndarray, boundary_faces: List[int]
    ):
        num_samples = 100000
        filename_pts = output_dir / "pointcloud_boundary.npz"

        tmesh = o3d.t.geometry.TriangleMesh()
        tmesh.vertex.positions = mesh["vertices"]
        tmesh.triangle.indices = mesh["triangles"][boundary_faces]
        lmesh = tmesh.to_legacy()

        coords = coord_grid_from_bbox(
            resolution=sdf.shape[0],
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
            dimension=3,
        )
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf,
            bounds_error=False,
            fill_value=1.0,
        )

        boundary_points = []
        boundary_normals = []

        num_points_on_boundary = 0
        while num_points_on_boundary < num_samples:
            pcd = lmesh.sample_points_uniformly(num_samples, use_triangle_normal=True)
            points = np.asarray(pcd.points).astype(np.float32)
            normals = np.asarray(pcd.normals).astype(np.float32)

            threshold = -0.5 / sdf.shape[0]
            sdf_values = grid_interp(points[:, [2, 1, 0]])
            mask = sdf_values > threshold
            boundary_points.append(points[mask])
            boundary_normals.append(normals[mask])
            num_points_on_boundary += boundary_points[-1].shape[0]

        points = np.concatenate(boundary_points, axis=0)[:num_samples]
        normals = np.concatenate(boundary_normals, axis=0)[:num_samples]

        # save points
        generated_files.append(filename_pts)
        np.savez_compressed(
            filename_pts,
            points=points.astype(np.float16),
            normals=normals.astype(np.float16),
        )
        return points

    points_boundary = sample_pts_from_boundary(mesh, sdf_256, sdf256_boundary_faces)

    def sample_octree_sdf(
        mesh: o3d.t.geometry.TriangleMesh,
        points: np.ndarray,
        sdf_grid: np.ndarray,
        depth: int = 6,
        full_depth: int = 4,
    ):
        grid_resolution = 2 ** (depth + 1)
        sample_num = 4  # number of samples in each octree node

        points = 2 * points  # rescale points to [-1, 1]

        coords = coord_grid_from_bbox(
            resolution=sdf_grid.shape[0],
            bbox_min=(-0.5, -0.5, -0.5),
            bbox_max=(0.5, 0.5, 0.5),
            dimension=3,
        )
        voxel_size = (coords[1, 1, 1] - coords[0, 0, 0])[0]
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf_grid,
            bounds_error=False,
            fill_value=1.0,
        )

        # build octree
        points = ocnn.points_new(
            torch.from_numpy(points),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )
        octree2points = ocnn.Points2Octree(depth=depth, full_depth=full_depth)
        octree = octree2points(points)

        xyzs = []
        for d in range(full_depth, depth + 1):
            xyz = ocnn.octree_property(octree, "xyz", d)
            xyz = ocnn.octree_decode_key(xyz)

            # sample k points in each octree node
            xyz = xyz[:, :3].float()  # + 0.5 -> octree node center
            xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
            xyz = xyz.view(-1, 3)  # (N, 3)
            xyz = xyz * (grid_resolution / 2**d)  # normalize to [0, 2^sdf_depth]
            xyz = xyz[
                (xyz < (grid_resolution - 1)).all(dim=1)
            ]  # remove out-of-bound points
            xyzs.append(xyz)

        # concat xyzs
        xyzs = torch.cat(xyzs, dim=0).numpy()
        points = (xyzs / (grid_resolution / 2) - 1) * 0.5

        sdf = grid_interp(points[:, [2, 1, 0]])
        grad_x = grid_interp(points[:, [2, 1, 0]] + np.array([0, 0, voxel_size])) - sdf
        grad_y = grid_interp(points[:, [2, 1, 0]] + np.array([0, voxel_size, 0])) - sdf
        grad_z = grid_interp(points[:, [2, 1, 0]] + np.array([voxel_size, 0, 0])) - sdf
        grad = np.stack([grad_x, grad_y, grad_z], axis=-1)

        grad_norm = np.maximum(1e-6, np.linalg.norm(grad, axis=-1, keepdims=True))
        grad /= grad_norm

        generated_files.append(output_dir / f"sdf_octree_depth_{depth}.npz")
        np.savez_compressed(
            output_dir / f"sdf_octree_depth_{depth}.npz",
            points=points.astype(np.float16),
            grad=grad.astype(np.float16),
            sdf=sdf.astype(np.float16),
        )

    sample_octree_sdf(tmesh, points_boundary, sdf_256)
    sample_octree_sdf(tmesh, points_boundary, sdf_256, 7)
    sample_octree_sdf(tmesh, points_boundary, sdf_256, 8)

    with open(output_dir / "info.json", "w") as f:
        o3d.io.write_triangle_mesh(
            (output_dir / f"{output_dir.name}.obj").as_posix(),
            tmesh,
            write_triangle_uvs=False,
        )
        json.dump(
            dict(
                id=output_dir.name,
                file_path=(output_dir / f"{output_dir.name}.obj").as_posix(),
            ),
            f,
            indent=2,
        )

    return generated_files


def check_data(generated_files: list):
    result = True
    if not generated_files:
        return False
    for file in generated_files:
        try:
            if Path(file).suffix in (".npz",):
                data = np.load(file)
                for k, v in data.items():
                    nonfinite_count = np.count_nonzero(~np.isfinite(v))
                    if nonfinite_count:
                        print(f"{str(file)}:{k} has {nonfinite_count} nonfinite values")
                        result = False
            elif Path(file).suffix in (".json",):
                with open(file, "r") as f:
                    data = json.load(f)
        except Exception as e:
            result = False
            print(f"{str(file)}:{k} has unexpected error {e}")
    return result


def main():
    args = parse_args()
    print(args)

    for max_iter in (1, 2, 3):
        rng = np.random.default_rng(max_iter)
        output_dir = args.output_dir / f"mandelbulb_{max_iter}"
        if output_dir.exists():
            print(f"skipping {str(output_dir)} because it already exists")
            continue
        try:
            generated_files = preprocess(max_iter, output_dir, args, rng)
            if not check_data(generated_files):
                print(f"removing {str(output_dir)} because there were errors")
                shutil.rmtree(output_dir, ignore_errors=True)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print(f"removing {str(output_dir)} because there were errors")
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
