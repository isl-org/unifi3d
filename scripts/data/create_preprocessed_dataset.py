#!/usr/bin/env python3
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

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)

from unifi3d.data.data_iterators import ShapenetIterator, ObjaverseIterator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocesses the raw ShapeNetCore or Objaverse dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "output_dir", type=Path, help="Target directory for the preprocessed data"
    )
    parser.add_argument(
        "--dataset_root",
        default=Path(root_path) / "data" / "ShapeNetCore.v1",
        type=Path,
        help="Root directory of the ShapeNetCore.v1 dataset",
    )
    parser.add_argument(
        "--objaverse_json_path",
        type=Path,
        help="json file with metadata. This is only needed for Objaverse",
    )
    parser.add_argument(
        "--lvis_categories",
        type=str,
        nargs="+",
        help="List of categories to use. This is optional and only affects Objaverse",
    )
    # make the value of the env var the default but allow to override it
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id_kwargs = dict(default=int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        task_id_kwargs = dict(required=True)
    parser.add_argument(
        "--task_id",
        type=int,
        **task_id_kwargs,
        help="The task_id",
    )
    parser.add_argument(
        "--task_offset",
        type=int,
        default=0,
        help="The task offset that is added to the task_id",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1000")),
        help="The total number of tasks",
    )
    parser.add_argument(
        "--objs",
        type=Path,
        nargs="+",
        help="List of obj files to process. This bypasses the dataset iterator",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


def preprocess(item, output_dir, args, rng):
    """Function doing all the data processing. Expects the dict returned from the dataiterator"""
    import torch
    import open3d as o3d
    import ocnn
    from scipy.interpolate import RegularGridInterpolator
    from unifi3d.utils.data.sdf_utils import compute_sdf_fill, coord_grid_from_bbox
    from unifi3d.utils.data.convert_to_numpy import convert_to_numpy
    from unifi3d.utils.data.geometry import get_combined_geometry
    from unifi3d.utils.data.materials import get_material_attributes

    generated_files = []

    # read mesh, scale to unit cube, and convert to a simple numpy format
    file_path = Path(item["file_path"])
    mesh_scale = 0.8
    if file_path.suffix in (".glb", ".gltf"):
        scene = convert_to_numpy(
            file_path,
            target_size=mesh_scale,
        )
    else:
        scene = convert_to_numpy(
            file_path, target_size=mesh_scale, up_axis="Z", forward_axis="Y"
        )
    mesh = get_combined_geometry(scene)
    mesh["vertices"] = mesh["vertices"].astype(np.float32)
    mesh["triangles"] = mesh["triangles"].astype(np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files.append(output_dir / "mesh.npz")
    np.savez_compressed(
        output_dir / "mesh.npz",
        **mesh,
    )

    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = mesh["vertices"]
    tmesh.triangle.indices = mesh["triangles"]
    lmesh = tmesh.to_legacy()

    #
    # sample points from the surface following shapenet.sample_pts_from_mesh()
    #
    def sample_pts_from_mesh(mesh: o3d.geometry.TriangleMesh, scene: dict):
        num_samples = 40000
        filename_pts = output_dir / "pointcloud.npz"

        for i in range(1):
            # filename_pts = output_dir / "pointcloud" / f"pointcloud_{i:02d}.npz"
            pcd = mesh.sample_points_uniformly(num_samples, use_triangle_normal=True)
            points = np.asarray(pcd.points).astype(np.float32)
            normals = np.asarray(pcd.normals).astype(np.float32)
            try:
                materials = get_material_attributes(points, scene)
                for k, v in materials.items():
                    materials[k] = v.astype(np.float16)
            except Exception as e:
                print(f"Error in get_material_attributes: {e}")
                materials = {
                    "base_color": np.zeros(points.shape, dtype=np.float16),
                    "metallic": np.zeros((points.shape[0], 1), dtype=np.float16),
                    "roughness": np.zeros((points.shape[0], 1), dtype=np.float16),
                }

            # save points
            generated_files.append(filename_pts)
            np.savez_compressed(
                filename_pts,
                points=points.astype(np.float16),
                normals=normals.astype(np.float16),
                **materials,
            )
        return points

    points = sample_pts_from_mesh(tmesh.to_legacy(), scene)

    def sample_grid_sdf(mesh: dict, resolution: int = 128):
        sdf, _, boundary_faces = compute_sdf_fill(
            **mesh, resolution=resolution, return_boundary_faces=True
        )
        generated_files.append(output_dir / f"sdf_grid_{resolution}.npz")
        np.savez_compressed(
            output_dir / f"sdf_grid_{resolution}.npz",
            sdf=sdf.astype(np.float16),
        )
        return sdf, boundary_faces

    sample_grid_sdf(mesh, resolution=128)
    sdf_256, sdf256_boundary_faces = sample_grid_sdf(mesh, resolution=256)

    generated_files.append(output_dir / "mesh_boundary.npz")
    np.savez_compressed(
        output_dir / "mesh_boundary.npz",
        vertices=mesh["vertices"],
        triangles=mesh["triangles"][sdf256_boundary_faces],
    )

    def sample_octree_sdf(
        mesh: o3d.t.geometry.TriangleMesh, points: np.ndarray, sdf_grid: np.ndarray
    ):
        depth, full_depth = 6, 4
        grid_resolution = 2 ** (depth + 1)
        sample_num = 4  # number of samples in each octree node

        points = 2 * points  # rescale points to [-1, 1]

        coords = coord_grid_from_bbox(
            sdf_grid.shape[0], (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
        )
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf_grid,
            bounds_error=False,
            fill_value=1.0,
        )

        # build octree
        points = ocnn.octree.Points(torch.from_numpy(points))
        octree = ocnn.octree.init_octree(depth, full_depth)
        _ = octree.build_octree(points)

        xyzs = []
        for d in range(full_depth, depth + 1):
            xyz = torch.stack(octree.xyzb(d, nempty=False), dim=-1)

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

        coords = coord_grid_from_bbox(sdf.shape[0], (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
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
        mesh: dict, scene: dict, sdf: np.ndarray, boundary_faces: List[int]
    ):
        num_samples = 100000
        filename_pts = output_dir / "pointcloud_boundary.npz"

        tmesh = o3d.t.geometry.TriangleMesh()
        tmesh.vertex.positions = mesh["vertices"]
        tmesh.triangle.indices = mesh["triangles"][boundary_faces]
        lmesh = tmesh.to_legacy()

        coords = coord_grid_from_bbox(sdf.shape[0], (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
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
        try:
            materials = get_material_attributes(points, scene)
            for k, v in materials.items():
                materials[k] = v.astype(np.float16)
        except Exception as e:
            print(f"Error in get_material_attributes: {e}")
            materials = {
                "base_color": np.zeros(points.shape, dtype=np.float16),
                "metallic": np.zeros((points.shape[0], 1), dtype=np.float16),
                "roughness": np.zeros((points.shape[0], 1), dtype=np.float16),
            }

        # save points
        generated_files.append(filename_pts)
        np.savez_compressed(
            filename_pts,
            points=points.astype(np.float16),
            normals=normals.astype(np.float16),
            **materials,
        )
        return points

    points_boundary = sample_pts_from_boundary(
        mesh, scene, sdf_256, sdf256_boundary_faces
    )

    def sample_octree_sdf2(
        mesh: o3d.t.geometry.TriangleMesh,
        points: np.ndarray,
        sdf_grid: np.ndarray,
        depth: int = 6,
        full_depth: int = 4,
    ):
        """Like sample_octree_sdf but computes the grad by interpolating the SDF"""
        grid_resolution = 2 ** (depth + 1)
        sample_num = 4  # number of samples in each octree node

        points = 2 * points  # rescale points to [-1, 1]

        coords = coord_grid_from_bbox(
            sdf_grid.shape[0], (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
        )
        voxel_size = (coords[1, 1, 1] - coords[0, 0, 0])[0]
        grid_interp = RegularGridInterpolator(
            (coords[:, 0, 0, 2], coords[0, :, 0, 1], coords[0, 0, :, 0]),
            sdf_grid,
            bounds_error=False,
            fill_value=1.0,
        )

        # build octree
        points = ocnn.octree.Points(torch.from_numpy(points))
        octree = ocnn.octree.init_octree(depth, full_depth)
        _ = octree.build_octree(points)

        xyzs = []
        for d in range(full_depth, depth + 1):
            xyz = torch.stack(octree.xyzb(d, nempty=False), dim=-1)

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

    sample_octree_sdf2(tmesh, points_boundary, sdf_256)
    sample_octree_sdf2(tmesh, points_boundary, sdf_256, 7)
    sample_octree_sdf2(tmesh, points_boundary, sdf_256, 8)

    generated_files.append(output_dir / "info.json")
    with open(output_dir / "info.json", "w") as f:
        json.dump(item, f, indent=2)

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

    task_id = args.task_id
    task_id += args.task_offset

    if args.objs is None:
        if args.objaverse_json_path:
            dataiter = ObjaverseIterator(
                args.dataset_root.as_posix(),
                json_path=args.objaverse_json_path,
                categories=args.lvis_categories,
            )
        else:
            # default to ShapenetIterator
            dataiter = ShapenetIterator(args.dataset_root.as_posix(), mode="all")
    else:
        dataiter = [{"file_path": obj.as_posix(), "id": obj.stem} for obj in args.objs]
    print(f"{len(dataiter)=}")
    tasks = np.arange(len(dataiter))

    my_task = np.array_split(tasks, args.num_tasks)[task_id]

    for idx in tqdm(my_task):
        rng = np.random.default_rng(idx)
        item = dataiter[idx]
        file_path = Path(item["file_path"])
        if isinstance(dataiter, ObjaverseIterator):
            if "category" in item:
                output_dir = args.output_dir / item["category"] / item["id"]
            else:
                output_dir = args.output_dir / file_path.parent.name / item["id"]
        else:
            output_dir = args.output_dir / file_path.parent.parent.name / item["id"]
        if output_dir.exists():
            print(f"skipping {str(output_dir)} because it already exists")
            continue
        try:
            generated_files = preprocess(dataiter[idx], output_dir, args, rng)
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
