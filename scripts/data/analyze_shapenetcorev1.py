"""Script that collects geometric stats on shapenet objects"""

from typing import Tuple, Callable, List
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import fnmatch
import open3d as o3d
import rootutils

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)

from unifi3d.utils.data.sdf_utils import (
    compute_sdf_naive,
    compute_sdf_fill,
    volume_to_mesh,
)
from unifi3d.utils.data.convert_to_numpy import convert_to_numpy
from unifi3d.utils.data.geometry import get_combined_geometry
from unifi3d.utils.evaluate.metrics.utils import point_to_mesh_distances


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the ShapeNetCore.v1 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "output_dir", type=Path, help="Target directory for the generated json files"
    )
    parser.add_argument(
        "--dataset_root",
        default=Path(root_path) / "data" / "ShapeNetCore.v1",
        type=Path,
        help="Root directory of the ShapeNetCore.v1 dataset",
    )
    parser.add_argument(
        "--sdf_method",
        type=str,
        default="sdf_fill",
        choices=("sdf_naive", "sdf_fill", "mesh2sdf"),
        help="The method to use for converting to a grid SDF representation",
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
        "--mode",
        type=str,
        default="train",
        help="The mode for the dataset iterator",
    )

    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="The categories to process. If not given, all categories are processed. The categories for the paper were: airplane car chair",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


def fscore_and_cd(
    mesh: o3d.t.geometry.TriangleMesh,
    ground_truth_mesh: o3d.t.geometry.TriangleMesh,
    threshold: float | List[float],
    num_points: int = 100000,
):
    """Computes the F-score and Chamfer distance for the mesh with respect to the ground truth.

    Computes the F-score as defined in the Tanks and Temples paper and the symmetric Chamgfer distance.

    Args:
        mesh: The predicted mesh.
        gt: The ground truth mesh.
        threshold: The threshold for computing the F-score.
            For comparing meshes generated from SDFs the value should be chosen
            smaller than the voxel size. 0.25*voxel_size seems to work well.
            This can be a list of thresholds.
        num_points: The number of points used for computing distances.
    Returns:
        Returns a tuple (fscore, cd)
        The F-score is a percentage in the range [0..100]
    """
    if isinstance(threshold, (float, int)):
        threshold_list = [threshold]
    else:
        threshold_list = threshold

    if mesh.triangle.indices.shape[0] == 0:
        if isinstance(threshold, (float, int)):
            return 0.0, np.nan
        else:
            return len(threshold_list) * [0.0], np.nan

    gt = ground_truth_mesh

    mesh_legacy = mesh.to_legacy()
    gt_legacy = gt.to_legacy()
    mesh_points = mesh_legacy.sample_points_uniformly(num_points)
    gt_points = gt_legacy.sample_points_uniformly(num_points)

    d1 = point_to_mesh_distances(
        o3d.t.geometry.PointCloud.from_legacy(mesh_points), gt
    ).numpy()
    d2 = point_to_mesh_distances(
        o3d.t.geometry.PointCloud.from_legacy(gt_points), mesh
    ).numpy()

    fscores = []
    for t in threshold_list:
        precision = 100 * np.count_nonzero(d1 < t) / num_points
        recall = 100 * np.count_nonzero(d2 < t) / num_points
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        fscores.append(fscore)

    if isinstance(threshold, (float, int)):
        fscores = fscores[0]

    cd = 0.5 * (d1.mean() + d2.mean())
    return fscores, cd


def _analyze_mesh(
    path: Path, result: dict, sdf_method: str | Callable, blender: str = "blender"
):
    import sys

    if isinstance(sdf_method, str):
        sdf_method_dict = {
            "sdf_fill": compute_sdf_fill,
            "sdf_naive": compute_sdf_naive,
        }
        assert sdf_method in sdf_method_dict.keys()
        sdf_method = sdf_method_dict[sdf_method]

    print(" -", sys.executable)
    scene = convert_to_numpy(path, target_size=0.9, blender=blender)
    mesh = get_combined_geometry(scene)

    result["num_objects"] = len(fnmatch.filter(scene.keys(), f"*/vertices"))
    result["n_vertices"] = len(mesh["vertices"])
    result["n_triangles"] = len(mesh["triangles"])

    for param in ("base_color", "metallic", "roughness"):
        materials = fnmatch.filter(scene.keys(), f"*/{param}")
        tex_size = (0, 0, 0)
        for m in materials:
            if np.prod(scene[m].shape) > np.prod(tex_size):
                tex_size = scene[m].shape
        result[f"max_{param}_tex_size"] = tex_size

    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = mesh["vertices"].astype(np.float32)
    tmesh.triangle.indices = mesh["triangles"]
    lmesh = tmesh.to_legacy()

    result["surface_area"] = tmesh.compute_triangle_areas().triangle.areas.sum().item()
    result["is_watertight"] = lmesh.is_watertight()
    result["is_edge_manifold"] = lmesh.is_edge_manifold()
    result["is_vertex_manifold"] = lmesh.is_vertex_manifold()

    bbox = {"bbox_min": 3 * (-0.5,), "bbox_max": 3 * (0.5,)}
    for resolution in (64, 128, 256, 384):
        sdf, voxel_size = sdf_method(**mesh, resolution=resolution, **bbox)
        inside_ratio = np.count_nonzero(sdf < 0) / sdf.size
        result[f"voxel_inside_ratio_r{resolution}"] = inside_ratio
        m2 = volume_to_mesh(sdf, **bbox)
        thresholds = np.array((4, 2, 1, 0.5, 0.25, 0.125))
        fscores, cd = fscore_and_cd(m2, tmesh, threshold=thresholds * voxel_size[0])
        result[f"chamfer_distance_r{resolution}"] = cd
        for i, t in enumerate(thresholds):
            result[f"fscore_t{t}_r{resolution}"] = fscores[i]


def analyze_mesh(
    path: Path, sdf_method: str | Callable = "sdf_fill", blender: str = "blender"
):
    """Returns various stats for the mesh"""
    path = Path(path)
    result = {}

    try:
        _analyze_mesh(path, result, sdf_method, blender=blender)
    except Exception as e:
        print(e)

    return result


def mesh2sdf(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = 128,
    bbox_min=(-0.5, -0.5, -0.5),
    bbox_max=(0.5, 0.5, 0.5),
):
    assert bbox_min == (-0.5, -0.5, -0.5) and bbox_max == (0.5, 0.5, 0.5)
    from unifi3d.utils.data.sdf_utils import mesh2sdf_manifold_mesh
    import open3d as o3d

    verts = 2 * vertices

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(triangles)
    )
    manifold_mesh = mesh2sdf_manifold_mesh(mesh, resolution, level=2 / resolution)
    new_verts = 0.5 * np.asarray(manifold_mesh.vertices).astype(np.float32)
    new_tris = np.asarray(manifold_mesh.triangles)
    # o3d.io.rpc.set_mesh_data(
    #     path=f"mesh/original_{resolution}", vertices=vertices, faces=triangles
    # )
    # o3d.io.rpc.set_mesh_data(
    #     path=f"mesh/mesh2sdf_{resolution}", vertices=new_verts, faces=new_tris
    # )
    return compute_sdf_naive(
        new_verts, new_tris, resolution=resolution, bbox_min=bbox_min, bbox_max=bbox_max
    )


def preprocess(item, args):
    """Function doing all the data processing. Expects the dict returned from the dataiterator"""

    if args.sdf_method == "mesh2sdf":
        sdf_method = mesh2sdf
    else:
        sdf_method = args.sdf_method
    item.update(analyze_mesh(item["file_path"], sdf_method=sdf_method))
    return item


def main():
    args = parse_args()
    print(args)

    task_id = args.task_id
    task_id += args.task_offset
    num_tasks = args.num_tasks

    from unifi3d.data.data_iterators import ShapenetIterator

    dataiter = ShapenetIterator(
        args.dataset_root.as_posix(),
        mode=args.mode,
        categories=(
            ShapenetIterator.categories()
            if args.categories is None
            else args.categories
        ),
    )
    print("dataset size is", len(dataiter))
    tasks = np.arange(len(dataiter))

    my_task = np.array_split(tasks, num_tasks)[task_id]
    print("task size for this worker", len(my_task))

    output_path = (
        args.output_dir / f"{args.sdf_method}_{task_id:04d}_{num_tasks-1:04d}.json"
    )
    if output_path.exists():
        print(f"skipping {output_path.as_posix()} because it already exists")
        return

    out = []
    for idx in tqdm(my_task):
        out.append(preprocess(dataiter[idx], args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
