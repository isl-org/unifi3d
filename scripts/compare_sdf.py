import json
import time
import argparse
import os
import subprocess
import h5py
import open3d as o3d
import numpy as np
import torch
import mcubes
import trimesh
from skimage.segmentation import flood_fill
from skimage.util import view_as_windows
from unifi3d.utils.evaluate.metrics.chamfer_distance import ChamferDistance

# from mesh_to_sdf import mesh_to_voxels
from unifi3d.data.data_iterators import (
    ShapenetIterator,
    PartNetIterator,
    DemoIterator,
)
from unifi3d.utils.evaluate.visualize import save_combined_mesh_with_rows
from unifi3d.utils.data.sdf_utils import (
    generate_query_points,
    sdf_to_mesh,
    mesh_to_sdf,
    center_scale_mesh,
)


def compute_sdf_flood_fill(
    mesh,
    resolution: int = 128,
    bbox_min=(-0.5, -0.5, -0.5),
    bbox_max=(0.5, 0.5, 0.5),
    seed_point=(0, 0, 0),
):
    """
    Computes the signed distance field to the surface.
    The sign is determined by filling the outside volume starting from a seed
    point. This function does not need correcly oriented faces.

    Args:
        vertices: (n,3) numpy array with float type
        triangles: (n,3) numpy array with int type
        resolution: Grid resolution
        bbox_min: (3,) bounding box min values
        bbox_max: (3,) bounding box max values
        seed_point: The seed point as voxel index (3,)

    Returns:
        Tuple with (SDF, voxel size)
    """
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    triangles = np.asarray(mesh.triangles)

    grid = np.meshgrid(
        np.linspace(bbox_min[0], bbox_max[0], num=resolution),
        np.linspace(bbox_min[1], bbox_max[1], num=resolution),
        np.linspace(bbox_min[2], bbox_max[2], num=resolution),
        indexing="ij",
    )
    grid = np.stack(grid[::-1], axis=-1).astype(np.float32)

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
    vol = flood_fill(vol, seed_point=seed_point, connectivity=1, new_value=2)
    # estimate a vector pointing outside by looking at the voxels labelled with '2'
    windows = view_as_windows(np.pad(vol, 1, mode="reflect"), 3)
    dirs = (
        np.stack(np.unravel_index(np.arange(27), (3, 3, 3)), axis=-1).reshape(
            3, 3, 3, -1
        )[..., ::-1]
        - 1
    )
    dirs = dirs.astype(np.int8)
    outside_dir = ((windows == 2).astype(np.int8)[..., None] * dirs).sum(
        axis=(3, 4, 5), dtype=np.int8
    )

    border = np.logical_and(np.any(windows == 2, axis=(-1, -2, -3)), vol == 1)
    sign = np.sign((outside_dir[border] * diffvec[border]).sum(axis=-1))
    # define which border voxels are outside by adding the sign
    vol[border] += sign.astype(np.int8)

    sdf = np.linalg.norm(diffvec, axis=-1)

    # compute inside distances. Consider only triangles intersecting the border voxels
    select_face_idx = list(set(closest_points["primitive_ids"][border]))
    selected_faces = triangles[select_face_idx]
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    inside = vol != 2
    closest_points = {
        k: v.numpy()
        for k, v in scene.compute_closest_points(
            grid[inside],
        ).items()
    }
    diffvec = grid[inside] - closest_points["points"]
    dist_inside = -np.linalg.norm(diffvec, axis=-1)
    sdf[inside] = dist_inside
    return torch.from_numpy(sdf.swapaxes(2, 0))  # had to add swapaxes


def test_sdf_reconstruction(
    iterator,
    manifold_dataset_path: str,
    max_files_per_ds: int = 10,
    sample_res: int = 64,
    grid_resolution: int = 64,
    padding: float = 0.1,
    level: float = 0.0,
    manifoldize: bool = True,
    method: str = "o3d",
    manifold_exec_path: str = None,
    debug: bool = False,
):
    print(
        "--- Starting test with configuration",
        sample_res,
        grid_resolution,
        padding,
        level,
        manifoldize,
        method,
    )

    np.random.seed(42)
    sample_inds_to_test = np.random.permutation(len(iterator))[:max_files_per_ds]

    # Generate query points once
    query_points = generate_query_points(grid_resolution=sample_res, padding=padding)

    meshes, runtimes = [], []

    chamfer_distance_mesh = ChamferDistance()

    res_dict = {}
    res_dict["chamfer_dist_to_original"] = {}
    res_dict["chamfer_dist_to_manifoldized"] = {}

    chamfer_distance_dict = {}
    for sample_ind in sample_inds_to_test:
        sample = iterator[sample_ind]
        inp_path = sample["file_path"]

        if inp_path[-4:] not in [".glb", ".obj"]:
            print("skipping", inp_path)
            continue

        # manifoldize if desired
        if manifoldize:
            manifold_path = os.path.join(
                manifold_dataset_path, (inp_path.split(os.sep)[-1])[:-4] + ".obj"
            )
            # print(manifold_path)
            if not os.path.exists(manifold_path):
                # print(os.path.exists(inp_path))
                result = subprocess.run(
                    [manifold_exec_path] + [inp_path, manifold_path, "50000"],
                    stdout=subprocess.PIPE,
                    text=True,
                )
            read_mesh_path = manifold_path
        else:
            read_mesh_path = inp_path

        # load gt mesh (without manifoldization) and center
        mesh_gt = o3d.io.read_triangle_mesh(inp_path)
        mesh_gt = center_scale_mesh(mesh_gt)

        # load manifoldized mesh for further processing steps
        if manifoldize:
            mesh = o3d.io.read_triangle_mesh(read_mesh_path)
            mesh = center_scale_mesh(mesh)
        else:
            mesh = mesh_gt

        # transform to sdf
        tic = time.time()
        if method == "o3d+flood":
            sdf = compute_sdf_flood_fill(
                mesh,
                resolution=grid_resolution,
            )
        elif method == "o3d":
            sdf = mesh_to_sdf(mesh, sample_res, grid_resolution)[0]
        else:
            raise NotImplementedError("Method not available")
        runtimes.append(time.time() - tic)
        assert sdf.shape[0] == grid_resolution
        # transform back to mesh
        mesh_reconstructed = sdf_to_mesh(
            sdf.cpu().numpy(), level=level, padding=padding
        )

        # # Debugging: save sdf and check if the scales are correct:
        # with h5py.File("test_sdf.h5", "w") as hf:
        #     hf.create_dataset("pc_sdf_sample", data=sdf)
        # verts = np.asarray(mesh.vertices)
        # print(np.min(verts, axis=0), np.max(verts, axis=0))
        # verts = np.asarray(mesh_gt.vertices)
        # print(np.min(verts, axis=0), np.max(verts, axis=0))
        # verts = np.asarray(mesh_reconstructed.vertices)
        # print("pred", np.mean(verts), np.std(verts))
        # print(np.min(verts, axis=0), np.max(verts, axis=0))

        # compute error
        chamfer = chamfer_distance_mesh(mesh_gt, mesh_reconstructed)
        if manifoldize:
            chamfer_to_manifoldized = chamfer_distance_mesh(mesh, mesh_reconstructed)
        else:
            chamfer_to_manifoldized = chamfer
        # print(chamfer, chamfer_to_manifoldized)
        res_dict["chamfer_dist_to_original"][inp_path] = chamfer
        res_dict["chamfer_dist_to_manifoldized"][inp_path] = chamfer_to_manifoldized

        if debug:
            meshes.append([mesh_gt, mesh_reconstructed])

    # put everythin in dict
    res_dict["runtime_sdf"] = np.mean(runtimes)
    res_dict["avg_chamfer_to_original"] = np.mean(
        list(res_dict["chamfer_dist_to_original"].values())
    )
    res_dict["avg_chamfer_to_manifoldized"] = np.mean(
        list(res_dict["chamfer_dist_to_manifoldized"].values())
    )
    print(
        "Average chamfer ",
        res_dict["avg_chamfer_to_original"],
        res_dict["avg_chamfer_to_manifoldized"],
    )
    if debug:
        save_combined_mesh_with_rows("outputs/debug_mesh_comp.obj", meshes)
    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="shapenet", help="Dataset to test on"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/ShapeNetCore.v1",
        required=True,
        help="Dataset path",
    )
    parser.add_argument(
        "--manifold_save_path",
        type=str,
        required=True,
        help="where to save manifold obj files that are generated on the fly",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Where to save experiment outputs",
    )
    parser.add_argument(
        "--manifold_exec_path",
        type=str,
        default=None,
        help="Path to manifold exec",
    )
    parser.add_argument(
        "--nr_files", type=int, default=10, help="Max number of files to use"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # manifoldization
    manifold_exec_path = args.manifold_exec_path

    os.makedirs(args.out_dir, exist_ok=True)

    manifold_dataset_path = os.path.join(args.manifold_save_path, args.dataset)
    os.makedirs(manifold_dataset_path, exist_ok=True)

    configs_to_test = []
    # how many points are sampled for converting to sdf
    for sample_res in [64, 128, 256]:
        # what is the resolution of the final grid
        for grid_resolution in [64, 128]:
            if sample_res < grid_resolution:
                continue
            for padding in [0, 0.2]:
                for level in [0, 0.01]:
                    for manifoldize in [False, True]:
                        for method in ["o3d", "o3d+flood"]:
                            configs_to_test.append(
                                {
                                    "sample_res": sample_res,
                                    "grid_resolution": grid_resolution,
                                    "padding": padding,
                                    "level": level,
                                    "manifoldize": manifoldize,
                                    "method": method,
                                }
                            )
    if args.debug:  # overwrite the configs in the for loops with a single example
        configs_to_test = [
            {
                "sample_res": 64,
                "grid_resolution": 64,
                "padding": 0,
                "level": 0,
                "manifoldize": True,
                "method": "o3d",
            }
        ]
    print(f"Testing {len(configs_to_test)} different configurations")

    if args.dataset == "shapenet":
        iterator = ShapenetIterator(args.data_path)
    elif args.dataset == "partnet":
        iterator = PartNetIterator(args.data_path)
    elif args.dataset == "demo":
        iterator = DemoIterator(args.data_path)

    for config in configs_to_test:
        chamfer_distance_res_dict = test_sdf_reconstruction(
            iterator,
            manifold_dataset_path,
            max_files_per_ds=args.nr_files,
            debug=args.debug,
            manifold_exec_path=manifold_exec_path,
            **config,
        )
        save_dict = config.copy()
        save_dict["dataset"] = args.dataset
        save_dict.update(chamfer_distance_res_dict)

        outfn = (
            f"experiment_{int(time.time())}.json" if not args.debug else "debug.json"
        )
        out_path = os.path.join(args.out_dir, outfn)
        with open(out_path, "w", encoding="latin-1") as outfile:
            json.dump(save_dict, outfile)
