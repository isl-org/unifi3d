import ocnn
import open3d as o3d
import os
from plyfile import PlyData, PlyElement
import numpy as np
import skimage.measure
import trimesh
import torch


def get_mgrid(size, dim=3):
    r"""
    Example:
    >>> get_mgrid(3, dim=2)
        array([[0.0,  0.0],
               [0.0,  1.0],
               [0.0,  2.0],
               [1.0,  0.0],
               [1.0,  1.0],
               [1.0,  2.0],
               [2.0,  0.0],
               [2.0,  1.0],
               [2.0,  2.0]], dtype=float32)
    """
    coord = np.arange(0, size, dtype=np.float32)
    coords = [coord] * dim
    output = np.meshgrid(*coords, indexing="ij")
    output = np.stack(output, -1)
    output = output.reshape(size**dim, dim)
    return output


def calc_sdf(model, size=256, max_batch=64**3, bbmin=-1.0, bbmax=1.0):
    # generate samples
    num_samples = size**3
    samples = get_mgrid(size, dim=3)
    samples = samples * ((bbmax - bbmin) / size) + bbmin  # [0,sz]->[bbmin,bbmax]
    samples = torch.from_numpy(samples)
    sdfs = torch.zeros(num_samples)

    # forward
    head = 0
    while head < num_samples:
        tail = min(head + max_batch, num_samples)
        sample_subset = samples[head:tail, :]
        idx = torch.zeros(sample_subset.shape[0], 1)
        pts = torch.cat([sample_subset, idx], dim=1).cuda()
        pred = model(pts).squeeze().detach().cpu()
        sdfs[head:tail] = pred
        head += max_batch
    sdfs = sdfs.reshape(size, size, size).numpy()
    return sdfs


def create_mesh(
    model,
    bbox=None,
    sdf_scale=0.9,
    size=256,
    max_batch=64**3,
    level=0,
    bbmin=-0.9,
    bbmax=0.9,
    mesh_scale=1.0,
    export_type="o3d",  # "o3d or trimesh"
    return_sdf=False,
):

    # data is a combination of batch and output, where bbox is from batch, and neural_mpu is from output.
    if bbox is not None:
        bbmin, bbmax = bbox[:3], bbox[3:]
    else:
        sdf_scale = sdf_scale
        bbmin, bbmax = -sdf_scale, sdf_scale
    # marching cubes
    sdf_values = calc_sdf(model, size, max_batch, bbmin, bbmax)
    vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
    try:
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)
    except:
        pass
    if vtx.size == 0 or faces.size == 0:
        print("Warning from marching cubes: Empty mesh!")
        return

    # normalize vtx
    vtx = vtx * ((bbmax - bbmin) / size) + bbmin  # [0,sz]->[bbmin,bbmax]
    vtx = vtx * mesh_scale  # rescale

    # save to ply and npy
    if export_type == "trimesh":
        mesh = trimesh.Trimesh(vtx, faces)
    elif export_type == "o3d":
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vtx)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
    if return_sdf:
        return mesh, sdf_values
    else:
        return mesh


def points2ply(filename, points, scale=1.0):
    xyz = ocnn.points_property(points, "xyz")
    normal = ocnn.points_property(points, "normal")
    has_normal = normal is not None
    xyz = xyz.numpy() * scale
    if has_normal:
        normal = normal.numpy()

    # data types
    data = xyz
    py_types = (float, float, float)
    npy_types = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if has_normal:
        py_types = py_types + (float, float, float)
        npy_types = npy_types + [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        data = np.concatenate((data, normal), axis=1)

    # format into NumPy structured array
    vertices = []
    for idx in range(data.shape[0]):
        vertices.append(tuple(dtype(d) for dtype, d in zip(py_types, data[idx])))
    structured_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(structured_array, "vertex")

    # write ply
    PlyData([el]).write(filename)


def doctree_post_process(batch, outputs, config):
    print("Saving outputs...")
    gt_filenames = []
    pred_filenames = []
    for filename in batch["file_name"]:
        print(filename)
        pos = filename.rfind(".")
        if pos != -1:
            filename = filename[:pos]  # remove the suffix
        pred_filename = os.path.join(config.output_dir, filename + ".obj")
        gt_filename = pred_filename.replace(".obj", ".input.ply")

        if not os.path.exists(pred_filename):
            print(f"{pred_filename} does not exist.")
            folder = os.path.dirname(pred_filename)
            if not os.path.exists(folder):
                os.makedirs(folder)
            bbox = batch["bbox"][0].numpy() if "bbox" in batch else None
            output = create_mesh(
                model=outputs["neural_mpu"],
                bbox=bbox,
                sdf_scale=config.sdf_scale,
                size=config.size,
                mesh_scale=config.mesh_scale,
                export_type="trimesh",
                return_sdf=True,
            )
            if output is not None:
                trimesh_mesh, sdf_values = output
                trimesh_mesh.export(pred_filename)
                if config.save_sdf:
                    np.save(pred_filename.replace(".obj", ".sdf.npy"), sdf_values)

                # save the input point cloud
                points2ply(gt_filename, batch["points_in"][0].cpu(), config.mesh_scale)
                print(f"saving to {gt_filename}")
            else:
                print("output is None!")
        else:
            print(f"{pred_filename} already exist.")
        if os.path.exists(gt_filename) and os.path.exists(pred_filename):
            gt_filenames.append(gt_filename)
            pred_filenames.append(pred_filename)

    return (gt_filenames, pred_filenames)
