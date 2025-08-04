"""Functions related to materials"""

import numpy as np
import open3d as o3d
from scipy.interpolate import interpn
from collections import defaultdict

_SEP = "/"


def _interpolate_UVs(uvmap, primitive_ids, primitive_uvs):
    """Returns the interpolated uv coordinates
    Args:
        uvmap: The UV map of shape (N,3,2) with N as the number of triangles.
        primitive_ids: Array with the primitive (triangle) id with shape (M,).
        primitive_uvs: Array with the primitive UV coordinates with shape (M,2).
    Returns:
        Array with the interpolated UV coordinates with shape (M,2).
    """
    assert uvmap.ndim == 3 and uvmap.shape[1:] == (3, 2)
    assert primitive_ids.ndim == 1
    assert (
        primitive_uvs.ndim == 2
        and primitive_uvs.shape[0] == primitive_ids.shape[0]
        and primitive_uvs.shape[1] == 2
    )
    uvs = uvmap[primitive_ids]
    result = (
        (1 - primitive_uvs.sum(axis=-1, keepdims=True)) * uvs[:, 0]
        + primitive_uvs[:, 0:1] * uvs[:, 1]
        + primitive_uvs[:, 1:] * uvs[:, 2]
    )
    return result


def _texture_bilinear(tex: np.ndarray, coords: np.ndarray, mode="repeat"):
    """Texture lookup with bilinear interpolation
    Args:
        tex: 3D Array with shape (H,W,C) and dtype float32.
        coords: 2D uv coordinates array (N,2) at which the texture will be sampled.
            The coordinates should be normalized and in the range [0,1].

    Returns:
        An array of shape (N,C).
    """
    assert mode in ("repeat", "clamp")
    assert tex.ndim == 3
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    result = np.empty((coords.shape[0], tex.shape[-1]), dtype=np.float32)

    height, width, channels = tex.shape
    uspace = (np.arange(width) + 0.5) / width
    vspace = (np.arange(height) + 0.5) / height

    if mode == "clamp":
        coords2 = np.clip(
            coords[:, ::-1], [vspace[0], uspace[0]], [vspace[-1], uspace[-1]]
        )
    elif mode == "repeat":
        coords2 = np.modf(coords[:, ::-1] + 1)[0]

    for i in range(channels):
        result[..., i] = interpn(
            points=(vspace, uspace),
            values=tex[..., i],
            xi=coords2,
            bounds_error=False,
            fill_value=0,
        )
    return result


def _get_material_attributes_object(closest_points, obj_dict):
    """Helper function for getting the material attributes for a single object.
    Args:
        closest_points: Dict of subsampled numpy arrays of the output of
            RaycastingScene.closest_points.
        obj_dict: A dictionary with the object geometry and material fields.
            The keys do not include the object name.
            See blender_utils.scene_to_numpy() for an example of the structure
            with object names.

    Returns:
        A dict with the material attributes.
    """
    # TODO vertex colors are not supported yet
    num_values = len(closest_points["primitive_ids"])

    result = {
        "base_color": np.empty(shape=(num_values, 3), dtype=np.float32),
        "metallic": np.empty(shape=(num_values, 1), dtype=np.float32),
        "roughness": np.empty(shape=(num_values, 1), dtype=np.float32),
    }

    material_ids = obj_dict["material_index"][closest_points["primitive_ids"]]

    for mat_id in set(material_ids):
        mask = material_ids == mat_id
        for attr, data in result.items():
            # single value
            mat_data = obj_dict[f"materials{_SEP}{mat_id}{_SEP}{attr}"]
            if mat_data.shape[:2] == (1, 1):
                data[mask] = mat_data[0, 0]
            # texture
            else:
                if mat_data.dtype == np.uint8:
                    tex = mat_data.astype(np.float32) / 255
                elif mat_data.dtype == np.float32:
                    tex = mat_data
                else:
                    raise Exception(
                        f"unexpected dtype for texture {mat_data} {mat_data.dtype}"
                    )

                if tex.shape[-1] > data.shape[-1]:
                    tex = tex[..., : data.shape[-1]]

                uv_layer_keys = [
                    x for x in obj_dict.keys() if x.startswith(f"uv_layers{_SEP}")
                ]
                assert len(uv_layer_keys) == 1
                uvs = _interpolate_UVs(
                    obj_dict[uv_layer_keys[0]],
                    closest_points["primitive_ids"][mask],
                    closest_points["primitive_uvs"][mask],
                )
                data[mask] = _texture_bilinear(tex, uvs)

    return result


def get_material_attributes(points, scene_dict):
    """Returns the material attributes for every query point for the scene.

    Args:
        points: (N,3) array with the xyz positions.
        scene_dict: A dict as returned by blender_utils.scene_to_numpy()

    Returns:
        A dict with the attributes for each point.
    """
    assert points.ndim == 2 and points.shape[-1] == 3
    # split with respect to object name into subdicts
    objects = defaultdict(dict)
    for k, v in scene_dict.items():
        name = k.split(_SEP)[0]
        field = _SEP.join(k.split(_SEP)[1:])
        objects[name][field] = v

    scene = o3d.t.geometry.RaycastingScene()
    geom_id_obj = {}
    for obj in objects.values():
        geom_id = scene.add_triangles(
            obj["vertices"].astype(np.float32), obj["triangles"].astype(np.uint32)
        )
        geom_id_obj[geom_id] = obj

    closest_points = {
        k: v.numpy()
        for k, v in scene.compute_closest_points(points.astype(np.float32)).items()
    }

    num_values = len(closest_points["primitive_ids"])
    result = {
        "base_color": np.empty(shape=(num_values, 3), dtype=np.float32),
        "metallic": np.empty(shape=(num_values, 1), dtype=np.float32),
        "roughness": np.empty(shape=(num_values, 1), dtype=np.float32),
    }

    for geom_id, obj in geom_id_obj.items():
        mask = closest_points["geometry_ids"] == geom_id
        if not np.count_nonzero(mask):
            continue
        tmp_closest_points = {k: v[mask] for k, v in closest_points.items()}
        tmp = _get_material_attributes_object(tmp_closest_points, obj)
        for k, v in result.items():
            result[k][mask] = tmp[k]

    return result
