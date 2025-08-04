"""Module with utility functions for manipulating objects and scenes in blender"""

from typing import Optional, Sequence, Union, List
import sys
from pathlib import Path

import bpy
from bpy import data as D
from bpy import context as C
from mathutils import Vector, Matrix
import numpy as np

_SEP = "/"
_blender_version = bpy.app.version


def import_mesh(
    path: Path, up_axis: Optional[str] = None, forward_axis: Optional[str] = None
):
    """Imports the mesh and returns a reference to the object.

    This function assumes that the scene is empty in order to return the right object!

    Args:
        path: Path to the mesh
        up_axis: The up axis of the imported mesh. Blender will make this the +Z axis
        forward_axis: The forward axis of the imported mesh. Blender will make this the +Y axis

    Returns:
        Returns the imported object
    """
    path = Path(path)
    mesh = None
    if _blender_version >= (4, 0, 0):
        read_fn = {
            ".obj": bpy.ops.wm.obj_import,
            ".ply": bpy.ops.import_mesh.ply,
            ".stl": bpy.ops.import_mesh.stl,
            ".glb": bpy.ops.import_scene.gltf,
            ".gltf": bpy.ops.import_scene.gltf,
        }[path.suffix]
    else:
        read_fn = {
            ".obj": bpy.ops.import_scene.obj,
            ".ply": bpy.ops.import_mesh.ply,
            ".stl": bpy.ops.import_mesh.stl,
            ".glb": bpy.ops.import_scene.gltf,
            ".gltf": bpy.ops.import_scene.gltf,
        }[path.suffix]
    read_args = {}
    if up_axis is not None:
        read_args["up_axis"] = up_axis
    if forward_axis is not None:
        read_args["forward_axis"] = forward_axis
    print(read_args)
    read_fn(filepath=str(path), **read_args)

    if _blender_version >= (4, 1, 0):
        # remove objects in the glTF_not_exported collection
        for obj in C.scene.objects.values():
            for col in obj.users_collection:
                if col.name == "glTF_not_exported":
                    bpy.data.objects.remove(obj)

    for k, v in bpy.data.objects.items():
        if isinstance(v.data, bpy.types.Mesh):
            mesh = v
            break

    if mesh is not None:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
    return mesh


def delete_all_objects():
    """Helper function for deleting all objects in the scene."""
    # delete everything
    for obj in C.scene.objects:
        obj.select_set(True)
        bpy.ops.object.delete()


def compute_bbox(objs):
    """Computes the bounding box that includes all mesh objects.

    This function ignores all objects that are not meshes.

    Args:
        objs: List of objects

    Returns:
        A tuple with two 3D vectors (bbox_min,bbox_max).
    """
    bbox_min = np.array((np.inf,) * 3)
    bbox_max = np.array((-np.inf,) * 3)
    for obj in objs:
        if isinstance(obj.data, (bpy.types.Mesh)):
            for x in obj.bound_box:
                x = np.array(obj.matrix_world @ Vector(x))
                bbox_min = np.minimum(bbox_min, x)
                bbox_max = np.maximum(bbox_max, x)
    return bbox_min, bbox_max


def compute_scene_bbox():
    """Compute the boundind box that includes all meshes in the scene

    Returns:
        A tuple with two 3D vectors (bbox_min,bbox_max).
    """
    return compute_bbox(bpy.context.scene.objects.values())


def create_environment_lighting(envmap_path: Path, axis=(0, 0, 1), angle=0):
    """Setup the node tree for environment lighting with cycles
    Args:
        envmap_path: Path to the .hdr file
        axis: The axis for rotating the environment map
        angle: Rotation angle in radians
    """
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    nodes.clear()

    texcoord = nodes.new("ShaderNodeTexCoord")
    rotate = nodes.new("ShaderNodeVectorRotate")
    rotate.inputs["Axis"].default_value = axis
    rotate.inputs["Angle"].default_value = angle

    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(envmap_path))
    out = nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(texcoord.outputs["Generated"], rotate.inputs["Vector"])
    world.node_tree.links.new(rotate.outputs["Vector"], env.inputs["Vector"])
    world.node_tree.links.new(env.outputs["Color"], out.inputs["Surface"])
    world.cycles_visibility.camera = True


def create_compositing_with_autoexposure():
    """Adds nodes to implement autoexposure to the compositing nodetree

    This function assumes the default nodes present in the compositing nodetree

    Returns:
        The tonemap node with default parameters
            tonemap.intensity = 0
            tonemap.contrast = 0
            tonemap.adaptation = 0
            tonemap.correction = 0

        See https://docs.blender.org/manual/en/latest/compositing/types/color/adjust/tone_map.html
        for more information.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = scene.node_tree.nodes

    # input and output nodes
    render_layers = nodes["Render Layers"]
    composite = nodes["Composite"]

    nt.links.remove(render_layers.outputs["Image"].links[0])

    tonemap = nodes.new("CompositorNodeTonemap")
    tonemap.tonemap_type = "RD_PHOTORECEPTOR"
    tonemap.intensity = 0
    tonemap.contrast = 0
    tonemap.adaptation = 0
    tonemap.correction = 0
    nt.links.new(render_layers.outputs["Image"], tonemap.inputs["Image"])
    nt.links.new(tonemap.outputs["Image"], composite.inputs["Image"])

    # return the tonemap node to allow changing the default parameters
    return tonemap


def normalize_objects(objs, target_size=1, normalize_all=False):
    """Normalize the objects (scale and translation).

    This normalizes the objects such that the joint bounding box of all meshes
    is centered at the origin and the longest edge is 'target_size'.
    Note that the normalization affects all objects but the computed
    transformation is based only on the bbox including all meshes.

    Args:
        target_size (float): The new size of the normalized objects. This is the
            length of the longest edge. Default is 1.0
        normalize_all (bool): If True, normalize all objects in the scene, else
            only the provided objects
    """
    assert target_size > 0
    objs_to_normalize = bpy.context.scene.objects.values() if normalize_all else objs
    bbox_min, bbox_max = compute_bbox(objs)
    center = 0.5 * (bbox_min + bbox_max)
    scale = target_size / max(bbox_max - bbox_min)
    for obj in objs_to_normalize:
        if obj.parent is None:
            obj.scale *= scale
            obj.location *= scale
    bpy.context.view_layer.update()

    bbox_min, bbox_max = compute_bbox(objs)
    center = 0.5 * (bbox_min + bbox_max)
    for obj in objs_to_normalize:
        if obj.parent is None:
            obj.location -= Vector(center)
    bpy.context.view_layer.update()


def normalize_scene(target_size=1):
    """Normalize all objects in the scene (scale and translation).

    This normalizes the scene such that the joint bounding box of all meshes
    is centered at the origin and the longest edge is 'target_size'.
    Note that the normalization affects all objects but the computed
    transformation is based only on the bbox including all meshes.

    Args:
        target_size (float): The new size of the normalized scene. This is the
            length of the longest edge. Default is 1.0
    """
    normalize_objects(bpy.context.scene.objects.values(), target_size=target_size)


def triangulate_mesh(obj):
    """Triangulates the mesh object by applying a modifier"""
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new("Triangulate", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def all_faces_are_triangles(obj):
    """Test if all faces are triangles for a mesh object."""
    assert obj.type == "MESH"
    mesh = obj.data
    num_polys = len(mesh.polygons)
    loop_total = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_total)
    return all(loop_total == 3)


def setup_vertex_color_shading(obj):
    """Modifies the shading node graph to use vertex colors as albedo.

    This function has no effect if there is no 'Col' vertex color attribute or the object is not a mesh.

    Args:
        obj An object as in bpy.context.objects
    """
    if not isinstance(obj.data, bpy.types.Mesh):
        return

    if not "Col" in obj.data.vertex_colors:
        return

    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    mat_node = nodes["Material Output"]
    bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(bsdf_node.outputs["BSDF"], mat_node.inputs["Surface"])

    node_base = nodes.new("ShaderNodeVertexColor")
    node_base.layer_name
    node_base.layer_name = "Col"
    mat.node_tree.links.new(node_base.outputs["Color"], bsdf_node.inputs["Base Color"])


def image_to_numpy(image, scale=1):
    """Converts a blender image to a numpy array."""
    assert image.type == "IMAGE"
    w, h, c = image.size[0], image.size[1], image.channels
    assert image.depth in (
        8,
        16,
        24,
        32,
        64,
        96,
        128,
    ), f"{image.name=}, {image.depth=}, {image.channels=}, {image.colorspace_settings.name=}"
    arr = scale * np.array(image.pixels[:]).reshape(h, w, c)

    if image.is_float:
        num_channels = image.depth // 32
    else:
        num_channels = image.depth // 8

    if image.is_float:
        result = arr[..., :num_channels].astype(np.float32)
    else:
        dtype = np.uint8
        result = (np.clip(arr[..., :num_channels], 0, 1) * np.iinfo(dtype).max).astype(
            dtype
        )
    return result


def material_to_numpy(mat):
    """Converts a blender material to a dictionary of numpy arrays"""
    result = {}
    assert mat.use_nodes
    nodes = mat.node_tree.nodes
    principled = nodes["Material Output"].inputs["Surface"].links[0].from_node
    assert (
        principled.type == "BSDF_PRINCIPLED"
    ), f'Expected "BSDF_PRINCIPLED" but got "{principled.type}"'

    result["name"] = mat.name

    def parse_material_value(pname):
        # This function parses the node tree for the given parameter input pname
        # and returns a numpy array. We assume that we always have a texture image
        # or use the default value of the input.
        if len(principled.inputs[pname].links):
            input_type = principled.inputs[pname].type
            selected_channel = None
            scale = 1
            link = principled.inputs[pname].links[0]
            node = link.from_node
            while node.type not in ("TEX_IMAGE",):
                assert node.type in ("SEPARATE_COLOR", "MATH"), f"{node.type=}"
                if node.type == "SEPARATE_COLOR":
                    selected_channel = {"Red": 0, "Green": 1, "Blue": 2}[
                        link.from_socket.name
                    ]
                    link = node.inputs[0].links[0]
                    node = link.from_node
                elif node.type == "MATH":
                    assert node.operation == "MULTIPLY"
                    if node.inputs[0].links:
                        link = node.inputs[0].links[0]
                        scale = node.inputs[1].default_value
                    else:
                        link = node.inputs[1].links[0]
                        scale = node.inputs[0].default_value
                    node = link.from_node
                else:
                    raise Exception(f"unexpected node type {node.type=}")

            if node.type == "TEX_IMAGE":
                if input_type == "RGBA":
                    return image_to_numpy(node.image, scale)
                elif input_type == "VALUE" and selected_channel is not None:
                    return image_to_numpy(node.image, scale)[..., selected_channel][
                        ..., None
                    ]
                else:
                    raise Exception(
                        f"unknown tex image configuration {input_type=} {selected_channel=}"
                    )
        else:
            default_value = principled.inputs[pname].default_value
            if not isinstance(default_value, float):
                # discard unused alpha for the base color
                default_value = default_value[:3]
            return np.array([[default_value]], dtype=np.float32)

    for pname in ("Base Color", "Metallic", "Roughness"):
        key = pname.lower().replace(" ", "_")
        result[key] = parse_material_value(pname)

    return result


def mesh_to_numpy2(obj):
    """Creates a dict of numpy arrays that describes the mesh.

    Args:
        obj: mesh object

    Returns:
        A dictionary of numpy arrays describing the mesh
    """
    result = {}
    assert obj.type == "MESH"
    mesh = obj.data

    matrix_obj2world = np.asarray(obj.matrix_world)
    result["matrix_world"] = matrix_obj2world

    num_vertices = len(mesh.vertices)
    vertices = np.empty(num_vertices * 3, np.float32)
    mesh.vertices.foreach_get("co", vertices)
    result["vertices"] = vertices.reshape(-1, 3)

    normals = np.empty(num_vertices * 3, np.float32)
    mesh.vertices.foreach_get("normal", normals)
    result["vertex_normals"] = normals.reshape(-1, 3)

    num_loops = len(mesh.loops)
    loops = np.empty(num_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loops)

    num_polys = len(mesh.polygons)
    loop_start = np.empty(num_polys, dtype=np.int32)
    loop_total = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)
    mesh.polygons.foreach_get("loop_total", loop_total)
    assert all(loop_total == 3), "not all faces are triangles!"

    material_index = np.empty(num_polys, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", material_index)
    result["material_index"] = material_index

    triangles = np.empty((num_polys, 3), dtype=np.int32)
    triangles[:, 0] = loops[loop_start]
    triangles[:, 1] = loops[loop_start + 1]
    triangles[:, 2] = loops[loop_start + 2]
    result["triangles"] = triangles.astype(np.int64)

    # color layers
    for cl in mesh.vertex_colors:
        # vertex colors are rgba and per triangle index
        arr = np.empty(num_loops * 4, dtype=np.float32)
        cl.data.foreach_get("color", arr)
        arr = arr.reshape(-1, 4)
        vertex_color = np.empty((num_vertices, 4), dtype=np.float32)
        vertex_color[loops] = arr
        result[f"vertex_colors{_SEP}{cl.name}"] = vertex_color

    # uvmap
    if len(mesh.uv_layers):
        for uvmap in obj.data.uv_layers:
            uvmap_arr = np.zeros(len(uvmap.data) * 2, dtype=np.float64)
            uvmap.data.foreach_get("uv", uvmap_arr)
            uvmap_arr = uvmap_arr.reshape(-1, 3, 2)
            result[f"uv_layers{_SEP}{uvmap.name}"] = uvmap_arr.astype(np.float32)

    # materials
    for i, mat in enumerate(obj.material_slots):
        try:
            for k, v in material_to_numpy(mat.material).items():
                result[f"materials{_SEP}{i}{_SEP}{k}"] = v
        except Exception as e:
            print(f"Failed to convert material with id {i}\n", e)

    return result


def scene_to_numpy(objects):
    """Converts a blender scene to a dictionary of numpy arrays.

    This function only converts the mesh objects and ignores all other object
    types.


    Args:
        objects: A list of blender objects

    Returns:
        A dictionary of numpy arrays describing the mesh objects.
        The dict structure looks like this

        {
            'obj_name1/vertices': np.array (N,3) dtype=np.float32,
            'obj_name1/triangles': np.array (M,3) dtype=np.int64,
            'obj_name1/matrix_world': np.array (4,4) dtype=np.float32,
            'obj_name1/material_index': np.array (M,) dtype=np.int32,
            'obj_name1/vertex_colors/name1': np.array (N,4) dtype=np.float32,
            'obj_name1/vertex_colors/name2': np.array (N,4) dtype=np.float32,
            'obj_name1/uv_layers/name1': np.array (M,3,2) dtype=np.float32,
            'obj_name1/uv_layers/name2': np.array (M,3,2) dtype=np.float32,
            'obj_name1/materials/id/name': str,
            'obj_name1/materials/id/base_color': np.array (H,W,4) dtype=np.float32 or np.uint8,
            'obj_name1/materials/id/metallic': np.array (H,W,1) dtype=np.float32 or np.uint8,
            'obj_name1/materials/id/roughness': np.array (H,W,1) dtype=np.float32 or np.uint8,
        }

    """
    result = {}
    for obj in objects:
        if obj.type == "MESH":
            for k, v in mesh_to_numpy2(obj).items():
                result[f"{obj.name}{_SEP}{k}"] = v
    return result
