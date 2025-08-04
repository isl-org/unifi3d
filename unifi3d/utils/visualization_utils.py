from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d

import platform
import torch
import os
import sys
from matplotlib import cm
import trimesh
from transforms3d.euler import euler2mat

import random
import math
import json

# import bpy
# from mathutils import Vector
# from mathutils.noise import random_unit_vector


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

node = platform.node()
if "microsoft-standard" in platform.uname().release:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
else:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender


class Renderer(ABC):
    """
    Base Renderer class
    """

    @abstractmethod
    def generate_cam_poses(self):
        pass

    @abstractmethod
    def render(self, pointclouds=None, meshes=None):
        pass


class PyrenderRenderer(Renderer):
    def __init__(self, render_config=None):
        # Default render configuration
        self.render_config = render_config or {}
        self.shape = self.render_config.get("shape", (640, 640))
        self.light_intensity = self.render_config.get("light_intensity", 1.0)
        self.light_vertical_angle = self.render_config.get(
            "light_vertical_angle", -np.pi / 4
        )
        self.yfov = self.render_config.get("yfov", np.pi / 3.0)
        self.cam_angle_yaw = self._to_list(self.render_config.get("cam_angle_yaw", 0.0))
        self.cam_angle_pitch = self._to_list(
            self.render_config.get("cam_angle_pitch", 0.0)
        )
        self.cam_angle_z = self._to_list(self.render_config.get("cam_angle_z", 0.0))
        self.cam_dist = self._to_list(self.render_config.get("cam_dist", 2.0))
        self.cam_height = self._to_list(self.render_config.get("cam_height", 0.0))
        self.perpoint_color_flag = self.render_config.get("perpoint_color_flag", False)
        self.render_flags = self.render_config.get(
            "render_flags", pyrender.RenderFlags.NONE
        )
        self.cam_pose_list = self.generate_cam_poses()

    def _to_list(self, value):
        """Ensure the value is a list."""
        return value if isinstance(value, list) else [value]

    def generate_cam_poses(self):
        """
        Generate camera poses based on the input angles and distances.
        """
        # Ensure all camera parameters have the same length
        N_cam_pose = len(self.cam_angle_yaw)
        assert all(
            len(lst) == N_cam_pose
            for lst in [
                self.cam_angle_pitch,
                self.cam_angle_z,
                self.cam_dist,
                self.cam_height,
            ]
        )

        # Generate camera poses
        cam_pose_list = []
        for i in range(N_cam_pose):
            cam_pose = np.eye(4)
            R = euler2mat(
                self.cam_angle_yaw[i],
                -self.cam_angle_pitch[i],
                self.cam_angle_z[i],
                "ryxz",
            )
            cam_pose[:3, :3] = R
            cam_pose[2, 3] += self.cam_dist[i]
            cam_pose[:3, 3:] = R @ cam_pose[:3, 3:]
            cam_pose[1, 3] += self.cam_height[i]
            cam_pose_list.append(cam_pose)
        return cam_pose_list

    def create_renderer_and_scene(self):
        """Create the renderer and scene, adding directional lights."""
        self.renderer = pyrender.OffscreenRenderer(self.shape[0], self.shape[1])
        self.scene = pyrender.Scene()

        # Add directional lights
        for rot in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            dlight = pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0], intensity=self.light_intensity
            )
            T_dl = np.eye(4)
            T_dl[:3, :3] = euler2mat(self.light_vertical_angle, rot, 0.0, "sxyz")
            self.scene.add(dlight, pose=T_dl)

    def add_pointclouds_to_scene(self, pointclouds):
        """Add point clouds to the scene."""
        if pointclouds is None:
            return
        pcl_list = pointclouds.get("pcl_list", [])
        pcl_color_list = pointclouds.get("pcl_color_list", [])
        pcl_radius_list = pointclouds.get("pcl_radius_list", [])
        pcl_fallback_colors = pointclouds.get(
            "pcl_fallback_colors", [0.9, 0.9, 0.9, 1.0]
        )

        assert len(pcl_list) == len(pcl_radius_list)
        for i, (pcl, radius) in enumerate(zip(pcl_list, pcl_radius_list)):
            assert isinstance(radius, float)
            color = pcl_fallback_colors if pcl_color_list is None else pcl_color_list[i]
            if isinstance(color, torch.Tensor):
                color = color.numpy()
            if isinstance(color, np.ndarray) and color.ndim == 1:
                color = cm.cool(color)
            if radius <= 0.0:
                m = pyrender.Mesh.from_points(pcl, colors=color)
                self.scene.add(m)
            else:
                sm = trimesh.creation.uv_sphere(radius=radius)
                if self.perpoint_color_flag and isinstance(color, np.ndarray):
                    for p, c in zip(pcl, color):
                        sm = trimesh.creation.uv_sphere(radius=radius)
                        sm.visual.vertex_colors = c
                        tfs = np.eye(4)
                        tfs[:3, 3] = p
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        self.scene.add(m)
                else:
                    sm.visual.vertex_colors = color
                    tfs = np.tile(np.eye(4), (pcl.shape[0], 1, 1))
                    tfs[:, :3, 3] = pcl
                    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                    self.scene.add(m)

    def add_meshes_to_scene(self, meshes):
        """Add meshes to the scene."""
        if meshes is None:
            return

        mesh_color_list = []

        for i, mesh in enumerate(meshes):
            if not len(mesh.vertices):
                continue

            # Fix normals and face orientations
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.fix_normals()

            material = None
            if mesh_color_list and mesh_color_list[i] is not None:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                    alphaMode="OPAQUE",
                    doubleSided=True,
                    baseColorFactor=mesh_color_list[i],
                )
            else:
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.1,
                    roughnessFactor=0.5,
                    alphaMode="OPAQUE",
                    doubleSided=True,
                    baseColorFactor=np.array([0.0, 0.40784313725, 1.0, 1.0]),
                )
            self.scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))

    def render(self, pointclouds=None, meshes=None):
        """
        Render scenes with the given point clouds and meshes.

        :param pointclouds: list of pointcloud objects
        :param meshes: list of mesh objects
        """
        self.create_renderer_and_scene()
        self.add_pointclouds_to_scene(pointclouds)
        self.add_meshes_to_scene(meshes)

        rgb_list = []
        for cam_pose in self.cam_pose_list:
            camera = pyrender.PerspectiveCamera(
                yfov=self.yfov, aspectRatio=self.shape[0] / self.shape[1]
            )
            camera_node = self.scene.add(camera, pose=cam_pose)
            rgb, _ = self.renderer.render(self.scene, flags=self.render_flags)
            self.scene.remove_node(camera_node)
            rgb_list.append(np.expand_dims(rgb, 0))

        self.renderer.delete()

        concatenated_rgb = np.concatenate(rgb_list, axis=0).squeeze()
        return concatenated_rgb
