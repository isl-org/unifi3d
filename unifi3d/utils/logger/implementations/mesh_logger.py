import numpy as np
import open3d as o3d
import os
import torch
import trimesh
from .base_logger import BaseLogger
from .interfaces.base_logint import BaseLogint


class MeshLogger(BaseLogger):
    """
    MeshLogger logs meshes for batches and phases.
    """

    def __init__(
        self, logger_interface: BaseLogint, log_path: str, config: dict
    ) -> None:
        super().__init__(logger_interface, log_path, config)

        self.NAME = "mesh"
        self.iter_counter = 0

        # Make sure the required folders exist
        os.makedirs(self.log_path, exist_ok=True)

    def log_iter(self, batch) -> None:
        """
        Logs mesh data from the batch to BaseLogint and saves it to files.

        :param batch: Dictionary containing batch data, metadata, and configurations.
        """
        if self.NAME not in batch["output_parser"]:
            return

        keys_list = batch["output_parser"][self.NAME]
        if not keys_list:
            return

        # Update logger state
        self.phase = batch["phase"]
        self.epoch = batch["epoch"]

        epoch_dir = os.path.join(self.log_path, f"epoch_{self.epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for mesh_key in keys_list:
            if mesh_key not in batch:
                continue

            kdata = batch[mesh_key]

            # Handle open3d or trimesh objects
            if isinstance(kdata, list):
                self._log_mesh_list(kdata, mesh_key, epoch_dir)

            # Handle point cloud tensors
            elif isinstance(kdata, torch.Tensor):
                self._log_point_cloud(kdata, mesh_key, epoch_dir)

            else:
                raise ValueError(f"Unsupported data type for mesh_key '{mesh_key}'")

        self.iter_counter += 1

    def _log_mesh_list(self, kdata, mesh_key, epoch_dir):
        """
        Handles both open3d and trimesh mesh lists.

        :param kdata: List of mesh objects.
        :param mesh_key: Key for the mesh data.
        :param epoch_dir: Directory to save the mesh files.
        """
        if all(isinstance(m, o3d.geometry.TriangleMesh) for m in kdata):
            for batch_id, mesh in enumerate(kdata):
                # Export .obj mesh files
                filename = os.path.join(
                    epoch_dir, f"{self.iter_counter}_{mesh_key}_{batch_id}.obj"
                )
                o3d.io.write_triangle_mesh(filename, mesh)

                # Add mesh to the logger interface
                self.logger_interface.log_mesh(
                    tag=f"{mesh_key}/{self.phase}",
                    vertices=torch.tensor(mesh.vertices).unsqueeze(0).float(),
                    faces=torch.tensor(mesh.triangles).unsqueeze(0).int(),
                    epoch=self.epoch,
                    config_dict=self._mesh_config(),
                    context={"subset": self.phase},
                )

        elif all(isinstance(m, trimesh.Trimesh) for m in kdata):
            for batch_id, mesh in enumerate(kdata):
                # Export .obj mesh files
                filename = os.path.join(
                    epoch_dir, f"{self.iter_counter}_{mesh_key}_{batch_id}.obj"
                )
                mesh.export(filename)

                # Add mesh to the logger interface
                self.logger_interface.log_mesh(
                    tag=f"{mesh_key}/{self.phase}",
                    vertices=torch.tensor(mesh.vertices).unsqueeze(0).float(),
                    faces=torch.tensor(mesh.faces).unsqueeze(0).int(),
                    epoch=self.epoch,
                    config_dict=self._mesh_config(),
                    context={"subset": self.phase},
                )

        else:
            raise ValueError(f"Incompatible mesh types in list for '{mesh_key}'")

    def _log_point_cloud(self, kdata, mesh_key, epoch_dir):
        """
        Logs point cloud data, supports both 3D (xyz) and 6D (xyzrgb) point clouds.

        :param kdata: Tensor of point cloud data.
        :param mesh_key: Key for the mesh data.
        :param epoch_dir: Directory to save the mesh files.
        """
        if kdata.dim() == 3 and kdata.size(2) in (3, 6):
            for i, pc in enumerate(kdata):
                # Export pointcloud as a .txt file
                np.savetxt(f"{epoch_dir}/{mesh_key}_{i}.txt", pc.detach().cpu().numpy())

                # Add point cloud to the logger interface
                self.logger_interface.log_mesh(
                    tag=f"{mesh_key}/{self.phase}",
                    vertices=pc[..., :3].unsqueeze(0).float(),
                    colors=(
                        pc[..., 3:].unsqueeze(0).float() if pc.size(-1) == 6 else None
                    ),
                    epoch=self.epoch,
                    config_dict={"material": {"cls": "PointsMaterial", "size": 10}},
                    context={"subset": self.phase},
                )
        elif kdata.dim() == 2 and kdata.size(1) in (3, 6):
            # Export pointcloud as a .txt file
            pc = kdata
            np.savetxt(f"{epoch_dir}/{mesh_key}.txt", pc.detach().cpu().numpy())

            # Add point cloud to the logger interface
            self.logger_interface.log_mesh(
                tag=f"{mesh_key}/{self.phase}",
                vertices=pc[..., :3].unsqueeze(0).float(),
                colors=pc[..., 3:].unsqueeze(0).float() if pc.size(-1) == 6 else None,
                epoch=self.epoch,
                config_dict={"material": {"cls": "PointsMaterial", "size": 10}},
                context={"subset": self.phase},
            )
        else:
            assert False, print("Point cloud logger accepts shape B,N,3/6 or N, 3/6")

    def log_epoch(self) -> None:
        """
        Placeholder for logging at the end of each phase.
        """
        self.iter_counter = 0

    def _mesh_config(self):
        """
        Helper function that returns the default configuration dictionary for the mesh.

        :return: Default mesh configuration settings.
        """
        return {
            "camera": {"cls": "PerspectiveCamera", "fov": 75},
            "lights": [
                {"cls": "AmbientLight", "color": "#ffffff", "intensity": 0.7},
                {
                    "cls": "DirectionalLight",
                    "color": "#ffffff",
                    "intensity": 0.65,
                    "position": [0, 2, 0],
                },
            ],
            "material": {"cls": "MeshStandardMaterial", "roughness": 1, "metalness": 0},
        }

    def print_iter_log(self) -> None:
        """
        Prints out relevant logged quantity for the current iteration.
        """
        pass

    def print_epoch_log(self) -> None:
        """
        Prints out relevant logged quantity for the current epoch.
        """
        pass
