import open3d as o3d
import numpy as np
import torch
from unifi3d.utils.data.dataset_utils import sample_points_from_simple_geometries
from unifi3d.utils.data.sdf_utils import generate_query_points
from unifi3d.data.base_dataset import BaseDataset
from unifi3d.utils.data.sdf_utils import (
    sdf_to_mesh,
)


class SimpleGeometries(BaseDataset):
    def __init__(self, mode, config):
        self.num_surface_points = config.num_surface_points
        self.num_query_points = config.num_query_points
        self.data = sample_points_from_simple_geometries(
            num_sample_points=self.num_query_points,
            num_surface_points=self.num_surface_points,
            pt_range=config.range,
            shape_names=config.shape_names,
            fraction_near=config.fraction_near,
            threshold_near=config.threshold_near,
            mode=mode,
        )

    def __getitem__(self, idx):
        data = self.data[idx]
        pcl = data["surface_points"]
        query_points = data["sample_points"]
        occupancy = data["sample_occupancy"]
        sdf = data["sample_sdf"]

        return {
            "pcl": torch.tensor(pcl, dtype=torch.float32),
            "query_points": torch.tensor(query_points, dtype=torch.float32),
            "occupancy": torch.tensor(occupancy, dtype=torch.float32),
            "sdf": torch.tensor(sdf, dtype=torch.float32),
        }

    def representation_to_mesh(self, representation):
        sdf_np = representation.detach().cpu().numpy()
        return sdf_to_mesh(sdf_np, level=0, smaller_values_are_inside=False)
