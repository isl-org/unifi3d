import numpy as np
import os
from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from unifi3d.data.base_dataset import BaseDataset
from unifi3d.utils.data.doctree_utils import create_mesh


class TransformShape:

    def __init__(self, config):
        self.distort = config.distort
        self.load_sdf = config.load_sdf
        self.sample_surf_points = config.sample_surf_points

        self.point_sample_num = 3000
        self.sdf_sample_num = 5000
        self.points_scale = 0.5  # the points are in [-0.5, 0.5]
        self.noise_std = 0.005

        self.depth = config.depth
        self.full_depth = config.full_depth

    def process_points_cloud(self, sample):
        # get the input
        points, normals = sample["points"], sample["normals"]
        points = points / self.points_scale  # scale to [-1.0, 1.0]

        # transform points to octree
        points_gt = Points(
            points=torch.from_numpy(points).float(),
            normals=torch.from_numpy(normals).float(),
        )
        points_gt.clip(min=-1, max=1)
        octree_gt = Octree(depth=self.depth, full_depth=self.full_depth)
        octree_gt.build_octree(points_gt)

        if self.distort:
            # randomly sample points and add noise
            # Since we rescale points to [-1.0, 1.0] in Line 24, we also need to
            # rescale the `noise_std` here to make sure the `noise_std` is always
            # 0.5% of the bounding box size.
            noise_std = self.noise_std / self.points_scale
            noise = noise_std * np.random.randn(self.point_sample_num, 3)
            rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
            points_noise = points[rand_idx] + noise

            points_in = Points(
                points=torch.from_numpy(points_noise).float(),
                labels=torch.ones(self.point_sample_num).float(),
            )
            points_in.clip(min=-1, max=1)
            octree_in = Octree(depth=self.depth, full_depth=self.full_depth)
        else:
            points_in = points_gt
            octree_in = octree_gt
        # construct the output dict
        return {
            "points_in": points_in,
            "points_gt": points_gt,
            "octree_in": octree_in,
            "octree_gt": octree_gt,
        }

    def sample_sdf(self, sample):
        sdf = sample["sdf"]
        grad = sample["grad"]
        points = sample["points"] / self.points_scale  # to [-1, 1]

        rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
        points = torch.from_numpy(points[rand_idx]).float()
        sdf = torch.from_numpy(sdf[rand_idx]).float()
        grad = torch.from_numpy(grad[rand_idx]).float()
        return {"pos": points, "sdf": sdf, "grad": grad}

    def sample_on_surface(self, points, normals):
        rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
        xyz = torch.from_numpy(points[rand_idx]).float()
        grad = torch.from_numpy(normals[rand_idx]).float()
        sdf = torch.zeros(self.sdf_sample_num)
        return {"pos": xyz, "sdf": sdf, "grad": grad}

    def sample_off_surface(self, xyz):
        xyz = xyz / self.points_scale  # to [-1, 1]

        rand_idx = np.random.choice(xyz.shape[0], size=self.sdf_sample_num)
        xyz = torch.from_numpy(xyz[rand_idx]).float()
        # grad = torch.zeros(self.sample_number, 3)  # dummy grads
        grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
        sdf = -1 * torch.ones(self.sdf_sample_num)  # dummy sdfs
        return {"pos": xyz, "sdf": sdf, "grad": grad}

    def __call__(self, sample):
        output = self.process_points_cloud(sample["point_cloud"])

        # sample ground truth sdfs
        if self.load_sdf:
            sdf_samples = self.sample_sdf(sample["sdf"])
            output.update(sdf_samples)

        # sample on surface points and off surface points
        if self.sample_surf_points:
            on_surf = self.sample_on_surface(sample["points"], sample["normals"])
            off_surf = self.sample_off_surface(sample["sdf"]["points"])
            sdf_samples = {
                "pos": torch.cat([on_surf["pos"], off_surf["pos"]], dim=0),
                "grad": torch.cat([on_surf["grad"], off_surf["grad"]], dim=0),
                "sdf": torch.cat([on_surf["sdf"], off_surf["sdf"]], dim=0),
            }
            output.update(sdf_samples)
        return output


class ReadFile:
    def __init__(self, load_sdf=False, load_occu=False, file_number=-1, depth=6):
        self.load_occu = load_occu
        self.load_sdf = load_sdf
        self.num_files = file_number
        self.depth = depth

    def get_pc_file_helper(self, filename, num=0):
        if num >= 0:
            filename_pc = os.path.join(filename, "pointcloud/pointcloud_%02d.npz" % num)
        else:
            filename_pc = os.path.join(filename, "pointcloud_boundary.npz")
        assert os.path.exists(filename_pc), f"{filename_pc} does not exist!"
        return filename_pc

    def get_occu_file_helper(self, filename, num=0):
        if num >= 0:
            filename_occu = os.path.join(
                filename, "points_iou/points_iou_%02d.npz" % num
            )
        else:
            filename_occu = os.path.join(filename, "points_iou.npz")
        assert os.path.exists(filename_occu), f"{filename_occu} does not exist!"
        return filename_occu

    def __call__(self, filename):
        num = -1
        if self.num_files > 0:
            num = np.random.randint(self.num_files)
        filename_pc = self.get_pc_file_helper(filename, num)
        raw = np.load(filename_pc)
        point_cloud = {"points": raw["points"], "normals": raw["normals"]}
        output = {"point_cloud": point_cloud}

        if self.load_occu:
            if self.num_files > 0:
                num = np.random.randint(self.num_files)

            filename_occu = self.get_occu_file_helper(filename, num)
            raw = np.load(filename_occu)
            occus = {"points": raw["points"], "occupancies": raw["occupancies"]}
            output["occus"] = occus

        if (
            self.load_sdf
        ):  # synthetic room does not load sdf.npz, so we do not need to use num_files here.
            filename_sdf = os.path.join(filename, f"sdf_octree_depth_{self.depth}.npz")
            assert os.path.exists(filename_sdf), f"{filename_sdf} does not exist!"
            raw = np.load(filename_sdf)
            sdf = {"points": raw["points"], "grad": raw["grad"], "sdf": raw["sdf"]}
            output["sdf"] = sdf
        return output


def collate_func(batch):
    collate_batch = CollateBatch(merge_points=False)
    output = collate_batch(batch)

    if "pos" in output:
        batch_idx = torch.cat(
            [torch.ones(pos.size(0), 1) * i for i, pos in enumerate(output["pos"])],
            dim=0,
        )
        pos = torch.cat(output["pos"], dim=0)
        output["pos"] = torch.cat([pos, batch_idx], dim=1)
    for key in ["grad", "sdf", "occu", "weight"]:
        if key in output:
            output[key] = torch.cat(output[key], dim=0)
    if "pos" in output:
        output["query_points"] = output["pos"][:, :3]
    return output


class DualOctreeDataset(BaseDataset):
    """
    Container for the DualOctree dataset.
    """

    def __init__(self, dataset, mode, config, read_file, transform):
        """
        Initializes the dataset object with configuration and mode.
        """
        super(DualOctreeDataset, self).__init__(dataset)
        self.read_file = read_file
        self.transform = transform
        self.create_mesh_config = config.create_mesh_config
        self.mode = mode
        if self.mode not in ["train", "val", "test"]:
            raise NotImplementedError

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.
        """
        data = self.data[index]
        # print("data keys", data.keys())
        # for key in data:
        #     print(key, data[key])
        if "base_dir" in data:
            occtree = self.read_file(data["base_dir"])
        else:
            assert (
                "file_path" in data
            ), "data needs to contain either base_dir or file_path keys."
            occtree = self.read_file(data["file_path"])

        output_data = self.transform(occtree)
        if "text" in data:
            output_data["text"] = data["text"]
        if "image" in data:
            output_data["image"] = data["image"]
        if "mesh_path" in data:
            output_data["mesh_path"] = data["mesh_path"]
        if "base_dir" in data:
            output_data["base_dir"] = data["base_dir"]
        elif "file_name" in data:
            output_data["file_name"] = data["file_name"]
        return output_data

    # def mesh_to_representation(self, data):
    def representation_to_mesh(self, data):
        bbox = data["bbox"][0].numpy() if "bbox" in data else None
        o3d_mesh = create_mesh(
            model=data["neural_mpu"],
            bbox=bbox,
            sdf_scale=self.create_mesh_config.sdf_scale,
            size=self.create_mesh_config.size,
            mesh_scale=self.create_mesh_config.mesh_scale,
            export_type="o3d",
            return_sdf=False,
        )

        return o3d_mesh


class DualOctreeDataModule(nn.Module):
    def __init__(
        self,
        num_workers,
        data_train=None,
        data_val=None,
        data_test=None,
        train_batch_size=16,
        val_batch_size=1,
        test_batch_size=1,
        train_collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt

        self.data_train: Optional[Dataset] = data_train
        self.data_val: Optional[Dataset] = data_val
        self.data_test: Optional[Dataset] = data_test

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_collate_fn = train_collate_fn
        self.val_collate_fn = val_collate_fn
        self.test_collate_fn = test_collate_fn

    def prepare_data(self):
        """
        Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        # TODO: put preprocessing script here.
        pass

    def setup(self):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # We are now passing data_train, data_val and data_test directly as part of input
        # as they may not share as much commonalities based on the synthetic_room dataset example.
        pass

    # TODO: Added dummy function to accomadate benchmarking.py function.
    def setup_inference(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.val_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.test_collate_fn,
            pin_memory=True,
        )
