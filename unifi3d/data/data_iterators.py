import warnings
import os
import open3d as o3d
import json
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import omegaconf
from typing import List
import logging
from collections import Counter


class BaseIterator:
    def __init__(self, path: str, mode: str = "test"):
        """Iterator over a dataset of meshes stored in a directory."""
        self.path = path
        self.mode = mode
        self.file_list = []

    def __len__(self):
        """Computes length of dataset"""
        return len(self.file_list)

    def __getitem__(self, index):
        """Returns directory with file path and text label for sample at this index"""
        pass


class DemoIterator(BaseIterator):
    """
    Iterator for a simple directory of obj files without any subdirectories
    """

    def __init__(self, path: str, mode: str = "test"):
        """
        Note: for the demo dataset, there are just 9 meshes, so no distinction between train and test
        """
        super(DemoIterator, self).__init__(path, mode)

        # create list of object files in directory
        self.file_list = [
            f for f in os.listdir(self.path) if f.endswith((".obj", ".glb", ".off"))
        ]

    def __getitem__(self, index):
        filename = self.file_list[index]
        file_path = os.path.join(self.path, filename)
        text, _ = os.path.splitext(filename)
        return {"text": text, "file_path": file_path, "id": text}


class ShapenetIterator(BaseIterator):

    def __init__(
        self,
        path: str,
        mode: str = "test",
        categories: List[str] = [],
        split_path="data/split_shapenet.csv",
    ):
        """
        Create list of shapenet object paths with corresponding labels
        TODO: distinguish train and test dataset
        """
        super(ShapenetIterator, self).__init__(path, mode)

        self.categories = categories
        self.split_path = split_path

        # load split
        data_split = pd.read_csv(split_path, index_col="id")
        cat_id_mapping = data_split.groupby("synsetId")["category"].first().to_dict()
        cat_id_mapping[4256520] = "sofa"
        cat_id_mapping[2828884] = "bench"

        self.file_list, self.text_labels = [], []
        filename_set = set()

        # iterate over category directories
        for cat_id_dir in sorted(os.listdir(path)):
            category_path = os.path.join(path, cat_id_dir)
            if not os.path.isdir(category_path):
                continue

            # check whether valid category
            try:
                cat_name = cat_id_mapping[int(cat_id_dir[1:])]
            except KeyError:
                warnings.warn("Category not found")
                continue
            if len(categories) > 0 and cat_name not in categories:
                continue

            # load semantic info
            tab = pd.read_csv(os.path.join(path, cat_id_dir + ".csv"))
            tab["id"] = tab["fullId"].str.split(".").str[1]
            text_label_dict = tab.set_index("id")["wnlemmas"].to_dict()

            for filename in sorted(os.listdir(category_path)):
                if filename[0] == ".":
                    continue

                # check if file exists in data split
                try:
                    sample_mode = data_split.loc[filename, "split"]
                except:
                    continue
                # check if file is already in list (some seem to be duplicates)
                if filename in filename_set:
                    continue

                if sample_mode == mode or mode == "all":
                    # get text label if available
                    text_label = text_label_dict.get(filename, "unknown")

                    self.text_labels.append(text_label)
                    self.file_list.append(
                        os.path.join(category_path, filename, "model.obj")
                    )

    def __getitem__(self, index: int):
        file_path = self.file_list[index]
        text = self.text_labels[index]
        return {"text": text, "file_path": file_path, "id": file_path.split(os.sep)[-2]}

    @staticmethod
    def categories():
        return [
            "airplane",
            "ashcan",
            "bag",
            "basket",
            "bathtub",
            "bed",
            "bench",
            "birdhouse",
            "bookshelf",
            "bottle",
            "bowl",
            "bus",
            "cabinet",
            "camera",
            "can",
            "cap",
            "car",
            "chair",
            "clock",
            "computer keyboard",
            "dishwasher",
            "display",
            "earphone",
            "faucet",
            "file",
            "guitar",
            "helmet",
            "jar",
            "knife",
            "lamp",
            "laptop",
            "loudspeaker",
            "mailbox",
            "microphone",
            "microwave",
            "motorcycle",
            "mug",
            "piano",
            "pillow",
            "pistol",
            "pot",
            "printer",
            "remote control",
            "rifle",
            "rocket",
            "skateboard",
            "sofa",
            "stove",
            "table",
            "telephone",
            "tower",
            "train",
            "vessel",
            "washer",
        ]


class SyntheticRoomIterator(BaseIterator):
    def __init__(self, path: str, mode: str = "test", num_samples: int = -1):
        """
        Iterate through Synthetic Room paths and corresponding labels
        """
        super(SyntheticRoomIterator, self).__init__(path, mode)
        if mode in ["train", "val", "test"]:
            data_root = os.path.join(path, "synthetic_room_dataset")
        elif mode == "eval":
            data_root = os.path.join(path, "test.input")
        else:
            raise f"{mode} dataset is not implemented!"
        filelist_txt = os.path.join(path, "filelist", f"{mode}.txt")
        self.filename_list, self.file_list, self.text_labels = [], [], []
        with open(filelist_txt) as fid:
            lines = fid.readlines()
        self.mode = mode
        if num_samples > 0:
            print(
                f"Only collecting {num_samples} samples for {mode}. Set num_samples to -1 to load the entire dataset."
            )

        for line in lines:
            if num_samples > 0 and len(self.file_list) == num_samples:
                # Only collect enough for faster debugging.
                break
            tokens = line.split()
            filename = tokens[0]
            label = tokens[1] if len(tokens) == 2 else 0
            file_path = os.path.join(data_root, filename)
            self.filename_list.append(filename)
            self.file_list.append(file_path)
            self.text_labels.append(label)
        print(f"Found total {len(self.file_list)} files.")

    def __getitem__(self, index: int):
        filename = self.filename_list[index]
        file_path = self.file_list[index]
        text = self.text_labels[index]
        return {"text": text, "file_path": file_path, "file_name": filename}


class PartNetIterator(BaseIterator):
    def __init__(
        self,
        data_path: str,
        data_class: str = "all",
        mode: str = "test",
    ):
        with open(
            os.path.join(data_path, "info_partnetmobility.json"),
            "r",
            encoding="latin-1",
        ) as inf:
            partnet_info = json.load(inf)
        with open(
            os.path.join(data_path, "splits_partnetmobility.json"),
            "r",
            encoding="latin-1",
        ) as inf:
            splits = json.load(inf)

        if data_class == "all":
            cats = partnet_info["all_cats"]
        else:
            cats = [data_class]

        self.file_list = []
        for cat in cats:
            objects_in_cat = splits[cat][mode]
            for mesh_id in objects_in_cat:
                self.file_list.append(os.path.join(data_path, cat, mesh_id + ".obj"))

    def __getitem__(self, index):
        file_path = self.file_list[index]
        text = file_path.split(os.sep)[-2]  # use category
        return {
            "file_path": file_path,
            "text": text,
            "id": file_path.split(os.sep)[-1].split(".")[0],
        }


class ShapenetPreprocessedIterator(BaseIterator):
    """
    Iterator for the preprocessed shapenetcore.v1 dataset
    """

    def __init__(
        self,
        path: str = "data/shapenet_preprocessed",
        original_path: str = "data/ShapeNetCore.v1",
        mode: str = "test",
        num_samples=-1,
        categories=[],
        split_path="data/split_shapenet.csv",
        rendered_img_path="data/shapenet2/renderings",
        rendered_img_file=None,
        is_val=False,
    ):
        """
        Arguments:
            rendered_img_path: path to rendered images (for image conditioning)
            rendered_img_file: img filename (if only one perspective should be used.
                By default, one image is randomly selected from the folder)
        """
        mode = "val" if is_val else mode  # dirty hack for overfitting on triplanes
        if is_val:
            logging.warning(
                f"Using is_val = True. Ensure this is intended. Only during overfitting!"
            )
        super().__init__(path, mode)
        self.categories = categories
        self.rendered_img_path = rendered_img_path
        self.rendered_img_file = rendered_img_file

        # load split
        self.data_split = pd.read_csv(split_path)
        # add base_dir column -> this is the path to the preprocessed data folder
        self.data_split["base_dir"] = self.data_split.apply(
            lambda row: os.path.join(path, "0" + str(row["synsetId"]), str(row["id"])),
            axis=1,
        )
        # add file_path column -> this is the path to the original mesh file
        self.data_split["file_path"] = self.data_split.apply(
            lambda row: os.path.join(
                original_path, "0" + str(row["synsetId"]), str(row["id"]), "model.obj"
            ),
            axis=1,
        )

        # paths are already precomputed -> just load them
        if "base_dir" in self.data_split.columns:
            self.create_file_list_from_table()
        else:
            self.create_file_list_from_scratch()

        if num_samples > 0 and len(self.filtered_file_path_list) > num_samples:
            self.file_list = self.filtered_file_path_list[:num_samples]
        else:
            self.file_list = self.filtered_file_path_list
        print(
            f"Requested {num_samples} has {len(self.filtered_file_path_list)}/{len(self.file_list)} files."
        )
        print(f"Found {len(self.file_list)} files.")

    def create_file_list_from_table(self):
        """Just filter table for correct mode and categories"""
        # filter for mode
        self.filtered_file_path_list = self.data_split[
            self.data_split["split"] == self.mode
        ]
        # filter for categories
        if len(self.categories) > 0:

            if not (
                isinstance(self.categories, list)
                or isinstance(self.categories, omegaconf.listconfig.ListConfig)
            ):
                raise TypeError(
                    "Categories need to be a list of strings. Other types are not supported."
                )

            # sanity checking that all categories are actually in the dataset
            for cat in self.categories:
                if cat not in self.filtered_file_path_list["category"].unique():
                    raise ValueError(f"Category {cat} not found in dataset.")

            self.filtered_file_path_list = self.filtered_file_path_list[
                self.filtered_file_path_list["category"].isin(self.categories)
            ]

    def create_file_list_from_scratch(self):
        self.data_split.set_index("modelId", inplace=True)
        path = Path(self.path)
        path_dict = []

        # iterate over all files of the dataset
        for cat_path in path.iterdir():
            if not cat_path.is_dir():
                continue
            for file in cat_path.iterdir():
                # check if is dir
                if not file.is_dir():
                    continue
                # check if file is in split
                if file.name not in self.data_split.index:
                    continue
                split_row = self.data_split.loc[file.name]

                # filter for category and split
                if (
                    len(self.categories) == 0
                    or split_row["category"] in self.categories
                ) and split_row["split"] == self.mode:
                    # load info
                    with open(file / "info.json", "r") as f:
                        info = json.load(f)
                        info["base_dir"] = str(file)
                        path_dict.append(info)

        self.filtered_file_path_list = pd.DataFrame(path_dict)

    def __getitem__(self, index):
        # get row for index
        file_info = self.file_list.iloc[index].to_dict()
        fp = file_info["base_dir"]
        # Dataset-specific folder structure
        file_info["sdf_grid128_path"] = os.path.join(fp, "sdf_grid_128.npz")
        file_info["sdf_grid256_path"] = os.path.join(fp, "sdf_grid_256.npz")
        file_info["sdf_octree_path"] = os.path.join(fp, "sdf_octree_depth_6.npz")
        file_info["pointcloud_path"] = os.path.join(fp, "pointcloud_boundary.npz")
        file_info["points_iou_path"] = os.path.join(fp, "points_iou.npz")
        file_info["mesh_path"] = os.path.join(fp, "mesh.npz")
        # conditioning image info
        file_info["image"] = os.path.join(
            self.rendered_img_path, fp.split(os.sep)[-2], fp.split(os.sep)[-1]
        )
        if self.rendered_img_file is not None:  # add filename
            file_info["image"] = os.path.join(
                file_info["image"], self.rendered_img_file
            )
        return file_info


class ObjaversePreprocessedIterator(ShapenetPreprocessedIterator):
    """
    Iterator for the preprocessed Objaverse dataset
    """

    def __init__(
        self,
        path: str = "data/objaverse_preprocessed",
        mode: str = "test",
        categories=[],
        split_path="data/split_objaverse.csv",
        rendered_img_path="data/objaverse_renderings",
        rendered_img_file=None,
        num_samples=-1,
        is_val=False,
    ):
        self.path = path
        self.mode = mode
        self.categories = categories
        self.rendered_img_path = rendered_img_path
        self.rendered_img_file = rendered_img_file

        # load split
        self.data_split = pd.read_csv(split_path)
        # add base_dir column -> this is the path to the preprocessed data folder
        self.data_split["base_dir"] = self.data_split.apply(
            lambda row: os.path.join(path, str(row["category"]), str(row["id"])),
            axis=1,
        )

        # filter for categories etc
        self.create_file_list_from_table()

        # add path to original file
        def get_original_path(row):
            with open(
                os.path.join(
                    self.path,
                    str(row["category"]),
                    str(row["id"]),
                    "info.json",
                ),
                "r",
                encoding="latin-1",
            ) as f:
                info = json.load(f)
                return info["file_path"]

        self.filtered_file_path_list["file_path"] = self.filtered_file_path_list.apply(
            get_original_path, axis=1
        )

        # restrict to less samples if desired
        if num_samples > 0 and len(self.filtered_file_path_list) > num_samples:
            self.file_list = self.filtered_file_path_list[:num_samples]
        else:
            self.file_list = self.filtered_file_path_list
        print(
            f"Requested {num_samples} has {len(self.filtered_file_path_list)}/{len(self.file_list)} files."
        )
        print(f"Found {len(self.file_list)} files.")


class ComplexIterator(BaseIterator):
    """
    Iterator for the preprocessed complex dataset
    """

    def __init__(
        self,
        path: str = "data/complex_objects/",
        mode: str = "train",
        categories=["cube"],
    ):
        """
        Arguments:
            path: path to object directories
            mode: "train", "test", or "val"
            num_samples: number of sample points.
            categories: can be any of ["cube", "sphere", "mandelbulb_1", "mandelbulb_2", "mandelbulb_3"]
        """
        super().__init__(path, mode)
        self.file_list = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name)) and name in categories
        ]

    def __getitem__(self, index):
        # get row for index
        fp = self.file_list[index]
        # Dataset-specific folder structure
        folder_name = os.path.basename(os.path.normpath(fp))
        return {
            "base_dir": fp,
            "file_path": os.path.join(fp, f"{folder_name}.obj"),
            "sdf_grid128_path": os.path.join(fp, "sdf_grid_128.npz"),
            "sdf_grid256_path": os.path.join(fp, "sdf_grid_256.npz"),
            "sdf_octree_path": os.path.join(fp, "sdf_octree_depth_6.npz"),
            "pointcloud_path": os.path.join(fp, "pointcloud_boundary.npz"),
            "points_iou_path": os.path.join(fp, "points_iou.npz"),
            "mesh_path": os.path.join(fp, "mesh.npz"),
        }


class ObjaverseIterator(BaseIterator):

    def __init__(
        self,
        path: str,
        json_path: str,
        mode: str = "test",
        categories: List[str] = [],
    ):
        """Iterate through Objaverse1.0 object paths and corresponding labels

        Args:
            path: Base path to the objaverse dataset. This is the path to the hf-objaverse-v1 directory.
            json_path: Path to the json file containing the metadata.
            mode: Does currently not have any effect.

        """
        super(ObjaverseIterator, self).__init__(path, mode)

        with open(json_path, "r", encoding="utf-8") as f:
            self.file_list = json.load(f)

        self.categories = categories
        if categories:
            # filter file_list for categories
            file_list = []
            self.category_frequencies = Counter()
            for x in self.file_list:
                category = x["lvis_category"]
                if category in categories:
                    file_list.append(x)
                if category is not None:
                    self.category_frequencies.update([category])
            self.file_list = file_list

            for c in categories:
                if c not in self.category_frequencies:
                    raise ValueError(
                        f"Category {c} not found in dataset. Available categories: {self.category_frequencies.most_common()}"
                    )

    def __getitem__(self, index: int):
        sample_info = self.file_list[index]
        result = {
            "file_path": os.path.join(self.path, sample_info["path"]),
            "text": sample_info["caption"],
            "id": sample_info["id"],
        }
        if self.categories:
            # only add category if categories are specified
            result["category"] = sample_info["lvis_category"]
        return result


class TestIterator(BaseIterator):
    """Iterator for testing on a single example"""

    def __init__(
        self, path: str = "o3d_knotmesh", mode: str = "test", n_repeat_sample=30
    ):
        super().__init__(path, mode)
        # this downloads the knot mesh from open3d (if not already downloaded)
        knot_data = o3d.data.KnotMesh()
        # to train on this mesh, we repeat it several times (to avoid epoch of 1 sample)
        self.file_list = [knot_data.path for _ in range(n_repeat_sample)]

    def __getitem__(self, index):
        file_path = self.file_list[index]
        return {"text": "knot", "file_path": file_path, "id": 0, "file_name": "knot"}
