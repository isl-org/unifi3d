from pathlib import Path
import numpy as np
import pandas as pd
import os
import arxiv
import json


def add_category(data_split, shapenet_path):
    # get categories
    with open(os.path.join(shapenet_path, "taxonomy.json"), "r") as f:
        taxonomy = json.load(f)
    # make mapping from category ID to category
    id_cat_mapping = {}
    for cat in taxonomy:
        cat_name = cat["name"].split(",")[0]
        ids = [cat["synsetId"]] + cat["children"]
        for idd in ids:
            if idd not in id_cat_mapping.keys():
                id_cat_mapping[idd] = cat_name

    data_split["category"] = ("0" + data_split["synsetId"].astype(str)).map(
        id_cat_mapping
    )
    print(
        "Number of samples where no category could be assigend:",
        data_split["category"].isna().sum(),
    )
    return data_split


if __name__ == "__main__":
    shapenet_path = "datasets/ShapeNetCore.v1/"
    path = Path("dataset/")
    out_path = "unifi3d/data/split_shapenet.csv"

    # load split
    data_split = pd.read_csv("unifi3d/data/raw_split_shapenet.csv", index_col="modelId")

    # add category label
    data_split = add_category(data_split, shapenet_path)

    path_dict = {}

    # iterate over all files of the dataset and add to dict
    for cat_path in path.iterdir():
        if not cat_path.is_dir():
            continue
        for file in cat_path.iterdir():
            # check if is dir
            if not file.is_dir():
                continue
            # check if file is in split
            if file.name not in data_split.index:
                continue

            with open(file / "info.json", "r", encoding="latin-1") as f:
                info = json.load(f)
            info["base_dir"] = str(file)
            path_dict[file.name] = info

    # merge dict and table
    new_info = pd.DataFrame(path_dict).swapaxes(1, 0)
    data_split_w_info = pd.merge(
        data_split.drop("id", axis=1),
        new_info,
        left_index=True,
        right_index=True,
        how="inner",
    )
    print(
        "Original number of files in split",
        len(data_split),
        "after preprocessing",
        len(data_split_w_info),
    )

    data_split_w_info.to_csv(out_path, index=False)
