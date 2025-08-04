import os
import pandas as pd

CATEGORY = "03001627"
CATEGORY_NAMES = ["chair"]
# Paths
first_dataset_path = f"dataset/{CATEGORY}"
first_split_path = "data/split_shapenet.csv"
second_dataset_path = f"dataset/ShapeNet/{CATEGORY}/"
second_train_path = os.path.join(second_dataset_path, "train.lst")
second_val_path = os.path.join(second_dataset_path, "val.lst")
second_test_path = os.path.join(second_dataset_path, "test.lst")


# Read IDs from the second dataset
def read_ids_from_file(file_path):
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)


# Compare splits
def compare_splits(first_split, second_split, split_name):
    first_split_ids = set(
        first_split_df[first_split_df["split"] == split_name]["id"].tolist()
    )
    second_split_ids = read_ids_from_file(
        os.path.join(second_dataset_path, f"{split_name}.lst")
    )

    not_in_second = first_split_ids - second_split_ids
    not_in_first = second_split_ids - first_split_ids
    print(
        f"IDs in the first {split_name} split but not in the second {split_name} split:",
        not_in_second,
    )
    print(
        f"IDs in the second {split_name} split but not in the first {split_name} split:",
        not_in_first,
    )


def create_file_list_from_table(data_split, mode):
    """Just filter table for correct mode and categories"""
    # filter for mode
    filtered_file_path_list = data_split[data_split["split"] == mode]
    # filter for categories
    if len(CATEGORY_NAMES) > 0:
        # sanity checking that all categories are actually in the dataset
        for cat in CATEGORY_NAMES:
            if cat not in filtered_file_path_list["category"].unique():
                raise ValueError(f"Category {cat} not found in dataset.")

        filtered_file_path_list = filtered_file_path_list[
            filtered_file_path_list["category"].isin(CATEGORY_NAMES)
        ]

    return filtered_file_path_list


if __name__ == "__main__":
    # Read IDs from the first dataset
    first_split_df = pd.read_csv(first_split_path)
    first_train_ids = set(create_file_list_from_table(first_split_df, "train")["id"])
    first_val_ids = set(create_file_list_from_table(first_split_df, "val")["id"])
    first_test_ids = set(create_file_list_from_table(first_split_df, "test")["id"])
    first_ids = first_train_ids | first_val_ids | first_test_ids

    second_train_ids = read_ids_from_file(second_train_path)
    second_val_ids = read_ids_from_file(second_val_path)
    second_test_ids = read_ids_from_file(second_test_path)
    second_ids = second_train_ids | second_val_ids | second_test_ids

    print(first_ids & second_ids)
    # Compare IDs
    first_not_in_second = first_ids - second_ids
    second_not_in_first = second_ids - first_ids

    print("IDs in the first dataset but not in the second:", first_not_in_second)
    print("IDs in the second dataset but not in the first:", second_not_in_first)

    # compare_splits(first_split_df, second_train_ids, "train")
    # compare_splits(first_split_df, second_val_ids, "val")
    # compare_splits(first_split_df, second_test_ids, "test")
