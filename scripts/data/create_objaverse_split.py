import os
import csv
import random

# Set random seed for reproducibility
random.seed(42)

data_dir = "data/objaverse_preprocessed"
splits = {"train": 0.75, "val": 0.10, "test": 0.15}
output_csv = "data/split_objaverse.csv"

rows = []

for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue
    # List all sample ids (folders) in this category
    ids = [
        d
        for d in os.listdir(category_path)
        if os.path.isdir(os.path.join(category_path, d))
    ]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    n_test = n - n_train - n_val  # Ensure all samples are used

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    for id_ in train_ids:
        rows.append({"id": id_, "split": "train", "category": category})
    for id_ in val_ids:
        rows.append({"id": id_, "split": "val", "category": category})
    for id_ in test_ids:
        rows.append({"id": id_, "split": "test", "category": category})

# Write to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["id", "split", "category"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Split written to {output_csv}")
