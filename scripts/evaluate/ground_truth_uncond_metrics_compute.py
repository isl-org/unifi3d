"""Script for computing the unconditional metrics on the ground truth data"""

import sys
import json
import os
from typing import Any, Optional, List
from pathlib import Path
import numpy as np
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from unifi3d.data.data_iterators import (
    ShapenetPreprocessedIterator as DatasetIterator,
)
from unifi3d.utils.evaluate.metrics.mesh_distribution_metrics import (
    MeshDistributionMetrics,
)
import open3d as o3d


def fail_on_missing(cfg: Any) -> None:
    if isinstance(cfg, ListConfig):
        for x in cfg:
            fail_on_missing(x)
    elif isinstance(cfg, DictConfig):
        for _, v in cfg.items():
            fail_on_missing(v)


def parse_args():
    config_str = """
        # Output directory
        output_dir: ???
        
        # Categories on which to cmopute the metrics
        categories: [airplane]

        # The number of samples to take from the generated and reference set for 
        # computing the metrics
        set_sizes: [10,50,100,200,400]

        # The number of samples to use for computing the stats like the variance
        id: ???
        """
    config = OmegaConf.create(config_str)
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Default config:\n", config_str)
        sys.exit(1)

    config = OmegaConf.merge(config, OmegaConf.from_cli())
    try:
        fail_on_missing(config)
    except omegaconf.errors.MissingMandatoryValue as exception:
        print(exception)
        sys.exit(1)

    return config


def compute(
    all_train_mesh_paths: List[str], all_test_mesh_paths: List[str], set_size: int, rng
):
    print(f"{set_size=}")

    test_mesh_paths = rng.choice(all_test_mesh_paths, set_size, replace=False)
    train_mesh_paths = rng.choice(all_train_mesh_paths, set_size, replace=False)

    def load_fn(x):
        tmp = np.load(x)
        return o3d.t.geometry.TriangleMesh(tmp["vertices"], tmp["triangles"])

    test_meshes = [load_fn(x) for x in test_mesh_paths]
    train_meshes = [load_fn(x) for x in train_mesh_paths]

    metrics_comp = MeshDistributionMetrics(show_progress=True, return_matrices=True)
    metrics = metrics_comp(generated=train_meshes, reference=test_meshes)
    return metrics


def main():
    cfg = parse_args()
    print(cfg)
    print(DatasetIterator.__name__)
    dataset_name = DatasetIterator.__name__.replace("Iterator", "")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in cfg.categories:
        train_iter = DatasetIterator(mode="train", category=category)
        all_train_mesh_paths = [x["mesh_path"] for x in train_iter]

        test_iter = DatasetIterator(mode="test", category=category)
        all_test_mesh_paths = [x["mesh_path"] for x in test_iter]

        for set_size in cfg.set_sizes:
            output_path = (
                output_dir
                / f"{dataset_name}_{category}_setsize{set_size}_id{cfg.id}.json"
            )
            if output_path.exists():
                print(f"skipping {str(output_path)}")
            else:
                rng = np.random.default_rng(abs(hash((cfg.id, set_size))))
                metric = compute(
                    all_train_mesh_paths=all_train_mesh_paths,
                    all_test_mesh_paths=all_test_mesh_paths,
                    set_size=set_size,
                    rng=rng,
                )
                metric["dataset_name"] = dataset_name
                metric["category"] = category
                metric["set_size"] = set_size
                metric["id"] = cfg.id
                del metric["distance_matrix"]
                del metric["chamfer_distance_matrix"]
                with open(output_path, "w") as f:
                    json.dump(metric, f, indent=2)


if __name__ == "__main__":
    main()
