"""Script for preparing the data for the user study (copy and convert to gltf)"""

import sys
import json
import os
from typing import Any, Optional, List
from pathlib import Path
import numpy as np
import omegaconf
from omegaconf import OmegaConf, DictConfig, ListConfig
import rootutils

root_path = rootutils.find_root(__file__, indicator=".project-root")
rootutils.set_root(path=root_path, pythonpath=True)

from unifi3d.utils.data.convert_to_gltf import convert_to_gltf


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
        
        # The maximum number of objects to use for each method
        limit: 25
        
        # Add data by adding key value pairs that point to the output directory for the method.
        # Use the method name as key.
        # methods:
        #   method_name: /path/to/method/output
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


def get_obj_paths(path: Path | str, limit: int):
    """Searches for the obj file paths using hardcoded guesses.
    Args:
        path: Path to search for .obj files
        limit: The maximum number of paths to return
    Returns:
        A list of paths to .obj files
    """
    path = Path(path)
    generated_samples_path = path / "generated_samples"
    if generated_samples_path.exists():
        path = generated_samples_path
    obj_paths = sorted(list(path.glob("*.obj")))[:limit]
    return obj_paths


def main():
    cfg = parse_args()
    print(cfg)
    output_dir = Path(cfg.output_dir) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for method_name, path in cfg.methods.items():
        print(method_name, path)

        obj_paths = get_obj_paths(path, cfg.limit)
        print(obj_paths)

        for obj_path in obj_paths:
            target_path = (
                output_dir / method_name / Path(obj_path.name).with_suffix(".glb")
            )
            print(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            convert_to_gltf(obj_path, target_path)


if __name__ == "__main__":
    main()
