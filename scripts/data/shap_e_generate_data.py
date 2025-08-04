import argparse
import os
from pathlib import Path
import sys

import torch

from unifi3d.utils.data.shap_e_utils import load_or_create_multimodal_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates prompts for a ground truth mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default="shapenet_airplane/models/model_normalized.obj",
        help="Path to mesh",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default="unifi3d/data/shapenet_for_shap_e",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if "--" in sys.argv:
        args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        args = parser.parse_args()

    fpath = args.input.parts
    cache_dir = os.path.join(args.cache_path, fpath[-3] + "/" + fpath[-2])

    device = torch.device("cuda")
    pcl_image = load_or_create_multimodal_batch(
        device,
        model_path=args.input,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir=cache_dir,
        verbose=False,
    )
