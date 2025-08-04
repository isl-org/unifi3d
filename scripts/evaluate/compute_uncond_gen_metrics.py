import json
import os
import hydra
from omegaconf import DictConfig
from typing import Optional
from unifi3d.utils.evaluate.unconditional_gen import evaluate_unconditional_generation


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="sample_diffusion.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """
    E.g. run with
    python scripts/evaluate/compute_uncond_gen_metrics.py
    output_path=logs/final_results_Diffusion/Airplane/sdf_ae_dit
    reference_dataset.categories=["airplane"]

    This will redo the unconditional metric computation, and save it in
    output_path/metrics.json
    """

    print(cfg)
    # load diffusion model memory footprint
    info_to_save = {}
    existing_metric_path = os.path.join(cfg.output_path, "metrics.json")
    if os.path.exists(existing_metric_path):
        with open(existing_metric_path, "r", encoding="latin-1") as f:
            memory_info = json.load(f)
        info_to_save["diffusion model"] = memory_info.get("diffusion model", {})
    print(info_to_save)

    #  = "outputs/dit_sdf/generated_samples"
    evaluate_unconditional_generation(
        cfg,
        os.path.join(cfg.output_path, "generated_samples"),
        info_to_save=info_to_save,
    )


if __name__ == "__main__":
    main()
