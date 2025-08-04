import os
import json
import pandas as pd
from unifi3d.utils.evaluate.metrics.mesh_distribution_metrics import (
    MeshDistributionMetrics,
)
from scipy.stats import pearsonr

from unifi3d.utils.data.mesh_utils import (
    load_reference_dataset,
    load_generated_dataset,
)


def relate_to_recon(gen_err, output_path):
    gen_err_df = pd.DataFrame(
        list(gen_err.items()), columns=["file_path", "chamfer_distance_gen"]
    )

    # the 1-MMD uses 0.5 in the CD, so we need to multiply by 2 to make it comparable
    gen_err_df["chamfer_distance_gen"] *= 2

    # load encoder errors (change path to AE results)
    path_enc_results = os.path.join(
        output_path.replace("results_Diffusion", "results_AE")
        .replace("_dit", "")
        .replace("_unet", ""),
        "results.csv",
    )
    print(path_enc_results)
    print(os.path.exists(path_enc_results))
    recon_err = pd.read_csv(path_enc_results)
    recon_err_df = recon_err[["file", "chamfer_distance"]].rename(
        columns={"file": "file_path", "chamfer_distance": "chamfer_distance_recon"}
    )

    # load reconstruction error
    with open(
        "unifi3d/data/shapenet_analysis/shapenet_info_test.json",
        "r",
    ) as f:
        sdf_err = json.load(f)
    sdf_err_df = pd.DataFrame(sdf_err)[["file_path", "chamfer_distance_r64"]]
    sdf_err_df["chamfer_distance_r64"] *= 2

    # merge all of them:
    gen_and_recon = gen_err_df.merge(
        recon_err_df, left_on="file_path", right_on="file_path", how="left"
    )
    gen_and_recon_and_sdf = gen_and_recon.merge(
        sdf_err_df, left_on="file_path", right_on="file_path", how="left"
    )

    # add percentage columns:
    gen_and_recon_and_sdf["percentage_recon"] = (
        gen_and_recon_and_sdf["chamfer_distance_recon"]
        / gen_and_recon_and_sdf["chamfer_distance_gen"]
    ) * 100
    gen_and_recon_and_sdf["percentage_sdf"] = (
        gen_and_recon_and_sdf["chamfer_distance_r64"]
        / gen_and_recon_and_sdf["chamfer_distance_gen"]
    ) * 100

    pearson_gen_compress = pearsonr(
        gen_and_recon_and_sdf["chamfer_distance_gen"].values,
        gen_and_recon_and_sdf["chamfer_distance_recon"].values,
    )
    pearson_gen_recon = pearsonr(
        gen_and_recon_and_sdf["chamfer_distance_gen"].values,
        gen_and_recon_and_sdf["chamfer_distance_r64"].values,
    )
    pearson_compress_recon = pearsonr(
        gen_and_recon_and_sdf["chamfer_distance_recon"].values,
        gen_and_recon_and_sdf["chamfer_distance_r64"].values,
    )
    pearson_dict = {
        "pearson_gen_compress_r": pearson_gen_compress[0],
        "pearson_gen_recon_r": pearson_gen_recon[0],
        "pearson_compress_recon_r": pearson_compress_recon[0],
        "pearson_gen_compress_p": pearson_gen_compress[1],
        "pearson_gen_recon_p": pearson_gen_recon[1],
        "pearson_compress_recon_p": pearson_compress_recon[1],
    }
    return gen_and_recon_and_sdf, pearson_dict


def evaluate_unconditional_generation(
    cfg, path_generated, info_to_save={}, do_relate_to_recon=False
):
    """Compute MMD, Cov and NNA"""

    # use sample diffusion cfg
    print("Using reference dataset:", cfg.reference_dataset)
    print("Using generated dataset:", path_generated)

    # load generated and reference dataset
    generated_meshes = load_generated_dataset(path_generated)
    ref_files, reference_dataset = load_reference_dataset(
        cfg.reference_dataset, num_samples=len(generated_meshes), return_files=True
    )
    print(
        "Loaded generated and referene dataset",
        len(generated_meshes),
        len(reference_dataset),
    )

    # compute metrics
    metrics_comp = MeshDistributionMetrics(num_points=cfg.num_points_metrics)
    metrics = metrics_comp(generated=generated_meshes, reference=reference_dataset)

    # Relate to reconstruction error
    if do_relate_to_recon and "MMD_vector" in metrics.keys():
        gen_err = {f: m for f, m in zip(ref_files, metrics["MMD_vector"])}
        gen_and_recon_and_sdf, pearson_dict = relate_to_recon(gen_err, cfg.output_path)
        metrics.update(pearson_dict)
        # save csv file with generation vs reconstruction error
        gen_and_recon_and_sdf.to_csv(
            os.path.join(path_generated, "..", "gen_vs_recon.csv")
        )

    # add additional information (usually memory info)
    metrics.update(info_to_save)

    # store metrics to file
    out_path_metrics = os.path.join(path_generated, "..", "metrics_sample.json")
    with open(out_path_metrics, "w", encoding="latin-1") as f:
        json.dump(metrics, f)
    print("Metrics are saved to", out_path_metrics)
