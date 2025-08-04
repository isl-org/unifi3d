import numpy as np
import pandas as pd
import os
import json
import shutil

from omegaconf import DictConfig
import rootutils
import hydra
from datetime import datetime

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

MODE = "standard"  # "complex_shapes"
MIN_RES_LEN = 1  # minimum of samples in the run to load the results


def rename_models(model):
    # Mapping of original segments to their replacements
    rename_map = {
        "doctree": "DualOctree",
        "dualoctree": "DualOctree",
        "voxel": "Voxel",
        "sdf": "SDF",
        "ae": "AE",
        "vae": "VAE",
        "vqvae": "VQVAE",
        "dit": "DiT",
        "unet": "UNet",
        "128": "128",
        "shape2vecset": "Shape2VecSet",
        "triplane": "Triplane",
        "shap": "NeRF",
        "wln": "(norm)",
        # "champion": "best",
    }
    # Split the original name into parts
    parts = model.split("_")
    # Use the map to get the new name
    new_parts = [rename_map[part] for part in parts if part in rename_map]
    # Join the renamed parts back together
    new_name = " ".join(new_parts)

    return new_name


def get_last_modified(path_to_results, path_ae_or_diff="autoencoders"):
    """
    Find out when the results were last modified, and if the ones in the old folder are
    more up to date, take that one
    """
    cat = path_to_results.split(os.sep)[-3]
    rep_sub_dir = path_to_results.split(os.sep)[-2]
    old_path = os.path.join(
        "logs/outputs",
        cat,
        path_ae_or_diff,
        rep_sub_dir,
        "results.csv",
    )
    if os.path.exists(old_path):
        res1 = pd.read_csv(path_to_results)
        res2 = pd.read_csv(old_path)
        if res1.equals(res2):
            path_to_results = old_path
    last_modified_ts = os.path.getmtime(path_to_results)
    # Convert timestamp to date
    date = datetime.fromtimestamp(last_modified_ts)
    last_modified = date.strftime("%Y-%m-%d")
    return path_to_results, last_modified


def load_final_results(
    result_dir,
    sub_dirs,
    names,
    is_generation=False,
    cats=["Chair"],
):
    """
    Iterate through a folder structure with categories and result files and load
    """
    min_res_len = MIN_RES_LEN
    all_res, all_metrics = [], []
    path_part = "diffusions" if is_generation else "autoencoders"
    for cat in cats:
        for rep_sub_dir, name in zip(sub_dirs, names):
            if "old" in rep_sub_dir:
                continue
            path_to_results = os.path.join(result_dir, cat, rep_sub_dir, "results.csv")
            if not os.path.exists(path_to_results):
                print("Warning: path does not exist", path_to_results)
                empty_frame = pd.DataFrame()
                empty_frame["name"] = name
                empty_frame["Category"] = cat
                all_res.append(empty_frame)
                # print(f"Appended empty DF for {name}, {cat}, at path {path_to_results}")
                continue
            last_modified_path, last_modified_date = get_last_modified(
                path_to_results, path_ae_or_diff=path_part
            )
            res = pd.read_csv(last_modified_path)
            if (
                len(res) < min_res_len
                and ("shap_e" not in rep_sub_dir)
                and (rep_sub_dir not in ["triplane_ae_unet", "voxel_vae_dit"])
            ):
                empty_frame = pd.DataFrame()
                empty_frame["name"] = name
                empty_frame["Last modified"] = last_modified_date
                empty_frame["Category"] = cat
                if not is_generation:
                    all_res.append(empty_frame)
                print("Warning: Results table not complete for", name, cat)
                continue
            res["name"] = name
            res["Category"] = cat
            res["Last modified"] = last_modified_date

            # load memory info
            memory_path = os.path.join(result_dir, cat, rep_sub_dir, "memory_info.json")
            if os.path.exists(memory_path):
                with open(memory_path, "r") as f:
                    memory_info = json.load(f)
                res["mem_allocated"] = memory_info.get("Allocated", 0)
                res["mem_reserved"] = memory_info.get("Reserved", 0)
                res["mem_computed"] = memory_info.get("Computed model size", 0)

            print(f"Loaded {name}, {cat} with length {len(res)}")
            all_res.append(res)

            metrics_path = os.path.join(
                result_dir, cat, rep_sub_dir, "metrics_sample.json"
            )
            print(metrics_path)
            if not os.path.exists(metrics_path):
                metrics_path = os.path.join(
                    result_dir, cat, rep_sub_dir, "metrics.json"
                )
            # # code needed for error analysis
            # else:
            #     print("COPY METRICS SAMPLE FOR", cat, rep_sub_dir)
            #     shutil.copy(
            #         metrics_path,
            #         f"outputs/error_analysis/{cat.lower()}_{rep_sub_dir}.json",
            #     )
            #     shutil.copy(
            #         metrics_path.replace("metrics_sample.json", "gen_vs_recon.csv"),
            #         f"outputs/error_analysis/{cat.lower()}_{rep_sub_dir}.csv",
            #     )
            if is_generation and os.path.exists(metrics_path):
                # load uncond gen metrics results
                with open(metrics_path, "r") as f:
                    res_metrics = json.load(f)
                res_metrics["name"] = name
                res_metrics["Category"] = cat
                res_metrics["Last modified"] = last_modified_date
                res_metrics.pop("dataset", None)
                res_metrics.pop("diffusion model", None)
                res_metrics.pop("MMD_vector", None)
                all_metrics.append(res_metrics)

    if is_generation:
        return pd.concat(all_res), pd.DataFrame(all_metrics)
    return pd.concat(all_res)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="make_tables")
def make_tables(
    cfg: DictConfig,
):
    """
    For generating the main table, run this script without any arguments.
    If you have a specific folder / ablation study to evaluate: specify a config in the
    configs/make_tables folder and run
    `python scripts/evaluate/make_reconstruction_table.py make_tables=your_config_name`
    """
    # Constant: which columns to remove
    group_cols = ["Category", "name", "Last modified"]
    rm_cols = [
        "Unnamed: 0",
        "batch_id",
        "file",
        "vertex_defects",
    ] + group_cols

    print(cfg)
    if "result_dir" not in cfg:
        # Use final result dir for general reconstruction table
        if MODE == "complex_shapes":
            result_dir = "logs/results"
            sub_dirs = os.listdir(os.path.join(result_dir, "cube"))
            categories = [
                "cube",
                "sphere",
                "mandelbulb_1",
                "mandelbulb_2",
                "mandelbulb_3",
            ]
        else:
            result_dir = "logs/final_results_AE"
            sub_dirs = os.listdir(os.path.join(result_dir, "Chair"))
            categories = ["Chair", "Airplane", "Car", "GeneralizeChairToAirplane"]
        names = [rename_models(n) for n in sub_dirs]
        all_res = load_final_results(result_dir, sub_dirs, names, cats=categories)
    else:
        result_dir = cfg.result_dir
        sub_dirs = cfg.sub_dirs
        names = cfg.names
        all_res = []
        for rep_sub_dir, name in zip(sub_dirs, names):
            res = pd.read_csv(os.path.join(result_dir, rep_sub_dir, "results.csv"))
            res["name"] = name
            res["Category"] = result_dir.split(os.sep)[-2]
            all_res.append(res)
        all_res = pd.concat(all_res)

    all_res["chamfer_distance"] *= 1000
    all_res["Category"] = all_res["Category"].apply(
        lambda x: "OOD (Chair2Airplane)" if x == "GeneralizeChairToAirplane" else x
    )
    all_res["time_total_decode"] = all_res["time_decode"] + all_res["time_reconstruct"]

    # Aggregate by mean and print
    res_grouped = (
        all_res.groupby(group_cols)
        .agg({col: "mean" for col in all_res.columns if col not in rm_cols})
        .reset_index()
    )
    # print(res_grouped.head(30).to_markdown())
    # # To print the table only with the mean
    # print(res_grouped.to_latex(index=False, float_format=".2f"))

    # Aggregate by mean and std
    res_grouped = (
        all_res.groupby(group_cols)
        .agg({col: ["mean", "std"] for col in all_res.columns if col not in rm_cols})
        .reset_index()
    )
    # res_grouped.sort_values(by=["Category", ("chamfer_distance", "mean")], inplace=True)

    # save as csv
    res_grouped_flat = res_grouped.copy()
    res_grouped_flat.columns = ["_".join(col) for col in res_grouped_flat.columns]
    res_grouped_flat["size"] = res_grouped_flat["name_"].apply(
        lambda x: "large" if "128" in x else "small"
    )
    out_name = (
        "complex_reconstruction_table"
        if MODE == "complex_shapes"
        else "reconstruction_table"
    )
    res_grouped_flat.sort_values(["Category_", "size", "name_"]).to_csv(
        os.path.join("outputs", f"{out_name}.csv"), index=False
    )

    res_grouped.sort_values(by=["Category", "name"], inplace=True)
    res_grouped["chamfer_distance", "mean"] /= 1000

    # Combine mean and std with pm and convert to string dataframe
    string_df = res_grouped[group_cols]
    for col in all_res.columns:
        if col in rm_cols:
            continue
        # no std needed for runtime and size
        if "time" in col:
            string_df[col] = res_grouped[col, "mean"].round(3).astype(str)
        elif "size" in col:
            string_df[col] = res_grouped[col, "mean"].round().astype(int).astype(str)
        else:
            mean_vals = res_grouped[col, "mean"].round(3).astype(str)
            std_vals = res_grouped[col, "std"].round(3).astype(str)
            string_df[col] = mean_vals + " $\pm$ " + std_vals

    string_df.columns = string_df.columns.get_level_values(0)

    # # Combine mean and std with ±
    # res_grouped = res_grouped.apply(combine_mean_std, axis=0)

    # Change here for column names
    final_names = {
        "name": "Method",
        "iou": "IoU",
        "fscore": "F-score (0.05)",
        "fscore_025": "F-score (0.025)",
        "fscore_0125": "F-score (0.0125)",
        "chamfer_distance": "CD",
        "time_convert": "t(transform)",
        "time_encode": "t(encode)",
        "time_decode": "t(decode)",
        "time_reconstruct": "t(mesh)",
        "size_encoded": "Latent size",
        "time_total_decode": "Runtime decode",
        "normal_consistency": "NC",
        "mesh_volume_iou": "IoU",
        "vertex_defects": "VD",
    }

    string_df.rename(columns=final_names, inplace=True)

    # include only a subset of the columns for the final latex table
    string_df = string_df[
        [
            "Category",
            "Method",
            # "Last modified",
            "Runtime decode",
            "IoU",
            "F-score (0.05)",
            "F-score (0.025)",
            "F-score (0.0125)",
            "CD",
            "NC",
        ]
    ]

    tab_recon = string_df[~string_df["Category"].str.contains("OOD")]
    tab_generalize = string_df[string_df["Category"].str.contains("OOD")]

    print(tab_recon.head(30).to_markdown())

    print("====== LATEX CODE (generalization) ======")
    print(tab_generalize.to_latex(index=False, escape=False))

    print("====== LATEX CODE (reconstruction) ======")
    print(tab_recon.to_latex(index=False, escape=False))


if __name__ == "__main__":
    make_tables()
