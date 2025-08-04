import numpy as np
import pandas as pd
import os
import json
from omegaconf import DictConfig
import rootutils
import hydra

from make_reconstruction_table import load_final_results, rename_models

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="make_tables")
def make_gen_table(
    cfg: DictConfig,
):
    group_cols = ["Category", "name", "Last modified"]
    rm_cols = [
        "Unnamed: 0",
        "batch_id",
        "file",
        "sample_id",
    ] + group_cols
    print(cfg)
    if "result_dir" not in cfg:
        result_dir = "logs/final_results_Diffusion"
        # get all models we ran by looking for all the subfolders of "Chair"
        sub_dirs = os.listdir(os.path.join(result_dir, "Chair"))
        names = [rename_models(n) for n in sub_dirs]
    else:
        result_dir = cfg.result_dir
        sub_dirs = cfg.sub_dirs
        names = cfg.names

    all_res, all_metrics = load_final_results(
        result_dir, sub_dirs, names, is_generation=True, cats=["Chair"]
    )

    # combine and average over samples
    res_grouped = (
        all_res.groupby(group_cols)
        .agg({col: "mean" for col in all_res.columns if col not in rm_cols})
        .reset_index()
    )
    # merge with metrics
    res_grouped = res_grouped.merge(all_metrics, on=group_cols)

    final_names = {c: c.replace("_", " ") for c in res_grouped.columns}
    final_names["name"] = "Method"
    final_names["time gen"] = "Time (s)"
    res_grouped.rename(columns=final_names, inplace=True)

    # save to csv file
    res_grouped.to_csv(os.path.join("outputs", "generation_table.csv"), index=False)

    # columns to include in final latex table
    res_grouped = res_grouped[["Category", "Method", "COV", "MMD", "1-NNA"]]

    print(res_grouped.sort_values("Method").head(30).to_markdown())
    print("====== LATEX CODE ======")
    print(res_grouped.to_latex(index=False, float_format="%.2f"))


if __name__ == "__main__":
    make_gen_table()
