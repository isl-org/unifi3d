import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def rename_models(model):
    # Mapping of original segments to their replacements
    rename_map = {
        "doctree": "DualOctree",
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
    parts = model.split("_")[:-1]  # remove metric part of name
    # Use the map to get the new name
    new_parts = [rename_map[part] for part in parts if part in rename_map]
    # Join the renamed parts back together
    new_name = "\n".join(new_parts)

    return new_name


mesh_properties = pd.read_csv("outputs/mesh_property_results.csv").dropna()


property_rename_dict = {
    "nr_triangles_binned": "Number of faces (quartile)",
    "nr_vertices_binned": "Number of vertices (quartile)",
    "surface_to_volume_binned": "Surface-to-volume ratio (quartile)",
    "fscore": "F-score",
    "CD": "Chamfer distance",
}
METRIC = "CD"  #  "fscore" or "CD"
PLOTTYPE = "lineplot"
NR_QUANTILES = 10

out_path = "outputs/figures_mesh_property_analysis"
os.makedirs(out_path, exist_ok=True)

for mesh_prop in ["surface_to_volume", "nr_triangles", "nr_vertices"]:

    # bin into quantiles
    new_property_col_name = mesh_prop + "_binned"

    mesh_properties[new_property_col_name] = pd.qcut(
        mesh_properties[mesh_prop],
        NR_QUANTILES,
        labels=np.arange(1, NR_QUANTILES + 1),
    )

    melted = pd.melt(
        mesh_properties,
        id_vars=[new_property_col_name],
        value_vars=[col for col in mesh_properties.columns if METRIC in col],
    )

    melted["Model"] = melted["variable"].apply(rename_models)
    melted.rename({"value": METRIC}, axis=1, inplace=True)
    melted.rename(property_rename_dict, axis=1, inplace=True)

    hue_col = property_rename_dict[new_property_col_name]

    # Figure
    plt.figure(figsize=(10, 5))
    if PLOTTYPE == "boxplot":
        sns.boxplot(data=melted, hue=hue_col, y=property_rename_dict[METRIC], x="Model")
    elif PLOTTYPE == "lineplot":
        sns.lineplot(
            data=melted, x=hue_col, y=property_rename_dict[METRIC], hue="Model"
        )
    else:
        sns.barplot(data=melted, hue=hue_col, y=property_rename_dict[METRIC], x="Model")

    if PLOTTYPE == "lineplot":
        plt.legend(ncol=4, loc="upper center")
    else:
        if METRIC == "fscore":
            plt.ylim(0, 120)
            plt.legend(ncol=4, loc="upper center", title=hue_col)
        else:
            if PLOTTYPE == "boxplot":
                plt.ylim(0, 0.2)
            plt.legend(ncol=4, loc="upper right", title=hue_col)

    plt.savefig(os.path.join(out_path, f"{mesh_prop}_{METRIC}_{PLOTTYPE}.png"))
