import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_reconstruction_bubbles(in_path, out_path):
    recon = pd.read_csv(in_path)
    recon["chamfer_distance_mean"] /= 1000
    recon["chamfer_distance_std"] /= 1000
    recon = recon.groupby(["name_"]).agg(
        {c: "mean" for c in recon.columns if "mean" in c or "std" in c}
    )  # , "Category_"
    recon["time_total_decode"] = (
        recon["time_decode_mean"] + recon["time_reconstruct_mean"]
    )

    var = "time_reconstruct_mean"
    yvar = "chamfer_distance_mean"
    svar = "Latent size"
    recon["size_encoded_mean"] = recon["size_encoded_mean"].astype(int)
    size_dict = {d: np.log(d) * 90 for d in recon["size_encoded_mean"].values}
    plt.figure(figsize=(8, 4))
    # plt.errorbar(recon[var], recon['chamfer_distance_mean'], yerr=recon['chamfer_distance_mean'], markerfacecolor="white", alpha=0.2, fmt='o', ecolor='gray', elinewidth=2, capsize=4)
    table_for_plot = recon.reset_index()
    table_for_plot.rename(
        columns={"name_": "Method", "size_encoded_mean": "Latent size"}, inplace=True
    )
    ax = sns.scatterplot(
        data=table_for_plot,
        x=var,
        y=yvar,
        hue="Method",
        sizes=size_dict,
        size=svar,
        legend="full",
    )

    # add latent size label inside the bubble
    def format_number(x: float) -> str:
        for val, unit in [(1, ""), (1000, "K"), (1000000, "M"), (1000000000, "G")]:
            if x >= val:
                result = str(int(round(x / val))) + unit
        return result

    for i in range(table_for_plot.shape[0]):
        ax.text(
            table_for_plot[var][i],
            table_for_plot[yvar][i],
            format_number(table_for_plot[svar][i]),
            horizontalalignment="center",
            verticalalignment="center",
            size="medium",
            color="white",
            weight="semibold",
        )

    plt.xlabel("Time mesh reconstruction [s]")

    # Remove the latent size from the legend
    h, l = ax.get_legend_handles_labels()
    ax.legend(
        h[0 : l.index(svar)], l[0 : l.index(svar)], bbox_to_anchor=(1, 1), ncol=1
    )  # , borderaxespad=0., fontsize=13)
    # plt.legend(bbox_to_anchor=(1, 1))# ncol=2) # title="Method")

    plt.ylabel("Chamfer distance")
    plt.ylim(0, 0.06)
    # remove top and right border which may look better (?)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    plot_reconstruction_bubbles(
        in_path="outputs/reconstruction_table.csv",
        out_path="outputs/mem_vs_quality.pdf",
    )
