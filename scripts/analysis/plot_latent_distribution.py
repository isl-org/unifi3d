import hydra
import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import umap
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet


def write_to_file(name_number_paris_list, out_path):
    with open(out_path, "w") as file:
        for name, number in name_number_paris_list:
            file.write(f"{name}: {number}\n")


def plot_robust_mahalanobis(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    mcd = MinCovDet()
    mcd.fit(latents)
    robust_mahalanobis_dist = mcd.mahalanobis(latents)
    plot_kde_hist_overlay(
        data=robust_mahalanobis_dist,
        title="Robust Mahalanobis",
        filepath=os.path.join(out_dir, "robust_mahalanobis.png"),
    )
    robust_skewness = stats.skew(robust_mahalanobis_dist)
    robust_kurtosis = stats.kurtosis(robust_mahalanobis_dist)
    print("Robust skewness")
    skewness_and_kurtosis = [
        ("robust_skewness", robust_skewness),
        ("robust_kurtosis", robust_kurtosis),
    ]
    write_to_file(
        skweness_and_kurtosis, os.path.join(out_dir, "robust_skewness_kurtosis.txt")
    )


def plot_kde_hist_overlay(data, title, filepath):

    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(min(data), max(data), 100)
    nomral_dist = stats.norm.pdf(x, mean, std_dev)
    plt.hist(data, bins=50, density=True, alpha=0.6, color="g", label="Histogram")
    sns.kdeplot(data, fill=True, color="blue", label="KDE")
    plt.plot(x, nomral_dist, color="red", label="Normal Distribution")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filepath)
    print(f"Saving to {filepath}")
    plt.clf()


def print_kurtosis_skewness(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    skewnesses = []
    excess_kurtosises = []
    for dim in range(latents.shape[1]):
        skewness = stats.skew(latents[:, dim])
        excess_kurtosis = stats.kurtosis(latents[:, dim])
        skewnesses.append(skewness)
        excess_kurtosises.append(excess_kurtosis)
        # print(
        #     f"Dimension {dim}: Skewness = {skewness}, Excess Kurtosis = {excess_kurtosis}"
        # )
    plot_kde_hist_overlay(
        data=skewnesses,
        title="Skewness Distribution",
        filepath=os.path.join(out_dir, "skewness.png"),
    )

    plot_kde_hist_overlay(
        data=excess_kurtosises,
        title="Excess Kurtosis Distribution",
        filepath=os.path.join(out_dir, "excess_kurtosis.png"),
    )


def plot_mahalanobis_dist(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    cov_matrix = np.cov(latents.T)
    mean_vector = np.mean(latents, axis=0)
    mahal_distances = [
        mahalanobis(x, mean_vector, np.linalg.inv(cov_matrix)) for x in latents
    ]

    "Mahalanobis Distance Distribution"
    plot_kde_hist_overlay(
        data=mahal_distances,
        title="Mahalanobis Distance Distribution",
        filepath=os.path.join(out_dir, "mahala_distribution.png"),
    )
    skewness = stats.skew(mahal_distances)
    kurtosis = stats.kurtosis(mahal_distances)
    skewness_and_kurtosis = [
        ("skewness", skewness),
        ("kurtosis", kurtosis),
    ]
    write_to_file(skewness_and_kurtosis, os.path.join(out_dir, "skewness_kurtosis.txt"))


def plot_hist(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    os.makedirs(out_dir, exist_ok=True)
    for dim in range(min(4, latents.shape[0])):
        path = os.path.join(out_dir, f"latent_dimension_{dim}_hist.png")
        if not os.path.exists(path):
            plt.hist(latents[:, dim], bins=50, alpha=0.6, label=f"Dim {dim}")
            plt.xlabel(f"Latent Dimension {dim}")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of Latent Dimension {dim}")
            plt.savefig(path)
            print(f"Saving to latent_dimension_{dim}_hist")
            plt.clf()


def plot_kde(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    os.makedirs(out_dir, exist_ok=True)
    for dim in range(min(4, latents.shape[0])):
        sns.kdeplot(latents[:, dim], fill=True)
        plt.title(f"KDE of Latent Dimension {dim}")
        plt.savefig(os.path.join(out_dir, f"latent_dimension_{dim}_kde.png"))
        print(f"Saving to latent_dimension_{dim}_kde")
        plt.clf()


def plot_tsne(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latents)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    plt.title("t-SNE of latent space")
    plt.savefig(os.path.join(out_dir, "latent_2d_tsne.png"))
    print("Saving to latent_2d_tsne")
    plt.clf()
    plot_kde_hist_overlay(
        data=latent_2d[:, 0],
        title="tsne latent 2d_0 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_tsne_0_distribution.png"),
    )

    plot_kde_hist_overlay(
        data=latent_2d[:, 1],
        title="tsne latent 2d_1 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_tsne_1_distribution.png"),
    )


def plot_umap(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    reducer = umap.UMAP()
    latent_2d = reducer.fit_transform(latents)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    plt.title("UMAP of latent space")
    plt.savefig(os.path.join(out_dir, "latent_2d_umap.png"))
    print("Saving to latent_2d_umap")
    plt.clf()
    plot_kde_hist_overlay(
        data=latent_2d[:, 0],
        title="umap latent 2d_0 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_umap_0_distribution.png"),
    )

    plot_kde_hist_overlay(
        data=latent_2d[:, 1],
        title="umap latent 2d_1 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_umap_1_distribution.png"),
    )


def plot_pca(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latents)
    normal_samples = np.random.randn(num_items, 2)
    plt.scatter(
        latent_pca[:, 0],
        latent_pca[:, 1],
        alpha=0.4,
        label="Latent Space (PCA)",
        color="b",
    )
    plt.scatter(
        normal_samples[:, 0],
        normal_samples[:, 1],
        label="Standard Normal",
        alpha=0.4,
        color="r",
    )
    plt.legend()
    plt.title("PCA of latent space vs Standard Normal")
    plt.savefig(os.path.join(out_dir, "latent_2d_pca.png"))
    print("Saving to latent_2d_pca")
    plt.clf()
    plot_kde_hist_overlay(
        data=latent_pca[:, 0],
        title="pca latent 2d_0 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_pca_0_distribution.png"),
    )

    plot_kde_hist_overlay(
        data=latent_pca[:, 1],
        title="pca latent 2d_1 Distribution",
        filepath=os.path.join(out_dir, "latent_2d_pca_1_distribution.png"),
    )


def plot_pca_3d(latents, out_dir):
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    pca = PCA(n_components=3)
    latent_pca = pca.fit_transform(latents)
    normal_samples = np.random.randn(num_items, 3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if latent_pca.size == 0:
        raise ValueError(
            "Latent 3D data is empty. Ensure the latent space has valid data."
        )
    ax.scatter(
        latent_pca[:, 0],
        latent_pca[:, 1],
        latent_pca[:, 2],
        alpha=0.3,
        label="Latent Space (PCA)",
        color="b",
    )
    ax.scatter(
        normal_samples[:, 0],
        normal_samples[:, 1],
        normal_samples[:, 2],
        label="Standard Normal",
        alpha=0.3,
        color="r",
    )
    ax.set_title("PCA of latent space vs Standard Normal")
    ax.legend()
    plt.savefig(
        os.path.join(out_dir, "latent_3d_pca.png"), dpi=300, bbox_inches="tight"
    )
    print("Saving to latent_3d_pca")
    plt.clf()
    plot_kde_hist_overlay(
        data=latent_pca[:, 0],
        title="pca latent 3d_0 Distribution",
        filepath=os.path.join(out_dir, "latent_3d_pca_0_distribution.png"),
    )

    plot_kde_hist_overlay(
        data=latent_pca[:, 1],
        title="pca latent 3d_1 Distribution",
        filepath=os.path.join(out_dir, "latent_3d_pca_1_distribution.png"),
    )

    plot_kde_hist_overlay(
        data=latent_pca[:, 2],
        title="pca latent 3d_2 Distribution",
        filepath=os.path.join(out_dir, "latent_3d_pca_2_distribution.png"),
    )


def kde_pdf_estimation(data, grid):
    kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(data)
    log_pdf = kde.score_samples(grid)
    pdf = np.exp(log_pdf)
    return pdf


def kl_divergence_kde(latents, epsilon=1e-10):
    """Use Kernel Density Estimation to approximate the probability desnity distribution."""
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    num_features = latents.shape[-1]
    # Fit KDE to the empirical latent distribution
    kde_kl_div = 0
    for i in range(num_features):
        latent_dim = latents[:, i].reshape(-1, 1)
        grid = np.linspace(-5, 5, 1000).reshape(-1, 1)
        kde_pdf = kde_pdf_estimation(latent_dim, grid)
        normal_pdf = stats.norm.pdf(grid)
        kde_pdf = np.clip(kde_pdf, epsilon, None)
        normal_pdf = np.clip(normal_pdf, epsilon, None)

        kl_dim = np.sum(kde_pdf * (np.log(kde_pdf) - np.log(normal_pdf))) * (
            grid[1] - grid[0]
        )
        kde_kl_div += kl_dim
    return kde_kl_div


def kl_divergence_sample(latents):
    """Directly estimate kl divergence between the distributions."""
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    empirical_mean = np.mean(latents, axis=0)
    empirical_var = np.var(latents, axis=0)
    mean_diff = np.mean(np.abs(empirical_mean))
    var_diff = np.mean(np.abs(empirical_var))
    kl_div = 0.5 * np.sum(
        np.log(empirical_var) - 1 + (empirical_var + empirical_mean**2) / empirical_var
    )
    return mean_diff, var_diff, kl_div


def gaussian_kernel(x, y, sigma=1.0):
    diff = x[:, None] - y[None, :]
    return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))


def mmd(latents):
    """Compares two sets of samples using a kernel function."""
    num_items = latents.shape[0]
    latents = latents.reshape(num_items, -1)
    num_features = latents.shape[-1]

    sigma = 1.0  # bandwidth for the gaussian kernel
    standard_normal_samples = np.random.randn(num_items, num_features)
    k_xx = gaussian_kernel(latents, latents, sigma)
    k_yy = gaussian_kernel(standard_normal_samples, standard_normal_samples, sigma)
    k_xy = gaussian_kernel(latents, standard_normal_samples, sigma)
    mmd_value = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return mmd_value


def empirical_kl_divergence(latents, out_dir):
    kde_kl_div = kl_divergence_kde(latents)
    mean_diff, var_diff, kl_div = kl_divergence_sample(latents)
    mmd_value = mmd(latents)
    name_number_paris_list = [
        ("kl_div", kl_div),
        ("mean_diff", mean_diff),
        ("var_diff", var_diff),
        ("kde_kl_div", kde_kl_div),
        ("mmd_value", mmd_value),
    ]
    out_path = os.path.join(out_dir, "kl_div.txt")
    write_to_file(name_number_paris_list, out_path)


def collect_latents_and_plot(autoencoder, out_dir, data_loader_iter, data_mode):
    out_dir = os.path.join(out_dir, data_mode)
    os.makedirs(out_dir, exist_ok=True)
    np_array_path = os.path.join(out_dir, f"{data_mode}.npy")
    if os.path.exists(np_array_path):
        latents = np.load(np_array_path)
    else:
        latents_list = []
        for batch_idx in tqdm(range(len(data_loader_iter))):
            # 1) create batch: load mesh and transform into the representation
            batch = next(data_loader_iter)
            # 2) encode batch: transform representation into latent space
            with torch.no_grad():
                encoded = autoencoder.encode_wrapper(batch)
            latents_list.append(encoded.cpu().numpy())
        latents = np.concatenate(latents_list, axis=0)
        np.save(np_array_path, latents)

    # plot_mahalanobis_dist(latents=latents, out_dir=out_dir)
    # plot_robust_mahalanobis(latents=latents, out_dir=out_dir)
    # print_kurtosis_skewness(latents=latents, out_dir=out_dir)
    # plot_kde(latents=latents, out_dir=out_dir)
    # plot_tsne(latents=latents, out_dir=out_dir)
    # plot_umap(latents=latents, out_dir=out_dir)
    plot_pca(latents=latents, out_dir=out_dir)
    plot_pca_3d(latents=latents, out_dir=out_dir)
    # empirical_kl_divergence(latents=latents, out_dir=out_dir)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="benchmark")
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError(
            "You must provide a configuration file with benchmark=<config_name>"
        )
    print(cfg)

    if cfg.net_encode:
        print(f"Found net_encode, instantiating model <{cfg.model._target_}>")
        autoencoder = hydra.utils.instantiate(cfg.net_encode).cuda()
        autoencoder.eval()
        if cfg.ckpt_path:
            autoencoder.load_checkpoint(cfg.ckpt_path)

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    val_loader = datamodule.val_dataloader()

    # create iterator
    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)
    val_loader_iter = iter(val_loader)

    out_dir = "./"

    os.makedirs(out_dir, exist_ok=True)
    assert cfg.net_encode

    collect_latents_and_plot(
        autoencoder=autoencoder,
        out_dir=out_dir,
        data_loader_iter=train_loader_iter,
        data_mode="train",
    )

    collect_latents_and_plot(
        autoencoder=autoencoder,
        out_dir=out_dir,
        data_loader_iter=test_loader_iter,
        data_mode="test",
    )

    collect_latents_and_plot(
        autoencoder=autoencoder,
        out_dir=out_dir,
        data_loader_iter=val_loader_iter,
        data_mode="val",
    )


if __name__ == "__main__":
    main()
