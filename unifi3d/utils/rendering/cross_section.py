import matplotlib.pyplot as plt
import os
import numpy as np


def plot_cross_sections(occ_volume, path="", ax=None):
    assert isinstance(occ_volume, np.ndarray)
    assert occ_volume.ndim == 3, "Cube must be 3D"
    layer = occ_volume.shape[0] // 2
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].imshow(occ_volume[:, :, layer], cmap="PiYG")
    ax[0].set_axis_off()
    ax[0].figure.colorbar(ax[0].images[0], ax=ax[0])

    ax[1].imshow(occ_volume[:, layer, :], cmap="PiYG")
    ax[1].set_axis_off()
    ax[1].figure.colorbar(ax[1].images[0], ax=ax[1])

    ax[2].imshow(occ_volume[layer, :, :], cmap="PiYG")
    ax[2].set_axis_off()
    ax[2].figure.colorbar(ax[2].images[0], ax=ax[2])
    plt.savefig(f"{path}/occ_hat_cross_section.png")
    return ax
