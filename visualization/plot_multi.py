# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from umap import UMAP
from pathlib import Path
from sklearn.manifold import TSNE

font = {"size": 18}
matplotlib.rc("font", **font)


def plot_by_seaborn(ax, x_feats, y_labels, method="t-sne"):
    n_components = 2

    if method.lower() == "t-sne":
        m = TSNE(
            n_components=n_components,
            perplexity=50,
            learning_rate="auto",
            init="pca",
            random_state=222,
        )
    elif method.lower() == "umap":
        m = UMAP(n_components=n_components, n_neighbors=50, init="pca", random_state=222)

    projections = m.fit_transform(x_feats)

    projections_df = pd.DataFrame(
        {
            "Dimension 1": projections[:, 0],
            "Dimension 2": projections[:, 1],
            "label": y_labels,
        }
    )

    sns.scatterplot(
        ax=ax,
        x="Dimension 1",
        y="Dimension 2",
        hue="label",
        palette=sns.color_palette("coolwarm", as_cmap=True),
        data=projections_df,
        legend=False,
        s=9,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xy_lim = (min(x_min, y_min), max(x_max, y_max))
    ax.set_xlim(xy_lim)
    ax.set_ylim(xy_lim)
    ax.set_aspect("equal")


if __name__ == "__main__":
    data_info_list = [
        {"path": "features/supervised_utkface_lb250_s0.npy", "name": "Supervised"},
        {"path": "features/mixmatch_utkface_lb250_s0.npy", "name": "MixMatch"},
        {"path": "features/supervised_fixmatch_utkface_lb250_s0.npy", "name": "RankUp"},
    ]
    method = "t-SNE"  # UMAP

    features = [np.load(d["path"], allow_pickle=True).item() for d in data_info_list]

    n_data = len(data_info_list)
    fig = plt.figure(figsize=(8 * n_data, 6))
    gs = GridSpec(1, n_data + 1, width_ratios=[1] * n_data + [0.05], wspace=0.0)

    axes = [fig.add_subplot(gs[i]) for i in range(n_data)]
    cbar_ax = fig.add_subplot(gs[-1])

    for i, ax in enumerate(axes):
        data_name = data_info_list[i]["name"]
        print(f"Running {method} for {data_name} features ...")
        plot_by_seaborn(ax, features[i]["feat"], features[i]["label"], method=method)
        ax.set_title(data_info_list[i]["name"])

    norm = plt.Normalize(
        min([feat["label"].min() for feat in features]),
        max([feat["label"].max() for feat in features]),
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=sns.color_palette("coolwarm", as_cmap=True), norm=norm),
        cax=cbar_ax,
        orientation="vertical",
        label="Label (Age)",
    )

    # Ensure output directory exists
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save the plot
    save_name = "_".join([d["name"].lower() for d in data_info_list]) + f"_{method.lower()}_2d.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")
