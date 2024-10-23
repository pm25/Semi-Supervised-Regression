# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from umap import UMAP
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, default="features/supervised_fixmatch_utkface_lb250_s0.npy")
parser.add_argument("--output_dir", type=str, default="visualization/figures")
parser.add_argument("--methods", type=str, nargs="+", default=["tsne", "umap"], choices=["tsne", "umap"])
parser.add_argument("--output_dim", type=int, default=2, choices=[2, 3], help="Dimension of output (2 or 3).")
parser.add_argument("--font_size", type=int, default=12, help="Font size for plots. Must be a positive integer.")
args = parser.parse_args()

# Set global font size for matplotlib
font = {"size": args.font_size}
matplotlib.rc("font", **font)

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
METHOD_NAME = {"tsne": "t-SNE", "umap": "UMAP"}


def plot_by_seaborn(x_feats, y_labels, method="t-sne", save_name="default", legend=False):
    n_components = 2

    # Choose between t-SNE and UMAP
    if method.lower() == "t-sne":
        m = TSNE(n_components=n_components, perplexity=50, learning_rate="auto", init="pca", random_state=222)
    elif method.lower() == "umap":
        m = UMAP(n_components=n_components, n_neighbors=50, init="pca", random_state=222, n_jobs=1)

    # Apply dimensionality reduction
    projections = m.fit_transform(x_feats)
    projections_df = pd.DataFrame({"Dimension 1": projections[:, 0], "Dimension 2": projections[:, 1], "label": y_labels})

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        x="Dimension 1",
        y="Dimension 2",
        hue="label",
        palette=sns.color_palette("coolwarm", as_cmap=True),
        data=projections_df,
        legend=legend,
        s=9,
    )

    # Ensure equal scaling for x and y axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xy_lim = (min(x_min, y_min), max(x_max, y_max))
    ax.set_xlim(xy_lim)
    ax.set_ylim(xy_lim)
    ax.set_aspect("equal")
    ax.set_title(f"{save_name} ({method})")

    # Normalize the color bar based on the labels
    norm = plt.Normalize(y_labels.min(), y_labels.max())
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.get_cmap("coolwarm"), norm=norm),
        orientation="vertical",
        label="Label",
        ax=plt.gca(),
    )

    # Save the plot
    save_path = OUTPUT_DIR / f"{save_name.lower()}_{method.lower()}_{n_components}d.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_by_plotly(x_feats, y_labels, method="t-SNE", use_3d=False, save_name="default"):
    n_components = 3 if use_3d else 2

    # Choose between t-SNE and UMAP
    if method.lower() == "t-sne":
        m = TSNE(n_components=n_components, perplexity=50, learning_rate="auto", init="pca", random_state=222)
    elif method.lower() == "umap":
        m = UMAP(n_components=n_components, n_neighbors=50, init="pca", random_state=222, n_jobs=1)

    # Apply dimensionality reduction
    projections = m.fit_transform(x_feats)

    if use_3d:
        fig = px.scatter_3d(projections, x=0, y=1, z=2, color=y_labels, labels={"color": "label"}, color_continuous_scale="RdBu_r")
    else:
        fig = px.scatter(projections, x=0, y=1, color=y_labels, labels={"color": "label"}, color_continuous_scale="RdBu_r")

    fig.update_traces(marker_size=5)
    fig.layout.yaxis.scaleanchor = "x"

    title = f"{save_name} ({method})"
    fig.update_layout(title=title, autosize=False, width=1000, height=1000, scene_aspectmode="cube", scene_aspectratio=dict(x=1, y=1, z=1))

    save_path = OUTPUT_DIR / f"{save_name.lower()}_{method.lower()}_{n_components}d.html"
    fig.write_html(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Load data
    data = np.load(args.load_path, allow_pickle=True).item()
    assert "feat" in data and "label" in data, "Data must contain 'feat' and 'label' keys."
    data_name = Path(args.load_path).stem

    # Run t-SNE or UMAP and save figures
    for method_key in args.methods:
        method = METHOD_NAME.get(method_key, None)
        if not method:
            print(f"Warning: Method '{method_key}' is not recognized. Skipping.")
            continue

        print("=" * 25)
        print(f"Running {method} for {data_name} ...")
        if args.output_dim == 2:
            plot_by_seaborn(data["feat"], data["label"], method=method, save_name=data_name)
        elif args.output_dim == 3:
            plot_by_plotly(data["feat"], data["label"], method=method, use_3d=True, save_name=data_name)
