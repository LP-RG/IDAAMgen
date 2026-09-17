import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter

# ===============================
# STANDARD PLOT STYLE
# ===============================

PLOT_STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 1080,
    "savefig.dpi": 1080,

    "axes.titlesize": 24,
    "axes.labelsize": 24,

    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    "legend.fontsize": 14,

    "lines.linewidth": 2,

    "axes.grid": False,
}

def setup_plot_style():
    plt.rcParams.update(PLOT_STYLE)

CMAP_NAME = "viridis"

def _sanitize(arr):
    arr = np.asarray(arr, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _cmap():
    cmap = plt.get_cmap(CMAP_NAME).copy()
    low = cmap(0.0)
    cmap.set_under(low)
    cmap.set_bad(low)
    return cmap

def _compute_ticks(bitwidth):
    max_val = 2 ** bitwidth
    ticks = np.linspace(0, max_val, num=6, dtype=int).tolist()
    return ticks

def _plot_matrix(mat, xlabel, ylabel, out_path, xticks, yticks, title=None, total_sum=None):
    mat = _sanitize(mat)
    cmap = _cmap()

    positive_mask = mat > 0
    if not np.any(positive_mask):
        mat_safe = np.ones_like(mat) * 1e-10
    else:
        min_pos = np.min(mat[positive_mask])
        mat_safe = np.where(mat <= 0, min_pos, mat)

    vmin, vmax = np.min(mat_safe), np.max(mat_safe)
    norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)

    fig, ax = plt.subplots()

    im = ax.imshow(
        mat_safe,
        cmap=cmap,
        origin='upper',
        norm=norm,
        interpolation='nearest',
        aspect='equal'
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, pad=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)

    ax.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=14)

    locator = LogLocator(numticks=6)
    cbar.locator = locator
    cbar.update_ticks()

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0e}"))
    cbar.update_ticks()

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def heat_maps_plotting(folder_path_npy, folder_path_png, bitwidth):
    ticks = _compute_ticks(bitwidth)
    files = glob.glob(os.path.join(folder_path_npy, "*.npy"))
    
    aggregate_mat = None

    for file_path in files:
        h = np.load(file_path)
        h = _sanitize(h)

        mat = np.sum(h, axis=0) if h.ndim == 3 else h
        total_sum = np.sum(mat)

        if aggregate_mat is None:
            aggregate_mat = np.copy(mat)
        else:
            aggregate_mat += mat

        out_path = re.sub(
            r"\.npy$", ".png",
            re.sub(re.escape(folder_path_npy), folder_path_png, file_path)
        )

        _plot_matrix(
            mat,
            xlabel="Weights",
            ylabel="Activations",
            out_path=out_path,
            xticks=ticks,
            yticks=ticks,
            total_sum=total_sum
        )

    # Plot della matrice aggregata
    if aggregate_mat is not None:
        agg_out_path = os.path.join(folder_path_png, "aggregate_distribution.png")
        _plot_matrix(
            aggregate_mat,
            xlabel="Weights",
            ylabel="Activations",
            out_path=agg_out_path,
            xticks=ticks,
            yticks=ticks,
            title="Aggregate Distribution",
            total_sum=np.sum(aggregate_mat)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmaps from NPY matrices.")
    parser.add_argument(
        "--folder_path_npy",
        type=str,
        default="./heat_maps/npy_matrix/7bit_resnet20_cifar10",
        help="Input directory containing .npy files"
    )
    parser.add_argument(
        "--folder_path_png",
        type=str,
        default="./heat_maps/plot/7bit_resnet20_cifar10",
        help="Output directory for generated plots"
    )
    parser.add_argument(
        "--bitwidth",
        type=int,
        default=7,
        help="Bitwidth used for quantization (e.g., 7 or 8)"
    )
    args = parser.parse_args()

    setup_plot_style()
    heat_maps_plotting(args.folder_path_npy, args.folder_path_png, args.bitwidth)