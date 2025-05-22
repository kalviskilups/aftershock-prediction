"""
Geographic visualizations for aftershock prediction results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D

from config import FIGURE_SIZE, FONT_SIZE


def plot_geographic_comparison(
    true_coords, pred_coords, mainshock, approach="multi_station", output_file=None
):
    """
    Create a geographic map comparing true and predicted aftershock locations.

    Args:
        true_coords: DataFrame with true lat, lon, depth coordinates
        pred_coords: DataFrame with predicted lat, lon, depth coordinates
        mainshock: Dictionary with mainshock information
        approach: Analysis approach used (for filename)
        output_file: Optional output filename
    """

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = plt.axes(projection=ccrs.Mercator())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)
    buffer = 0.3
    min_lon = min(true_coords["lon"].min(), pred_coords["lon"].min()) - buffer
    max_lon = max(true_coords["lon"].max(), pred_coords["lon"].max()) + buffer
    min_lat = min(true_coords["lat"].min(), pred_coords["lat"].min()) - buffer
    max_lat = max(true_coords["lat"].max(), pred_coords["lat"].max()) + buffer

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    true_scatter = ax.scatter(
        true_coords["lon"],
        true_coords["lat"],
        c="blue",
        s=40,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
        label="True Epicenter",
        edgecolor="white",
        linewidth=0.5,
    )

    pred_scatter = ax.scatter(
        pred_coords["lon"],
        pred_coords["lat"],
        c="red",
        s=40,
        alpha=0.7,
        marker="x",
        transform=ccrs.PlateCarree(),
        label="Predicted Epicenter",
        linewidth=1.5,
    )

    for i in range(len(true_coords)):
        ax.plot(
            [true_coords["lon"].iloc[i], pred_coords["lon"].iloc[i]],
            [true_coords["lat"].iloc[i], pred_coords["lat"].iloc[i]],
            "k-",
            alpha=0.2,
            transform=ccrs.PlateCarree(),
            linewidth=0.8,
        )

    mainshock_scatter = ax.scatter(
        mainshock["longitude"],
        mainshock["latitude"],
        c="yellow",
        s=200,
        marker="*",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="Mainshock",
        linewidth=1.5,
    )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": FONT_SIZE - 2}
    gl.ylabel_style = {"size": FONT_SIZE - 2}

    plt.legend(loc="lower left", framealpha=0.9)
    plt.title("Aftershock Location Prediction Results", pad=20)

    if output_file is None:
        output_file = f"prediction_results_geographic_{approach}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return fig


def plot_3d_aftershocks(
    true_coords,
    pred_coords,
    mainshock,
    approach="multi_station",
    feature_type="all",
    output_file=None,
):
    """
    Create 3D visualization of aftershock locations.

    Args:
        true_coords: DataFrame with true lat, lon, depth coordinates
        pred_coords: DataFrame with predicted lat, lon, depth coordinates
        mainshock: Dictionary with mainshock information
        approach: Analysis approach used (for filename)
        feature_type: Feature type used (for filename)
        output_file: Optional output filename
    """

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )

    fig = plt.figure(figsize=(17, 16))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        true_coords["lon"],
        true_coords["lat"],
        true_coords["depth"],
        c="blue",
        s=40,
        alpha=0.6,
        label="True Epicenter",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.scatter(
        pred_coords["lon"],
        pred_coords["lat"],
        pred_coords["depth"],
        c="red",
        s=40,
        alpha=0.6,
        marker="x",
        label="Predicted Epicenter",
        linewidth=1.5,
    )

    for i in range(len(true_coords)):
        ax.plot(
            [true_coords["lon"].iloc[i], pred_coords["lon"].iloc[i]],
            [true_coords["lat"].iloc[i], pred_coords["lat"].iloc[i]],
            [true_coords["depth"].iloc[i], pred_coords["depth"].iloc[i]],
            "k-",
            alpha=0.15,
            linewidth=0.8,
        )

    ax.scatter(
        [mainshock["longitude"]],
        [mainshock["latitude"]],
        [mainshock["depth"]],
        c="yellow",
        s=200,
        marker="*",
        label="Mainshock",
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Longitude", labelpad=15)
    ax.set_ylabel("Latitude", labelpad=15)
    ax.set_zlabel("Depth (km)", labelpad=15)

    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=5)
    ax.locator_params(axis="z", nbins=5)

    ax.invert_zaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1), framealpha=0.9)
    plt.title("3D Aftershock Location Prediction", y=0.98)
    ax.view_init(elev=25, azim=50)
    plt.tight_layout(pad=3.0)

    if output_file is None:
        output_file = f"3d_location_visualization_{approach}_{feature_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return fig


def plot_error_histogram(errors, approach="multi_station", output_file=None):
    """
    Create histogram of prediction errors.

    Args:
        errors: Dictionary with error metrics or array of 3D errors
        approach: Analysis approach used (for filename)
        output_file: Optional output filename
    """

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
        }
    )

    if isinstance(errors, dict) and "3d" in errors:
        distance_3d_km = errors["3d"]
    elif isinstance(errors, (list, np.ndarray)):
        distance_3d_km = np.array(errors)
    else:
        raise ValueError(
            "errors must be a dictionary with '3d' key or an array-like of 3D distances"
        )

    fig, ax = plt.subplots(figsize=(12, 8))

    n, bins, patches = ax.hist(
        distance_3d_km,
        bins=20,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        linewidth=1,
    )

    mean_line = ax.axvline(
        distance_3d_km.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {distance_3d_km.mean():.2f} km",
    )

    median_line = ax.axvline(
        np.median(distance_3d_km),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(distance_3d_km):.2f} km",
    )

    ax.set_xlabel("Location Error (km)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of 3D Location Errors")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if output_file is None:
        output_file = f"prediction_error_histogram_{approach}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return fig
