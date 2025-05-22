"""
Statistical visualizations for aftershock prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from config import FIGURE_SIZE, FONT_SIZE


def plot_feature_importance(
    feature_importance,
    coordinate,
    approach="multi_station",
    feature_type="all",
    top_n=20,
    output_file=None,
):
    """
    Plot feature importance for a specific coordinate.

    Args:
        feature_importance: DataFrame with feature and importance columns
        coordinate: Coordinate name (e.g., "relative_x")
        approach: Analysis approach used (for filename)
        feature_type: Feature type used (for filename)
        top_n: Number of top features to show
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

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    top_features = feature_importance.head(top_n)

    sns.barplot(
        x="importance", y="feature", data=top_features, palette="viridis", ax=ax
    )

    ax.set_title(f"Top {top_n} Features for {coordinate} Prediction")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if output_file is None:
        output_file = f"feature_importance_{coordinate}_{approach}_{feature_type}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return fig


def plot_predicted_vs_actual(
    y_true, y_pred, coordinate, approach="multi_station", output_file=None
):
    """
    Create scatter plot of predicted vs. actual values for a single coordinate.

    Args:
        y_true: Series or array of true values
        y_pred: Series or array of predicted values
        coordinate: Coordinate name (e.g., "relative_x")
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

    fig, ax = plt.subplots(figsize=(10, 10))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    range_val = max_val - min_val
    buffer = range_val * 0.05

    ax.plot(
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        "k--",
        alpha=0.7,
        label="Perfect Prediction",
    )

    scatter = ax.scatter(y_true, y_pred, alpha=0.7, c="steelblue", edgecolor="white")
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    textstr = f"RMSE: {rmse:.2f} km\n$R^2$: {r2:.4f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props
    )

    ax.set_xlabel(f"True {coordinate} (km)")
    ax.set_ylabel(f"Predicted {coordinate} (km)")
    ax.set_title(f"Predicted vs. Actual {coordinate}")
    ax.grid(alpha=0.3)
    ax.axis("equal")

    ax.set_xlim(min_val - buffer, max_val + buffer)
    ax.set_ylim(min_val - buffer, max_val + buffer)

    plt.tight_layout()
    if output_file is None:
        output_file = f"predicted_vs_actual_{coordinate}_{approach}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return fig


def plot_error_components(errors, approach="multi_station", output_file=None):
    """
    Create boxplot of error components.

    Args:
        errors: Dictionary with 'lat', 'lon', and 'depth' keys
        approach: Analysis approach used (for filename)
        output_file: Optional output filename
    """
    # Apply matplotlib settings
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

    # Check if we have the required components
    required_keys = ["lat", "lon", "depth"]
    if not all(key in errors for key in required_keys):
        raise ValueError(f"errors dictionary must contain keys: {required_keys}")

    # Prepare data for boxplot
    error_data = pd.DataFrame(
        {
            "Latitude Error": errors["lat"],
            "Longitude Error": errors["lon"],
            "Depth Error": errors["depth"],
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create boxplot
    boxplot = sns.boxplot(
        data=error_data, orient="h", palette="viridis", ax=ax, showfliers=True
    )

    # Add swarmplot for individual points
    swarmplot = sns.swarmplot(
        data=error_data, orient="h", color="black", size=4, alpha=0.6, ax=ax
    )

    # Customize plot
    ax.set_title("Distribution of Error Components")
    ax.set_xlabel("Error (km)")
    ax.set_ylabel("Component")
    ax.grid(axis="x", alpha=0.3)

    # Add means as text
    for i, col in enumerate(error_data.columns):
        mean_val = error_data[col].mean()
        ax.text(
            mean_val,
            i,
            f" Mean: {mean_val:.2f} km",
            va="center",
            fontsize=FONT_SIZE - 2,
            bbox=dict(facecolor="white", alpha=0.7, pad=0.1),
        )

    # Tight layout
    plt.tight_layout()

    # Save figure
    if output_file is None:
        output_file = f"error_components_{approach}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return fig


def plot_shap_summary(shap_values, X_test, max_display=20, output_file=None):
    """
    Create SHAP summary plot (beeswarm plot).

    Args:
        shap_values: SHAP values object
        X_test: Test feature DataFrame
        max_display: Number of features to display
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

    shap.summary_plot(shap_values.values, X_test, max_display=max_display, show=False)

    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE)
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Feature value", fontsize=FONT_SIZE)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return fig
