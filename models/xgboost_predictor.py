"""
XGBoost implementation for aftershock prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time

from models.base_predictor import BasePredictor
from features.extractor import prepare_multi_station_dataset
from utils.coordinate_utils import cartesian_to_geographic
from config import DEFAULT_XGB_PARAMS, FIGURE_SIZE, FONT_SIZE


class XGBoostPredictor(BasePredictor):
    """
    Class for predicting aftershock locations using XGBoost models
    with best-station and multi-station approaches
    """

    def __init__(
        self,
        data_file=None,
        validation_level="full",
        approach="multi_station",
        feature_type="all",
    ):
        """
        Initialize the predictor

        Args:
            data_file: Path to pickle or HDF5 file with preprocessed data
            validation_level: Level of validation to perform
            approach: Analysis approach to use
                    "best_station" - use only the best station for each event
                    "multi_station" - use all available stations for each event
            feature_type: Type of features to use
                    "all" - use all features
                    "signal" - use only signal-based features (Tier A)
                    "physics" - use physics features (Tier C)
        """
        super().__init__(data_file, validation_level, approach, feature_type)
        self.shap_results = None

    def prepare_dataset(self):
        """
        Prepare dataset for machine learning by extracting features from waveforms
        """
        if self.approach == "multi_station" and self.data_format == "multi_station":
            return prepare_multi_station_dataset(self.aftershocks_df, self.feature_type)
        else:
            raise ValueError(
                f"Approach {self.approach} is not compatible with data format {self.data_format}"
            )

    def train_model(self, X, y, perform_shap=True):
        """
        Train XGBoost models to predict aftershock locations

        Args:
            X: Feature DataFrame
            y: Target DataFrame with coordinates
            perform_shap: Whether to perform SHAP analysis (default: True)

        Returns:
            X_test, y_test: Test data for evaluation
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

        if self.validation_level != "none":
            if "event_id" in y.columns:
                print(
                    "Using GroupKFold with event_id as the group to prevent data leakage..."
                )
                gkf = GroupKFold(n_splits=10)
                groups = y["event_id"]
                train_idx, test_idx = next(gkf.split(X, y, groups))

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                if "event_id" in X_train.columns:
                    X_train = X_train.drop("event_id", axis=1)
                if "event_id" in X_test.columns:
                    X_test = X_test.drop("event_id", axis=1)

                y_train_coord = y_train.drop("event_id", axis=1)
                y_test_coord = y_test.drop("event_id", axis=1)

        numeric_columns = [
            col
            for col in X_train.columns
            if pd.api.types.is_numeric_dtype(X_train[col])
        ]
        non_numeric_columns = [
            col for col in X_train.columns if col not in numeric_columns
        ]

        if non_numeric_columns:
            print(
                f"Removing {len(non_numeric_columns)} non-numeric columns from training data: {non_numeric_columns}"
            )
            X_train_numeric = X_train[numeric_columns]
            X_test_numeric = X_test[numeric_columns]
        else:
            X_train_numeric = X_train
            X_test_numeric = X_test

        xgb_params = DEFAULT_XGB_PARAMS
        print(f"Training XGBoost model with parameters: {xgb_params}")

        base_xgb = XGBRegressor(**xgb_params)

        # Use MultiOutputRegressor to predict all three coordinates
        multi_model = MultiOutputRegressor(base_xgb)
        multi_model.fit(
            X_train_numeric, y_train_coord[["relative_x", "relative_y", "relative_z"]]
        )

        y_pred = multi_model.predict(X_test_numeric)

        multi_ml_errors = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = mean_squared_error(y_test_coord[coord], y_pred[:, i])
            rmse = np.sqrt(mse)
            multi_ml_errors[coord] = rmse

        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            r2 = r2_score(y_test_coord[coord], y_pred[:, i])
            print(f"  {coord} R²: {r2:.4f}")

        # Calculate 3D distance error
        multi_ml_3d_errors = np.sqrt(
            (y_pred[:, 0] - y_test_coord["relative_x"]) ** 2
            + (y_pred[:, 1] - y_test_coord["relative_y"]) ** 2
            + (y_pred[:, 2] - y_test_coord["relative_z"]) ** 2
        )
        multi_ml_errors["3d_distance"] = np.mean(multi_ml_3d_errors)

        print("\nModel Performance (RMSE):")
        for coord in ["relative_x", "relative_y", "relative_z", "3d_distance"]:
            print(f"  {coord}: {multi_ml_errors[coord]:.2f} km")

        self.models = {"multi_output": multi_model, "type": "xgboost"}
        self.scaler = None

        self.feature_importances = {}
        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            xgb_model = multi_model.estimators_[i]

            importance_df = pd.DataFrame(
                {
                    "feature": X_train_numeric.columns,
                    "importance": xgb_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            self.feature_importances[coord] = importance_df

            print(f"\nTop features for {coord} prediction:")
            print(importance_df.head(10))

            plt.figure(figsize=FIGURE_SIZE)
            top_features = importance_df.head(20)
            sns.barplot(x="importance", y="feature", data=top_features)
            plt.title(f"Top 20 Features for {coord} Prediction (XGBoost)")
            plt.tight_layout()
            plt.savefig(
                f"xgboost_feature_importance_{coord}_{self.approach}_{self.feature_type}.png",
                dpi=300,
            )
            plt.close()

        if perform_shap:
            self.shap_results = self.perform_shap_analysis(X_test_numeric, y_test_coord)

        return X_test_numeric, y_test_coord

    def perform_shap_analysis(self, X_test, y_test, max_display=20):
        """
        Perform SHAP analysis on the trained XGBoost models to interpret feature importance
        and feature contributions to individual predictions with unified scales.
        """
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        if self.models is None:
            raise ValueError("Models not trained yet. Call train_model first.")

        print("\n" + "=" * 50)
        print("PERFORMING SHAP ANALYSIS")
        print("=" * 50)

        shap_results = {"values": {}, "plots": {}, "feature_importance": {}}

        coords = ["relative_x", "relative_y", "relative_z"]

        max_summary_shap = 0
        max_bar_importance = 0

        for i, coord in enumerate(coords):
            print(f"\nAnalyzing SHAP values for {coord} prediction...")

            xgb_model = self.models["multi_output"].estimators_[i]
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_test)
            shap_results["values"][coord] = shap_values
            feature_importance = pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance": np.abs(shap_values.values).mean(0),
                }
            ).sort_values("importance", ascending=False)

            shap_results["feature_importance"][coord] = feature_importance

            current_max_shap = np.abs(shap_values.values).max()
            current_max_importance = feature_importance["importance"].max()

            max_summary_shap = max(max_summary_shap, current_max_shap)
            max_bar_importance = max(max_bar_importance, current_max_importance)

        max_summary_shap = max_summary_shap * 1.05
        print(f"Maximum SHAP value for unified scaling: {max_summary_shap:.4f}")

        for i, coord in enumerate(coords):
            print(f"\nCreating plots for {coord} prediction...")

            shap_values = shap_results["values"][coord]

            # 1. Summary plot (beeswarm plot)
            plt.figure(figsize=FIGURE_SIZE)
            shap.summary_plot(
                shap_values.values,
                X_test,
                max_display=max_display,
                show=False,
                plot_size=FIGURE_SIZE,
                plot_type="dot",
                color_bar=False,
            )

            cbar = plt.colorbar()
            cbar.ax.tick_params(direction="out", length=6, width=2, grid_alpha=0.5)
            cbar.set_label("Feature value", fontsize=FONT_SIZE)
            cbar.set_ticks([cbar.vmin, cbar.vmax])
            cbar.set_ticklabels(["Low", "High"])

            ax = plt.gca()
            ax.set_xlim(-max_summary_shap, max_summary_shap)
            ax.set_xlabel("SHAP value (impact on model output)", fontsize=FONT_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

            plt.tight_layout()
            plt.savefig(
                f"shap_summary_{coord}_{self.approach}_{self.feature_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # 2. Bar plot (mean absolute SHAP values)
            plt.figure(figsize=FIGURE_SIZE)
            shap.plots.bar(shap_values, max_display=max_display, show=False)
            plt.title(f"SHAP feature importance for {coord} coordinate")

            ax = plt.gca()
            ax.set_xlabel("Mean SHAP value (absolute)", fontsize=FONT_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
            ax.set_xlim(0, max_bar_importance * 1.1)

            plt.tight_layout()
            plt.savefig(
                f"shap_importance_bar_{coord}_{self.approach}_{self.feature_type}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            print(f"\nTop features for {coord} prediction (SHAP-based):")
            print(shap_results["feature_importance"][coord].head(10))

        print(f"\nSHAP analysis completed. All visualizations saved.")

        return shap_results

    def visualize_predictions(self, X_test, y_test):
        """
        Visualize prediction results on a geographic map
        """
        if self.models is None:
            raise ValueError("Models not trained yet")

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

        y_pred_array = self.models["multi_output"].predict(X_test)
        y_pred = pd.DataFrame(
            y_pred_array,
            columns=["relative_x", "relative_y", "relative_z"],
            index=y_test.index,
        )

        true_absolute = pd.DataFrame(index=y_test.index)
        pred_absolute = pd.DataFrame(index=y_test.index)

        for i in range(len(y_test)):
            lat, lon, depth = cartesian_to_geographic(
                y_test["relative_x"].iloc[i],
                y_test["relative_y"].iloc[i],
                y_test["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            true_absolute.loc[y_test.index[i], "lat"] = lat
            true_absolute.loc[y_test.index[i], "lon"] = lon
            true_absolute.loc[y_test.index[i], "depth"] = depth

            lat, lon, depth = cartesian_to_geographic(
                y_pred["relative_x"].iloc[i],
                y_pred["relative_y"].iloc[i],
                y_pred["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            pred_absolute.loc[y_test.index[i], "lat"] = lat
            pred_absolute.loc[y_test.index[i], "lon"] = lon
            pred_absolute.loc[y_test.index[i], "depth"] = depth

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = plt.axes(projection=ccrs.Mercator())

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        buffer = 0.3
        min_lon = min(true_absolute["lon"].min(), pred_absolute["lon"].min()) - buffer
        max_lon = max(true_absolute["lon"].max(), pred_absolute["lon"].max()) + buffer
        min_lat = min(true_absolute["lat"].min(), pred_absolute["lat"].min()) - buffer
        max_lat = max(true_absolute["lat"].max(), pred_absolute["lat"].max()) + buffer

        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        ax.scatter(
            true_absolute["lon"],
            true_absolute["lat"],
            c="blue",
            s=30,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            label="True Epicenter",
        )

        ax.scatter(
            pred_absolute["lon"],
            pred_absolute["lat"],
            c="red",
            s=30,
            alpha=0.7,
            marker="x",
            transform=ccrs.PlateCarree(),
            label="Predicted Epicenter",
        )

        for i in range(len(true_absolute)):
            ax.plot(
                [true_absolute["lon"].iloc[i], pred_absolute["lon"].iloc[i]],
                [true_absolute["lat"].iloc[i], pred_absolute["lat"].iloc[i]],
                "k-",
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )

        ax.scatter(
            self.mainshock["longitude"],
            self.mainshock["latitude"],
            c="yellow",
            s=200,
            marker="*",
            edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="Mainshock",
        )

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": FONT_SIZE}
        gl.ylabel_style = {"size": FONT_SIZE}

        plt.legend(loc="lower left")

        plt.savefig(
            f"xgboost_prediction_results_geographic_{self.approach}.png",
            dpi=300,
            bbox_inches="tight",
        )

        earth_radius = 6371.0

        lat_diff_deg = np.abs(true_absolute["lat"] - pred_absolute["lat"])
        lon_diff_deg = np.abs(true_absolute["lon"] - pred_absolute["lon"])
        lat_diff_km = lat_diff_deg * (np.pi / 180) * earth_radius
        avg_lat = (true_absolute["lat"] + pred_absolute["lat"]) / 2
        lon_diff_km = (
            lon_diff_deg * (np.pi / 180) * earth_radius * np.cos(np.radians(avg_lat))
        )
        depth_diff_km = np.abs(true_absolute["depth"] - pred_absolute["depth"])
        distance_3d_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2 + depth_diff_km**2)
        distance_2d_km = np.sqrt(lat_diff_km**2 + lon_diff_km**2)

        print("Prediction Error Statistics:")
        print(f"Mean latitude error: {lat_diff_km.mean():.2f} km")
        print(f"Mean longitude error: {lon_diff_km.mean():.2f} km")
        print(f"Mean depth error: {depth_diff_km.mean():.2f} km")
        print(f"Mean 3D error: {distance_3d_km.mean():.2f} km")
        print(f"Median 3D error: {distance_3d_km.median():.2f} km")
        print(f"Mean 2D error: {distance_2d_km.mean():.2f} km")

        plt.figure(figsize=FIGURE_SIZE)
        plt.hist(distance_3d_km, bins=20, alpha=0.7)
        plt.axvline(
            distance_3d_km.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {distance_3d_km.mean():.2f} km",
        )
        plt.axvline(
            distance_3d_km.median(),
            color="g",
            linestyle="--",
            label=f"Median: {distance_3d_km.median():.2f} km",
        )
        plt.xlabel("Error (km)")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(f"xgboost_prediction_error_histogram_{self.approach}.png", dpi=300)

        self.plot_3d_aftershocks(y_test, y_pred)

        return (
            true_absolute,
            pred_absolute,
            {
                "3d": distance_3d_km,
                "lat": lat_diff_km,
                "lon": lon_diff_km,
                "depth": depth_diff_km,
            },
        )

    def plot_3d_aftershocks(self, y_test, y_pred):
        """
        Create 3D visualization of aftershock locations showing true vs predicted positions
        with depth information.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if isinstance(y_pred, np.ndarray):
            y_pred = pd.DataFrame(
                y_pred,
                columns=["relative_x", "relative_y", "relative_z"],
                index=y_test.index,
            )

        true_absolute = pd.DataFrame(index=y_test.index)
        pred_absolute = pd.DataFrame(index=y_test.index)

        for i in range(len(y_test)):
            lat, lon, depth = cartesian_to_geographic(
                y_test["relative_x"].iloc[i],
                y_test["relative_y"].iloc[i],
                y_test["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            true_absolute.loc[y_test.index[i], "lat"] = lat
            true_absolute.loc[y_test.index[i], "lon"] = lon
            true_absolute.loc[y_test.index[i], "depth"] = depth

            lat, lon, depth = cartesian_to_geographic(
                y_pred["relative_x"].iloc[i],
                y_pred["relative_y"].iloc[i],
                y_pred["relative_z"].iloc[i],
                self.mainshock["latitude"],
                self.mainshock["longitude"],
                self.mainshock["depth"],
            )
            pred_absolute.loc[y_test.index[i], "lat"] = lat
            pred_absolute.loc[y_test.index[i], "lon"] = lon
            pred_absolute.loc[y_test.index[i], "depth"] = depth

        fig = plt.figure(figsize=(17, 16))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            true_absolute["lon"],
            true_absolute["lat"],
            true_absolute["depth"],
            c="blue",
            s=30,
            alpha=0.6,
            label="True Epicenter",
        )

        ax.scatter(
            pred_absolute["lon"],
            pred_absolute["lat"],
            pred_absolute["depth"],
            c="red",
            s=30,
            alpha=0.6,
            label="Predicted Epicenter",
        )

        for i in range(len(true_absolute)):
            ax.plot(
                [true_absolute["lon"].iloc[i], pred_absolute["lon"].iloc[i]],
                [true_absolute["lat"].iloc[i], pred_absolute["lat"].iloc[i]],
                [true_absolute["depth"].iloc[i], pred_absolute["depth"].iloc[i]],
                "k-",
                alpha=0.15,
            )

        ax.scatter(
            [self.mainshock["longitude"]],
            [self.mainshock["latitude"]],
            [self.mainshock["depth"]],
            c="green",
            s=100,
            marker="*",
            label="Mainshock",
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Depth (km)")

        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10

        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)
        ax.locator_params(axis="z", nbins=5)

        ax.invert_zaxis()

        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1))

        # Adjust view angle for better visualization and to minimize label overlap
        ax.view_init(elev=25, azim=50)

        plt.tight_layout(pad=3.0)

        plt.savefig(
            f"3d_location_visualization_{self.approach}_{self.feature_type}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close("all")
        print(f"Created 3D visualization for {self.approach}")
