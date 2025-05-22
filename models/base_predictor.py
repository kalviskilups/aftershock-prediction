"""
Base class for aftershock location prediction models.
"""

import numpy as np
import pandas as pd
import pickle
import os
import time

from config import MAINSHOCK_INFO, DEFAULT_WAVEFORM_LENGTH
from utils.coordinate_utils import geographic_to_cartesian
from data.preprocessor import standardize_waveforms
from utils.validation import validate_data_integrity, validate_coordinate_conversion


class BasePredictor:
    """
    Base class for predicting aftershock locations.
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
        self.data_dict = None
        self.aftershocks_df = None
        self.mainshock_key = None
        self.mainshock = None
        self.models = None
        self.feature_importances = None
        self.scaler = None
        self.validation_level = validation_level
        self.validation_results = {}
        self.approach = approach
        self.feature_type = feature_type
        self.data_format = None

        print(f"Validation level: {validation_level}")
        print(f"Analysis approach: {approach}")
        print(f"Feature type: {feature_type}")

        # Print more detailed feature tier description
        if feature_type == "signal":
            print("Using Tier A (signal statistics) features only")
        elif feature_type == "physics":
            print("Using Tier C (source physics) features only")
        elif feature_type in ["all"]:
            print("Using all features (Tiers A+C: signal + source physics)")

        # Load data
        if data_file and os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            if data_file.lower().endswith(".pkl") or data_file.lower().endswith(
                ".pickle"
            ):
                self.load_data(data_file)
            else:
                raise ValueError(
                    f"Unsupported file type: {data_file}. Must be .pkl, .pickle"
                )
        else:
            print("No file found or invalid path provided.")
            return

        self.standardize_waveforms(target_length=DEFAULT_WAVEFORM_LENGTH)

        if validation_level != "none":
            self.validate_data()

    def load_data(self, file_path):
        """
        Load data from file and set data_format
        """
        with open(file_path, "rb") as f:
            self.data_dict = pickle.load(f)

        is_multi_station = False
        if self.data_dict:
            first_event_key = next(iter(self.data_dict))
            first_event_data = self.data_dict[first_event_key]
            is_multi_station = isinstance(first_event_data, dict) and any(
                isinstance(k, str) and "." in k for k in first_event_data.keys()
            )

        self.data_format = "multi_station" if is_multi_station else "single_station"
        print(f"Detected {self.data_format} data format")
        return self.data_dict

    def standardize_waveforms(self, target_length=DEFAULT_WAVEFORM_LENGTH):
        """
        Standardize all waveforms to the same length by padding or trimming
        """
        return standardize_waveforms(self.data_dict, self.data_format, target_length)

    def find_mainshock(self):
        """
        Identify the mainshock in the dataset
        For the Iquique sequence, the mainshock occurred on April 1, 2014
        """
        # Use manual specification of Iquique mainshock (from USGS catalog)
        self.mainshock = MAINSHOCK_INFO

        print("Using manually specified mainshock coordinates from USGS catalog")

        # Create a mainshock key (we don't have this exact event in our dataset)
        self.mainshock_key = (
            self.mainshock["origin_time"],
            self.mainshock["latitude"],
            self.mainshock["longitude"],
            self.mainshock["depth"],
        )

        print(f"Mainshock: {self.mainshock}")

        return self.mainshock_key

    def create_relative_coordinate_dataframe(self):
        """
        Create a DataFrame with all events and their coordinates
        relative to the mainshock, supporting multi-station format
        """

        if self.mainshock_key is None:
            self.find_mainshock()

        mainshock_lat = self.mainshock["latitude"]
        mainshock_lon = self.mainshock["longitude"]
        mainshock_depth = self.mainshock["depth"]

        events = []

        # Process each event differently based on data format
        if self.data_format == "multi_station":
            for event_key, stations_data in self.data_dict.items():
                origin_time, lat, lon, depth = event_key

                # Calculate relative coordinates
                x, y, z = geographic_to_cartesian(
                    lat, lon, depth, mainshock_lat, mainshock_lon, mainshock_depth
                )

                # For each station recording of this event
                for station_key, station_data in stations_data.items():
                    events.append(
                        {
                            "origin_time": origin_time,
                            "absolute_lat": lat,
                            "absolute_lon": lon,
                            "absolute_depth": depth,
                            "relative_x": x,  # East-West (km)
                            "relative_y": y,  # North-South (km)
                            "relative_z": z,  # Depth difference (km)
                            "waveform": station_data["waveform"],
                            "station_key": station_key,
                            "station_distance": station_data["station_distance"],
                            "selection_score": station_data["selection_score"],
                            "metadata": station_data["metadata"],
                        }
                    )
        else:
            for event_key, event_data in self.data_dict.items():
                origin_time, lat, lon, depth = event_key

                x, y, z = geographic_to_cartesian(
                    lat, lon, depth, mainshock_lat, mainshock_lon, mainshock_depth
                )

                events.append(
                    {
                        "origin_time": origin_time,
                        "absolute_lat": lat,
                        "absolute_lon": lon,
                        "absolute_depth": depth,
                        "relative_x": x,  # East-West (km)
                        "relative_y": y,  # North-South (km)
                        "relative_z": z,  # Depth difference (km)
                        "waveform": event_data["waveform"],
                        "metadata": event_data["metadata"],
                        "is_mainshock": (event_key == self.mainshock_key),
                    }
                )

        self.aftershocks_df = pd.DataFrame(events)

        self.aftershocks_df["event_date"] = pd.to_datetime(
            self.aftershocks_df["origin_time"]
        ).dt.date

        if self.data_format == "multi_station":
            self.aftershocks_df["event_id"] = self.aftershocks_df.apply(
                lambda row: f"{row['origin_time']}_{row['absolute_lat']}_{row['absolute_lon']}_{row['absolute_depth']}",
                axis=1,
            )

        # Validate coordinate conversion if required
        if self.validation_level != "none":
            validate_coordinate_conversion()

        return self.aftershocks_df

    def validate_data(self):
        """
        Run validation checks on the dataset
        """
        validate_data_integrity(self.data_dict, self.data_format)
        validate_coordinate_conversion()
        return True

    def prepare_dataset(self):
        """
        Prepare dataset for machine learning by extracting features from waveforms.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement prepare_dataset method")

    def train_model(self, X, y, perform_shap=False):
        """
        Train the model to predict aftershock locations.
        Must be implemented by subclasses.

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            perform_shap: Whether to perform SHAP analysis

        Returns:
            X_test, y_test: Test data for evaluation
        """
        raise NotImplementedError("Subclasses must implement train_model method")

    def visualize_predictions(self, X_test, y_test):
        """
        Visualize prediction results.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement visualize_predictions method"
        )

    def run_complete_workflow(self, perform_shap=False):
        """
        Run the complete analysis workflow.

        Args:
            perform_shap: Whether to perform SHAP analysis (default: False)

        Returns:
            Dictionary with results
        """
        start_time = time.time()

        print("\n" + "=" * 70)
        print(f"AFTERSHOCK ANALYSIS WITH {self.approach.upper()} APPROACH".center(70))
        print(f"USING {self.feature_type.upper()} FEATURES".center(70))
        print("=" * 70)

        # 1. Find the mainshock
        self.find_mainshock()

        # 2. Create relative coordinate dataframe
        self.create_relative_coordinate_dataframe()

        if self.data_format == "multi_station":
            event_count = (
                len(set(self.aftershocks_df["event_id"]))
                if "event_id" in self.aftershocks_df.columns
                else len(self.aftershocks_df)
            )
            print(
                f"Created dataframe with {event_count} events and {len(self.aftershocks_df)} station recordings"
            )

            if "event_id" in self.aftershocks_df.columns:
                stations_per_event = self.aftershocks_df.groupby("event_id").size()
                print(f"Stations per event:")
                print(f"  Mean: {stations_per_event.mean():.2f}")
                print(f"  Median: {stations_per_event.median()}")
                print(f"  Min: {stations_per_event.min()}")
                print(f"  Max: {stations_per_event.max()}")
        else:
            print(f"Created dataframe with {len(self.aftershocks_df)} events")

        # 3. Prepare dataset for machine learning
        X, y = self.prepare_dataset()
        print(f"Prepared dataset with {len(X)} samples and {X.shape[1]} features")

        # 4. Train the model
        X_test, y_test = self.train_model(X, y, perform_shap=perform_shap)

        # 5. Visualize predictions
        true_abs, pred_abs, errors = self.visualize_predictions(X_test, y_test)

        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)"
        )

        results = {
            "models": self.models,
            "feature_importances": self.feature_importances,
            "mainshock": self.mainshock,
            "aftershocks_df": self.aftershocks_df,
            "test_results": {
                "true_absolute": true_abs,
                "pred_absolute": pred_abs,
                "errors": errors,
            },
            "validation_results": self.validation_results,
        }

        if hasattr(self, "shap_results") and self.shap_results is not None:
            results["shap_results"] = self.shap_results

        return results
