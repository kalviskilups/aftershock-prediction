#!/usr/bin/env python3
"""
Hyperparameter optimization for aftershock prediction models using Optuna.

This script performs hyperparameter optimization for XGBoost models using Optuna,
which employs efficient search algorithms.
"""
import os
import sys
import argparse
import pickle
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from models.xgboost_predictor import XGBoostPredictor
from utils.validation import validate_features


def objective(trial, X, y, groups, n_splits=5):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        X: Feature DataFrame
        y: Target DataFrame with coordinates
        groups: Groups for GroupKFold cross-validation
        n_splits: Number of cross-validation splits

    Returns:
        Mean RMSE across all CV folds and coordinates
    """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
    }

    cv = GroupKFold(n_splits=n_splits)

    coordinate_errors = {"relative_x": [], "relative_y": [], "relative_z": [], "3d": []}

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        base_model = XGBRegressor(**params)
        model = MultiOutputRegressor(base_model)

        model.fit(X_train, y_train[["relative_x", "relative_y", "relative_z"]])

        y_pred = model.predict(X_test)

        for i, coord in enumerate(["relative_x", "relative_y", "relative_z"]):
            mse = mean_squared_error(y_test[coord], y_pred[:, i])
            rmse = np.sqrt(mse)
            coordinate_errors[coord].append(rmse)

        distance_3d = np.sqrt(
            (y_pred[:, 0] - y_test["relative_x"]) ** 2
            + (y_pred[:, 1] - y_test["relative_y"]) ** 2
            + (y_pred[:, 2] - y_test["relative_z"]) ** 2
        )
        coordinate_errors["3d"].append(np.mean(distance_3d))

    mean_errors = {
        coord: np.mean(errors) for coord, errors in coordinate_errors.items()
    }

    trial_info = f"Trial {trial.number}: "
    trial_info += f"3D RMSE = {mean_errors['3d']:.2f} km, "
    trial_info += f"X RMSE = {mean_errors['relative_x']:.2f} km, "
    trial_info += f"Y RMSE = {mean_errors['relative_y']:.2f} km, "
    trial_info += f"Z RMSE = {mean_errors['relative_z']:.2f} km"
    print(trial_info)

    for coord, error in mean_errors.items():
        trial.set_user_attr(f"rmse_{coord}", float(error))

    return mean_errors["3d"]


def load_and_prepare_data(
    data_file, approach="multi_station", feature_type="all", validation_level="minimal"
):
    """
    Load and prepare data for optimization.

    Args:
        data_file: Path to pickle file with preprocessed data
        approach: Analysis approach ('multi_station' or 'best_station')
        feature_type: Type of features to extract ('all', 'signal', or 'physics')
        validation_level: Level of validation to perform

    Returns:
        X: Feature DataFrame
        y: Target DataFrame
        groups: Groups for GroupKFold cross-validation
    """

    print(f"Loading data from {data_file}")
    print(f"Approach: {approach}")
    print(f"Feature type: {feature_type}")

    predictor = XGBoostPredictor(
        data_file=data_file,
        validation_level=validation_level,
        approach=approach,
        feature_type=feature_type,
    )

    predictor.find_mainshock()
    predictor.create_relative_coordinate_dataframe()

    X, y = predictor.prepare_dataset()
    print(f"Prepared dataset with {len(X)} samples and {X.shape[1]} features")

    X, y = validate_features(X, y)

    if "event_id" in y.columns:
        groups = y["event_id"]
        print(
            f"Using 'event_id' as groups for cross-validation ({len(set(groups))} unique groups)"
        )
    elif "event_date" in y.columns:
        groups = y["event_date"]
        print(
            f"Using 'event_date' as groups for cross-validation ({len(set(groups))} unique groups)"
        )
    else:
        print(
            "Warning: No appropriate grouping column found. Using random groups for cross-validation."
        )
        groups = np.random.randint(0, 5, size=len(X))

    for col in ["event_id", "event_date"]:
        if col in X.columns:
            X = X.drop(col, axis=1)

    y = y[["relative_x", "relative_y", "relative_z"]]

    numeric_columns = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
    ]
    X = X[numeric_columns]

    X = X.fillna(X.mean())

    return X, y, groups


def run_optimization(
    data_file,
    approach="multi_station",
    feature_type="all",
    validation_level="minimal",
    n_trials=100,
    n_splits=5,
    output_dir="optuna_results",
):
    """
    Run hyperparameter optimization with Optuna.

    Args:
        data_file: Path to pickle file with preprocessed data
        approach: Analysis approach ('multi_station' or 'best_station')
        feature_type: Type of features to extract ('all', 'signal', or 'physics')
        validation_level: Level of validation to perform
        n_trials: Number of optimization trials
        n_splits: Number of cross-validation splits
        output_dir: Directory to save results

    Returns:
        study: Optuna study object with optimization results
    """

    os.makedirs(output_dir, exist_ok=True)

    X, y, groups = load_and_prepare_data(
        data_file,
        approach=approach,
        feature_type=feature_type,
        validation_level=validation_level,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"xgboost_{approach}_{feature_type}_{timestamp}"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print("\n" + "=" * 80)
    print(f"STARTING HYPERPARAMETER OPTIMIZATION WITH {n_trials} TRIALS")
    print(f"CV Strategy: GroupKFold with {n_splits} splits")
    print(f"Optimization metric: 3D RMSE")
    print("=" * 80 + "\n")

    start_time = time.time()

    study.optimize(
        lambda trial: objective(trial, X, y, groups, n_splits=n_splits),
        n_trials=n_trials,
    )

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best 3D RMSE: {study.best_value:.4f} km")
    print("Best hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print("=" * 80 + "\n")

    results_file = os.path.join(output_dir, f"{study_name}_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(study, f)

    params_file = os.path.join(output_dir, f"{study_name}_best_params.json")
    with open(params_file, "w") as f:
        json.dump(study.best_params, f, indent=4)

    additional_results = {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_3d_rmse": study.best_value,
        "best_x_rmse": study.best_trial.user_attrs.get("rmse_relative_x"),
        "best_y_rmse": study.best_trial.user_attrs.get("rmse_relative_y"),
        "best_z_rmse": study.best_trial.user_attrs.get("rmse_relative_z"),
        "optimization_duration": duration,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "approach": approach,
        "feature_type": feature_type,
        "timestamp": timestamp,
    }

    details_file = os.path.join(output_dir, f"{study_name}_details.json")
    with open(details_file, "w") as f:
        json.dump(additional_results, f, indent=4)

    fig = plot_optimization_history(study)
    history_file = os.path.join(output_dir, f"{study_name}_history.png")
    fig.write_image(history_file)

    fig = plot_param_importances(study)
    importance_file = os.path.join(output_dir, f"{study_name}_param_importance.png")
    fig.write_image(importance_file)

    print(f"Visualizations saved to {output_dir}")

    print(f"All results saved to {output_dir}")

    return study


def main():
    """
    Main function for hyperparameter optimization.
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for aftershock prediction models"
    )

    parser.add_argument(
        "--data", required=True, help="Path to pickle file with preprocessed data"
    )

    parser.add_argument(
        "--approach",
        choices=["multi_station", "best_station"],
        default="multi_station",
        help="Analysis approach to use (default: multi_station)",
    )

    parser.add_argument(
        "--feature-type",
        choices=["all", "physics", "signal"],
        default="signal",
        help="Type of features to use (default: all)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of cross-validation splits (default: 5)",
    )

    parser.add_argument(
        "--output-dir",
        default="optuna_results",
        help="Directory to save results (default: optuna_results)",
    )

    parser.add_argument(
        "--validation",
        choices=["full", "minimal", "none"],
        default="minimal",
        help="Level of validation to perform (default: minimal)",
    )

    args = parser.parse_args()

    try:
        study = run_optimization(
            args.data,
            approach=args.approach,
            feature_type=args.feature_type,
            validation_level=args.validation,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
