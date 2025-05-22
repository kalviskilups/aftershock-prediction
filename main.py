#!/usr/bin/env python3
"""
Aftershock prediction command-line interface.
"""
import argparse
import os
import sys
import time

from models.xgboost_predictor import XGBoostPredictor


def main():
    """
    Main entry point for the aftershock prediction command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Predict aftershock locations using seismic data"
    )

    parser.add_argument(
        "--data", required=True, help="Path to pickle file with preprocessed data"
    )

    parser.add_argument(
        "--feature-type",
        choices=["all", "physics", "signal"],
        default="all",
        help="Type of features to use (default: all)",
    )

    parser.add_argument(
        "--approach",
        choices=["multi_station", "best_station"],
        default="multi_station",
        help="Analysis approach to use (default: multi_station)",
    )

    parser.add_argument(
        "--validation",
        choices=["full", "minimal", "none"],
        default="full",
        help="Level of validation to perform (default: full)",
    )

    parser.add_argument(
        "--shap", action="store_true", help="Perform SHAP analysis on trained models"
    )

    parser.add_argument(
        "--results-dir",
        default="aftershock_results",
        help="Directory to save results (default: aftershock_results)",
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    original_dir = os.getcwd()
    os.chdir(args.results_dir)

    start_time = time.time()

    try:
        print("Initializing XGBoost predictor...")
        predictor = XGBoostPredictor(
            data_file=os.path.join(original_dir, args.data),
            validation_level=args.validation,
            approach=args.approach,
            feature_type=args.feature_type,
        )

        print("Running analysis workflow...")
        results = predictor.run_complete_workflow(perform_shap=args.shap)

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE".center(70))
        print("=" * 70)
        print(
            f"Total execution time: {duration:.1f} seconds ({duration/60:.1f} minutes)"
        )
        print(f"Results saved to: {os.path.abspath(args.results_dir)}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
