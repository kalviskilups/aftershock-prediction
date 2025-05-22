#!/usr/bin/env python3
"""
Process raw seismic data and save to pickle format.

This script loads data from the SeisBench Iquique dataset, selects stations from the CX network, applies instrument
response correction, and saves the processed data to a pickle file for later use in the aftershock prediction workflow.
"""
import os
import argparse
import sys
from pathlib import Path

from data.loader import load_aftershock_data_with_CX_waveforms
from data.preprocessor import standardize_waveforms


def main():
    """
    Main function for data processing.
    """
    parser = argparse.ArgumentParser(
        description="Process seismic data and save to pickle format"
    )

    parser.add_argument(
        "--inventory", required=True, help="Path to StationXML or SC3-ML inventory file"
    )

    parser.add_argument(
        "--output",
        default="aftershock_data_CX_only.pkl",
        help="Output pickle file path (default: aftershock_data_CX_only.pkl)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum number of stations to select per event (default: 15)",
    )

    parser.add_argument(
        "--min-stations",
        type=int,
        default=5,
        help="Minimum number of stations required to include an event (default: 5)",
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize waveform lengths after processing",
    )

    parser.add_argument(
        "--target-length",
        type=int,
        default=14636,
        help="Target length for waveforms if standardizing (default: 14636 samples)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.inventory):
        print(f"Error: Inventory file '{args.inventory}' not found")
        return 1

    print(f"Loading data from the Iquique dataset...")
    print(f"Using inventory file: {args.inventory}")
    print(
        f"Selecting up to {args.top_n} stations per event with at least {args.min_stations} stations"
    )

    data_dict = load_aftershock_data_with_CX_waveforms(
        args.inventory, top_n=args.top_n, min_stations=args.min_stations
    )

    if args.standardize:
        print(f"Standardizing waveforms to length {args.target_length} samples...")

        if data_dict:
            first_event_key = next(iter(data_dict))
            first_event_data = data_dict[first_event_key]
            is_multi_station = isinstance(first_event_data, dict) and any(
                isinstance(k, str) and "." in k for k in first_event_data.keys()
            )
            data_format = "multi_station" if is_multi_station else "single_station"
        else:
            print("Warning: No data to standardize")
            data_format = "multi_station"

        standardize_waveforms(data_dict, data_format, args.target_length)

    output_path = args.output
    print(f"Saving processed data to: {output_path}")

    import pickle

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    event_count = len(data_dict)

    # Count total station recordings for multi-station format
    if data_dict and isinstance(next(iter(data_dict.values())), dict):
        station_count = sum(len(stations) for stations in data_dict.values())
        print(
            f"Successfully processed {event_count} events with {station_count} total station recordings"
        )
    else:
        print(f"Successfully processed {event_count} events")

    return 0


if __name__ == "__main__":
    sys.exit(main())
