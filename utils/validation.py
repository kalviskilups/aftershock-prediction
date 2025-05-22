"""
Utilities for data validation and integrity checks.
"""

import numpy as np
from utils.coordinate_utils import geographic_to_cartesian, cartesian_to_geographic
from config import COORDINATE_VALIDATION_THRESHOLD, MAINSHOCK_INFO


def validate_data_integrity(data_dict, data_format):
    """
    Run validation checks on the dataset to catch potential issues

    Args:
        data_dict: Dictionary of event data
        data_format: Format of the data ('multi_station' or 'single_station')

    Returns:
        validated: Whether all checks passed
        issues: List of issues found
    """
    print("\n" + "=" * 50)
    print("VALIDATING DATA INTEGRITY")
    print("=" * 50)

    issues = []

    # Check waveform shapes and NaNs
    print("Checking waveform consistency...")
    expected_shape = None
    waveform_issues = 0

    # Check 20 waveforms (multi-station or single-station)
    if data_format == "multi_station":
        waveform_check_count = 0
        for event_key, stations_data in data_dict.items():
            for station_key, station_data in stations_data.items():
                waveform = station_data["waveform"]

                # Check number of components
                if waveform.shape[0] != 3:
                    issues.append(
                        f"Event {event_key}, Station {station_key} has {waveform.shape[0]} components instead of 3"
                    )
                    waveform_issues += 1
                    continue

                # Set expected length from first waveform
                if expected_shape is None:
                    expected_shape = waveform.shape[1]

                # Check length consistency
                if waveform.shape[1] != expected_shape:
                    issues.append(
                        f"Event {event_key}, Station {station_key} has length {waveform.shape[1]} instead of {expected_shape}"
                    )
                    waveform_issues += 1

                    # Check for NaNs
                    issues.append(
                        f"Event {event_key}, Station {station_key} contains NaN values"
                    )
                    waveform_issues += 1

                waveform_check_count += 1
                if waveform_check_count >= 20:
                    print(
                        f"✓ First {waveform_check_count} waveforms checked (skipping rest for speed)"
                    )
                    break

            if waveform_check_count >= 20:
                break
    else:
        for i, (event_key, event_data) in enumerate(data_dict.items()):
            waveform = event_data["waveform"]

            if waveform.shape[0] != 3:
                issues.append(
                    f"Event {event_key} has {waveform.shape[0]} components instead of 3"
                )
                waveform_issues += 1
                continue

            # Set expected length from first waveform
            if expected_shape is None:
                expected_shape = waveform.shape[1]

            # Check length consistency
            if waveform.shape[1] != expected_shape:
                issues.append(
                    f"Event {event_key} has length {waveform.shape[1]} instead of {expected_shape}"
                )
                waveform_issues += 1

            # Check for NaNs
            if np.isnan(waveform).any():
                issues.append(f"Event {event_key} contains NaN values")
                waveform_issues += 1

            # Only check first 20 events if dataset is large
            if i >= 20 and len(data_dict) > 50:
                print(
                    f"✓ First 20/{len(data_dict)} waveforms checked (skipping rest for speed)"
                )
                break

    if waveform_issues == 0:
        print(
            f"✓ All checked waveforms have consistent shape (3, {expected_shape}) with no NaNs"
        )
    else:
        print(f"❌ Found {waveform_issues} waveform issues!")

    validated = len(issues) == 0

    if validated:
        print("All data integrity checks passed!")
    else:
        print(f"Found {len(issues)} issues. Will attempt to proceed with caution.")
        if len(issues) <= 5:
            for i, issue in enumerate(issues[:5]):
                print(f"  {i+1}. {issue}")
        else:
            for i, issue in enumerate(issues[:5]):
                print(f"  {i+1}. {issue}")
            print(f"  ... and {len(issues) - 5} more issues")

    return validated, issues


def validate_coordinate_conversion(num_points=50):
    """
    Test coordinate conversion functions for accuracy by doing a round-trip conversion.

    Args:
        num_points: Number of random points to test

    Returns:
        passed: Whether all errors are below threshold
        max_errors: Dictionary of maximum errors in each dimension
    """
    print("\n" + "=" * 50)
    print("VALIDATING COORDINATE CONVERSION")
    print("=" * 50)

    np.random.seed(44)
    lats = np.random.uniform(-20.5, -19.5, num_points)
    lons = np.random.uniform(-71.0, -70.0, num_points)
    depths = np.random.uniform(10, 50, num_points)

    ref_lat = MAINSHOCK_INFO["latitude"]
    ref_lon = MAINSHOCK_INFO["longitude"]
    ref_depth = MAINSHOCK_INFO["depth"]

    lat_errors = []
    lon_errors = []
    depth_errors = []
    distance_errors = []

    for i in range(num_points):
        # Original coordinates
        orig_lat, orig_lon, orig_depth = lats[i], lons[i], depths[i]

        # Convert to Cartesian
        x, y, z = geographic_to_cartesian(
            orig_lat, orig_lon, orig_depth, ref_lat, ref_lon, ref_depth
        )

        # Convert back to geographic
        new_lat, new_lon, new_depth = cartesian_to_geographic(
            x, y, z, ref_lat, ref_lon, ref_depth
        )

        # Calculate errors
        lat_error = abs(orig_lat - new_lat)
        lon_error = abs(orig_lon - new_lon)
        depth_error = abs(orig_depth - new_depth)

        # Convert lat/lon errors to approximate distances in km
        earth_radius = 6371.0
        lat_error_km = lat_error * (np.pi / 180) * earth_radius
        lon_error_km = (
            lon_error * (np.pi / 180) * earth_radius * np.cos(np.radians(orig_lat))
        )

        # Calculate 3D distance error
        distance_error = np.sqrt(lat_error_km**2 + lon_error_km**2 + depth_error**2)

        lat_errors.append(lat_error_km)
        lon_errors.append(lon_error_km)
        depth_errors.append(depth_error)
        distance_errors.append(distance_error)

    # Get maximum errors
    max_errors = {
        "lat_km": max(lat_errors),
        "lon_km": max(lon_errors),
        "depth_km": max(depth_errors),
        "3d_distance_km": max(distance_errors),
    }

    # Check if errors are below threshold
    threshold = COORDINATE_VALIDATION_THRESHOLD
    passed = all(error < threshold for error in max_errors.values())

    print(f"Round-trip conversion test results ({num_points} random points):")
    print(f"  Max latitude error: {max_errors['lat_km']*1000:.2f} meters")
    print(f"  Max longitude error: {max_errors['lon_km']*1000:.2f} meters")
    print(f"  Max depth error: {max_errors['depth_km']*1000:.2f} meters")
    print(f"  Max 3D distance error: {max_errors['3d_distance_km']*1000:.2f} meters")

    if passed:
        print("All conversion errors below threshold (100 meters)")
    else:
        print("Some conversion errors exceed threshold (100 meters)")
        print("This could impact location accuracy - proceed with caution")

    return passed, max_errors


def validate_features(X, y):
    """
    Validate feature preparation and check for target leakage

    Args:
        X: Feature DataFrame
        y: Target DataFrame

    Returns:
        X: Cleaned feature DataFrame
        y: Target DataFrame
    """
    print("\n" + "=" * 50)
    print("VALIDATING FEATURE PREPARATION")
    print("=" * 50)

    print("Checking for target leakage...")

    # List of patterns that indicate potential target leakage
    forbidden_patterns = [
        "station_distance",
        "distance_normalized",
        "selection_score",
        "epicentral_distance",
        "relative_x",
        "relative_y",
        "relative_z",
        "absolute_lat",
        "absolute_lon",
        "absolute_depth",
        "station_lat",
        "station_lon",
        "station_elev",
        "_ranking_station_distance",
    ]

    leaked_features = []
    for pattern in forbidden_patterns:
        matching_cols = [col for col in X.columns if pattern in col]
        leaked_features.extend(matching_cols)

    leaked_features = list(set(leaked_features))

    clean = len(leaked_features) == 0

    if clean:
        print("No target leakage detected")
    else:
        print(f"Found {len(leaked_features)} leaked features: {leaked_features}")
        print("Removing leaked features to prevent data leakage")
        X = X.drop(leaked_features, axis=1)

    return X, y
