"""
Feature extraction from seismic waveforms.
"""

import numpy as np
import pandas as pd
from scipy import fft, signal, integrate
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from tqdm import tqdm
from config import DEFAULT_SAMPLING_RATE


def extract_waveform_features(waveform, feature_type="all", metadata=None):
    """
    Extract features from the 3-component waveform data based on selected feature type

    Feature types are organized in tiers:
    - "signal": Basic signal statistics (Tier A)
    - "physics": Source physics parameters requiring explicit models (Tier C)
    - "all": All feature types

    Args:
        waveform: 3-component seismic waveform array
        feature_type: Type of features to extract
        metadata: Optional metadata dictionary for the station

    Returns:
        Dictionary of extracted features
    """
    features = {}

    waveform_ms = waveform

    # Assume sampling rate of 100 Hz
    sampling_rate = DEFAULT_SAMPLING_RATE

    # Get depth-dependent velocity and density
    def get_velocity_density(depth_km, model="iasp91"):
        """Get shear velocity and density at a given depth using a 1D Earth model"""
        # Simple lookup table for IASP91 model (depth in km, vs in km/s)
        iasp91_vs = {
            0: 3.36,  # Upper crust
            20: 3.75,  # Mid-crust
            35: 4.47,  # Moho
            71: 4.4827,  # Upper mantle
            120: 4.5,  # Upper mantle
            171: 4.5102,  # Upper mantle
            210: 4.5220,  # Upper mantle
            271: 4.6281,  # Upper mantle
            371: 4.8021,  # Upper mantle
        }

        # Find closest depth
        depths = np.array(list(iasp91_vs.keys()))
        idx = np.abs(depths - depth_km).argmin()

        # Get velocity in km/s
        beta = iasp91_vs[depths[idx]]

        # Estimate density
        alpha = (
            0.9409
            + 2.0947 * beta
            - 0.8206 * beta**2
            + 0.2683 * beta**3
            - 0.0251 * beta**4
        )
        rho = (
            1.6612 * alpha
            - 0.4721 * alpha**2
            + 0.0671 * alpha**3
            - 0.0043 * alpha**4
            + 0.000106 * alpha**5
        )

        return beta, rho

    # Joint fit of Brune spectrum and Q attenuation in log-log space
    def fit_brune_model_with_Q(freqs, displ_spec, R_km, beta_km_s, snr_mask=None):
        """
        Joint fit of Brune spectrum and Q attenuation in log-log space

        Args:
            freqs: frequencies array (Hz)
            displ_spec: displacement spectrum array
            R_km: distance in km
            beta_km_s: shear wave velocity in km/s
            snr_mask: boolean mask for frequencies with good SNR

        Returns:
            fc: corner frequency (Hz)
            Omega0: low-frequency plateau
            Q: quality factor
            success: whether the fit was successful
        """
        # Safety check for dimension mismatch
        if snr_mask is not None and len(snr_mask) != len(freqs):
            print(
                f"Warning: SNR mask length ({len(snr_mask)}) doesn't match frequency array length ({len(freqs)}). Using default frequency mask."
            )
            snr_mask = None

        # Apply SNR mask if provided (and verified)
        if snr_mask is not None and np.sum(snr_mask) > 10:
            f_fit = freqs[snr_mask]
            U_fit = displ_spec[snr_mask]
        else:
            # Default: use frequencies between 0.5 Hz and 80% of max
            mask = (freqs >= 0.5) & (freqs <= freqs.max() * 0.8)
            f_fit = freqs[mask]
            U_fit = displ_spec[mask]

        if len(f_fit) < 10:
            return 1.0, 1.0, 600, False

        # Ensure positive values for log transform
        U_fit = np.maximum(U_fit, 1e-15)

        # Convert to log space
        log_f = np.log10(f_fit)
        log_U = np.log10(U_fit)

        # Define Brune model with Q in log space
        def log_brune_Q(log_f, log_Omega0, log_fc, Q):
            f = 10**log_f
            Omega0, fc = 10**log_Omega0, 10**log_fc
            term = (
                Omega0
                / (1 + (f / fc) ** 2)
                * np.exp(np.pi * f * R_km / (Q * beta_km_s))
            )
            return np.log10(term)

        # Initial parameter guess
        log_Omega0_guess = np.log10(np.max(U_fit))

        try:
            # Curve fit in log space
            popt, _ = curve_fit(
                log_brune_Q,
                log_f,
                log_U,
                p0=[log_Omega0_guess, np.log10(1.0), 600],
                bounds=(
                    [log_Omega0_guess - 2, -1.0, 50],
                    [log_Omega0_guess + 2, 1.5, 1500],
                ),
                maxfev=5000,
            )

            # Convert back to linear space
            log_Omega0, log_fc, Q = popt
            Omega0, fc = 10**log_Omega0, 10**log_fc

            return fc, Omega0, Q, True

        except:
            print("Fitting failed, returning default values")
            return 1.0, np.max(U_fit), 600, False

    Fs = sampling_rate
    ny = Fs / 2.0
    T = waveform_ms.shape[1] / Fs

    b_vel, a_vel = butter(4, [0.30 / ny, 35.0 / ny], btype="bandpass")
    low_hp_disp = max(0.05, 2.5 / T)
    b_hp_disp, a_hp_disp = butter(2, low_hp_disp / ny, btype="highpass")

    velocity_filtered_components = []

    # Process each component
    for i, component in enumerate(["Z", "N", "E"]):
        velocity_filtered = filtfilt(b_vel, a_vel, waveform_ms[i])
        velocity_filtered_components.append(velocity_filtered)
        acceleration = np.gradient(velocity_filtered, 1 / sampling_rate)
        displacement = integrate.cumulative_trapezoid(
            velocity_filtered, dx=1 / Fs, initial=0
        )
        displacement_filtered = filtfilt(b_hp_disp, a_hp_disp, displacement)

        # Calculate spectrum on filtered velocity
        f, Pxx = signal.welch(velocity_filtered, fs=sampling_rate, nperseg=1024)

        # TIER A: Low-level signal statistics
        if feature_type in ["all", "signal"]:
            # Simple statistics (now on filtered velocity)
            features[f"{component}_mean"] = np.mean(velocity_filtered)
            features[f"{component}_std"] = np.std(velocity_filtered)
            features[f"{component}_range"] = np.ptp(velocity_filtered)
            features[f"{component}_energy"] = np.sum(velocity_filtered**2)

            features[f"{component}_PGV"] = np.max(np.abs(velocity_filtered))
            features[f"{component}_PGA"] = np.max(np.abs(acceleration))
            features[f"{component}_PGD"] = np.max(np.abs(displacement_filtered))

            # RMS
            features[f"{component}_rms"] = np.sqrt(np.mean(velocity_filtered**2))

            # Zero crossings
            features[f"{component}_zero_crossings"] = np.sum(
                np.diff(np.signbit(velocity_filtered))
            )

            # Spectral features
            features[f"{component}_peak_freq"] = f[np.argmax(Pxx)]
            features[f"{component}_spectral_mean"] = np.mean(Pxx)
            features[f"{component}_spectral_std"] = np.std(Pxx)

            # Frequency bands energy
            band_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)]
            for j, (low, high) in enumerate(band_ranges):
                mask = (f >= low) & (f <= high)
                features[f"{component}_band_{low}_{high}_energy"] = np.sum(Pxx[mask])

        # TIER C: Source physics parameters (requiring physical model)
        if feature_type in ["all", "physics"]:
            # ---------- SITE CORRECTION (KAPPA) ----------
            kappa = 0.04

            freqs, Pxx = signal.welch(
                displacement_filtered,
                fs=sampling_rate,
                window="hann",
                nperseg=1024,
                noverlap=512,
                detrend="linear",
                scaling="spectrum",
            )
            displacement_spectrum = np.sqrt(Pxx)

            # Apply κ-correction (site attenuation)
            displ_corr = displacement_spectrum * np.exp(np.pi * kappa * freqs)

            # ---------- RADIATION PATTERN CORRECTION ----------
            if "station_distance" in metadata:
                R_km = metadata["station_distance"]
                src_depth = metadata["metadata"]["source_depth_km"]
                sta_elev = metadata["metadata"]["station_elevation_m"] / 1000.0
                R_km = np.sqrt(R_km**2 + (src_depth + sta_elev) ** 2)

            G = 0.63
            displ_corr_rad = displ_corr / G

            # ---------- GEOMETRICAL SPREADING CORRECTION ----------
            displ_corr_rad_geo = displ_corr_rad * R_km

            # ---------- JOINT BRUNE + Q FIT -------------------------------------
            source_depth_km = metadata["metadata"]["source_depth_km"]
            beta_km_s, rho_g_cm3 = get_velocity_density(source_depth_km)

            fc, Omega0, Q_path, fit_success = fit_brune_model_with_Q(
                freqs, displ_corr_rad_geo, R_km, beta_km_s
            )

            features[f"{component}_corner_freq"] = fc
            features[f"{component}_Omega0"] = Omega0
            features[f"{component}_Q_path"] = Q_path

            # ---------- DERIVED SOURCE PARAMETERS ----------
            beta_m_s = beta_km_s * 1000.0
            rho_kg_m3 = rho_g_cm3 * 1000.0
            mu = 30e9
            k = 0.26

            # Shear modulus (Pa)
            mu = 30e9

            # Calculate the moment (N·m) with correct units and scaling
            M0 = Omega0 * 4 * np.pi * rho_kg_m3 * (beta_m_s**3) * (R_km * 1000) / G
            features[f"{component}_M0"] = M0

            # Moment magnitude (Hanks & Kanamori)
            Mw = (2.0 / 3.0) * np.log10(M0) - 6.06
            features[f"{component}_Mw"] = Mw

            # Source radius (m)
            r_brune = k * beta_m_s / fc
            features[f"{component}_source_radius"] = r_brune

            # Calibrated stress drop (Pa)
            stress_drop = (7.0 / 16.0) * M0 / (r_brune**3)
            features[f"{component}_stress_drop"] = stress_drop

            # Rupture area (m^2)
            rupture_area = np.pi * r_brune**2
            features[f"{component}_rupture_area"] = rupture_area

            # Average slip (m)
            avg_slip = M0 / (mu * rupture_area)
            features[f"{component}_avg_slip"] = avg_slip

    # TIER B: Polarization & site features (no source model)
    if feature_type in ["all", "physics"]:
        # Polarization features
        features["pol_az"] = np.degrees(
            np.arctan2(
                np.std(velocity_filtered_components[2]),
                np.std(velocity_filtered_components[1]),
            )  # 0: Z, 1: N, 2: E
        )  # Azimuth
        features["pol_inc"] = np.degrees(
            np.arctan2(
                np.sqrt(
                    np.std(velocity_filtered_components[1]) ** 2
                    + np.std(velocity_filtered_components[2]) ** 2
                ),
                np.std(velocity_filtered_components[0]),
            )
        )  # Incidence

    # TIER C: Source physics parameters
    if feature_type in ["all", "physics"]:
        # Add average calibrated parameters across components
        # For stress drop, use geometric mean (log-normal distribution)
        stress_drops = [
            features.get("Z_stress_drop", np.nan),
            features.get("N_stress_drop", np.nan),
            features.get("E_stress_drop", np.nan),
        ]
        if all(~np.isnan(stress_drops)):
            features["avg_stress_drop"] = np.exp(np.nanmean(np.log(stress_drops)))

        # For moment, use median (more robust to outliers)
        moments = [
            features.get("Z_M0", np.nan),
            features.get("N_M0", np.nan),
            features.get("E_M0", np.nan),
        ]
        if all(~np.isnan(moments)):
            features["avg_M0"] = np.nanmedian(moments)
            # Add corresponding Mw
            features["avg_Mw"] = (2.0 / 3.0) * np.log10(features["avg_M0"]) - 6.06

    return features


def prepare_multi_station_dataset(aftershocks_df, feature_type="all"):
    """
    Prepare dataset using multiple stations for each event with improved
    station selection, distribution statistics, and reduced redundancy.

    Args:
        aftershocks_df: DataFrame with aftershock data
        feature_type: Type of features to extract

    Returns:
        X: Feature DataFrame
        y: Target DataFrame
    """
    print("Preparing enhanced multi-station dataset...")

    # Step 1: Extract features from all waveforms
    print("Extracting features from waveforms...")
    features_list = []
    errors = 0

    for idx, row in tqdm(aftershocks_df.iterrows(), total=len(aftershocks_df)):
        try:
            # Extract features with metadata
            features = extract_waveform_features(
                row["waveform"], feature_type=feature_type, metadata=row
            )

            # Add necessary metadata needed for aggregation later
            features["station_key"] = row.get("station_key", "")
            features["event_id"] = row.get("event_id", "")
            features["origin_time"] = row["origin_time"]
            features["event_date"] = row["event_date"]
            features["relative_x"] = row["relative_x"]
            features["relative_y"] = row["relative_y"]
            features["relative_z"] = row["relative_z"]

            # Add station_distance ONLY for ranking purposes (prefixed to mark it)
            features["_ranking_station_distance"] = row["station_distance"]

            features_list.append(features)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            errors += 1

    print(f"Successfully processed {len(features_list)} waveforms with {errors} errors")

    all_features_df = pd.DataFrame(features_list)

    # Step 2: Aggregate features across stations for each event
    print("Aggregating features across stations for each event...")

    # Group by event_id
    event_groups = all_features_df.groupby("event_id")

    # List of columns that shouldn't be aggregated
    skip_columns = [
        "origin_time",
        "event_date",
        "relative_x",
        "relative_y",
        "relative_z",
        "event_id",
        "station_key",
    ]

    # List of columns that should not be aggregated to prevent data leakage
    leakage_columns = [
        col
        for col in all_features_df.columns
        if "selection_score" in col
        or "epicentral_distance" in col
        or "station_lat" in col
        or "station_lon" in col
        or "station_elev" in col
        or col.startswith("_ranking_")  # Exclude ranking metrics
    ]

    skip_columns.extend(leakage_columns)

    # List of numeric feature columns to aggregate
    numeric_columns = [
        col
        for col in all_features_df.columns
        if col not in skip_columns
        and pd.api.types.is_numeric_dtype(all_features_df[col])
    ]

    # Create empty dataframe for aggregated features
    aggregated_features = []

    for event_id, group in event_groups:
        # Start with event metadata (same for all stations)
        event_data = {
            "event_id": event_id,
            "origin_time": group["origin_time"].iloc[0],
            "event_date": group["event_date"].iloc[0],
            "relative_x": group["relative_x"].iloc[0],
            "relative_y": group["relative_y"].iloc[0],
            "relative_z": group["relative_z"].iloc[0],
        }

        # For each feature, calculate various statistics across stations
        for feature in numeric_columns:
            values = group[feature].values
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                raise ValueError(
                    f"No valid values found for feature {feature} in event {event_id}"
                )

            # Basic statistics
            event_data[f"{feature}_mean"] = np.mean(valid_values)
            event_data[f"{feature}_median"] = np.median(valid_values)
            # Use NaN for std when there's only one value
            event_data[f"{feature}_std"] = (
                np.std(valid_values) if len(valid_values) > 1 else np.nan
            )
            event_data[f"{feature}_min"] = np.min(valid_values)
            event_data[f"{feature}_max"] = np.max(valid_values)
            event_data[f"{feature}_range"] = np.ptp(valid_values)

            # Add direct quantile values for better distribution representation
            if len(valid_values) >= 4:  # Need enough points for meaningful quartiles
                q25, q75 = np.percentile(valid_values, [25, 75])
                event_data[f"{feature}_q25"] = q25
                event_data[f"{feature}_q75"] = q75

        # Select representative stations using ONLY station_distance
        if "_ranking_station_distance" in group.columns and not all(
            np.isnan(group["_ranking_station_distance"])
        ):
            # Sort by station_distance (ascending) - closest stations first
            quality_sorted = group.sort_values("_ranking_station_distance")
        else:
            # Fallback to original order if station_distance isn't available
            print(
                f"Warning: No station_distance available for event {event_id}. Using original order."
            )
            quality_sorted = group

        # Add features from best, second best, and third best stations
        station_indices = quality_sorted.index.tolist()

        # Best station (sorted by distance)
        if len(station_indices) > 0:
            best_station = quality_sorted.iloc[0]
            for feature in numeric_columns:
                event_data[f"best_{feature}"] = best_station[feature]

        # Second best station
        if len(station_indices) > 1:
            second_station = quality_sorted.iloc[1]
            for feature in numeric_columns:
                event_data[f"second_{feature}"] = second_station[feature]

        # Third best station
        if len(station_indices) > 2:
            third_station = quality_sorted.iloc[2]
            for feature in numeric_columns:
                event_data[f"third_{feature}"] = third_station[feature]

        aggregated_features.append(event_data)

    # Convert to DataFrame
    merged_df = pd.DataFrame(aggregated_features)
    print(
        f"Created aggregated dataset with {len(merged_df)} events and {len(merged_df.columns)} features"
    )

    # Define features and targets
    drop_columns = [
        "origin_time",
        "relative_x",
        "relative_y",
        "relative_z",
        "event_date",
    ]

    # Keep event_id for GroupKFold
    X = merged_df.drop(drop_columns, axis=1)
    y = merged_df[["relative_x", "relative_y", "relative_z", "event_id"]]

    return X, y
