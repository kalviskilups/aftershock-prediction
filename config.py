"""
Configuration parameters for the aftershock prediction system.
"""

# Data processing constants
DEFAULT_SAMPLING_RATE = 100.0  # Hz
DEFAULT_WAVEFORM_LENGTH = 14636  # samples
MAINSHOCK_INFO = {
    "origin_time": "2014-04-01T23:46:50.000000Z",
    "latitude": -19.642,
    "longitude": -70.817,
    "depth": 25.0,
}

# Earth constants
EARTH_RADIUS = 6371.0  # km

# Feature extraction parameters
FEATURE_TYPES = {
    "all": "All feature types (signal + physics)",
    "signal": "Basic signal statistics (Tier A)",
    "physics": "Source physics parameters (Tier C)",
}

# Default XGBoost parameters (fine tuned)
DEFAULT_XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.017287387897853453,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.7980103868976611,
    "colsample_bytree": 0.8559520053515827,
    "reg_alpha": 0.231981075221874,
    "reg_lambda": 4.4044780906686425e-07,
    "gamma": 0.44340940726177724,
    "random_state": 42,
}

# Visualization settings
FIGURE_SIZE = (16, 14)
FONT_SIZE = 14

# Validation thresholds
COORDINATE_VALIDATION_THRESHOLD = 0.1  # km (100 meters)
