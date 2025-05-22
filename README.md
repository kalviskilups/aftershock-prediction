# Seismic Aftershock Prediction using Source Physics Features

A modular Python framework for predicting aftershock locations using machine learning techniques.

## Overview

This framework provides tools for analyzing seismic data and predicting the locations of aftershocks following a mainshock event. It includes modules for:

1. **Data Processing**: Converting raw seismic data from the Iquique dataset to the required format
2. **Feature Extraction**: Extracting time-domain and frequency-domain features from waveforms
3. **Model Training**: Training XGBoost regression models to predict aftershock locations
4. **Results Visualization**: Creating geographic and statistical visualizations of prediction results

The framework consists of two main scripts:
- `process_data.py`: Processes raw seismic data into the required pickle format
- `main.py`: Performs aftershock prediction using the processed data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aftershock_prediction.git
cd aftershock_prediction

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Processing

Before running the aftershock prediction workflow, you need to process the raw seismic data into the required pickle format:

```bash
# Basic usage
python process_data.py --inventory CX.xml --output aftershock_data.pkl

# Advanced usage with optional parameters
python process_data.py --inventory CX.xml \
                      --output aftershock_data.pkl \
                      --top-n 10 \
                      --min-stations 5 \
                      --standardize \
                      --target-length 14636
```

### Running Prediction Workflow

```bash
# Basic usage
python main.py --data /path/to/data.pkl --results-dir ./results

# Advanced usage with optional parameters
python main.py --data /path/to/data.pkl \
              --feature-type physics \
              --approach multi_station \
              --validation full \
              --shap \
              --results-dir ./custom_results
```

## Project Structure

```
aftershock_prediction/
├── __init__.py              # Package initialization
├── config.py                # Configuration parameters
├── data/                    # Data handling
│   ├── __init__.py
│   ├── loader.py            # Data loading functions
│   └── preprocessor.py      # Data preprocessing
├── features/                # Feature extraction
│   ├── __init__.py
│   └── extractor.py         # Feature extraction
├── models/                  # Prediction models
│   ├── __init__.py
│   ├── base_predictor.py    # Base predictor class
│   └── xgboost_predictor.py # XGBoost implementation
├── visualization/           # Visualization tools
│   ├── __init__.py
│   ├── geographic.py        # Geographic visualizations
│   └── statistics.py        # Statistical visualizations
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── coordinate_utils.py  # Coordinate conversion
│   └── validation.py        # Data validation
├── main.py                  # Main prediction workflow CLI
├── process_data.py          # Data preprocessing script
├── optimize_hyperparams.py  # Hyperparameter optimization script
└── requirements.txt         # Dependencies list
```