# Deep Learning Projects Collection

A collection of four deep learning and machine learning projects covering time series forecasting, weather prediction, stock market analysis, and earthquake data visualization.

## Projects Overview

1. **time-series-lstm/** — Univariate LSTM forecast on monthly airline passengers
2. **predict-weather-ml/** — Random Forest regression on historical global land/ocean temperatures
3. **stock-prediction-prophet/** — Facebook Prophet model for Google stock price forecasting
4. **earthquake-prediction/** — Earthquake data visualization and analysis using neural networks

## Requirements

- Python 3.12+
- Core packages: pandas, numpy, matplotlib, scikit-learn
- Deep Learning: tensorflow (or tensorflow-macos for Apple Silicon)
- Additional: seaborn, prophet, cartopy, kagglehub

## Setup

```bash
python -m venv aiml_env
source aiml_env/bin/activate
pip install -U pip
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow prophet kagglehub cartopy
```

> **Note for Apple Silicon**: Use `tensorflow-macos` if the standard TensorFlow wheel fails to install.

## How to Run

From the repository root:

```bash
# 1. LSTM time series forecast
cd time-series-lstm
python dl1.py

# 2. Random Forest temperature model
cd ../predict-weather-ml
python dl2.py

# 3. Prophet stock prediction
cd ../stock-prediction-prophet
python dl3.py

# 4. Earthquake visualization
cd ../earthquake-prediction
python dl4.py
```

All scripts print metrics to the console and display plots in a GUI window. For headless environments, set a non-interactive matplotlib backend (e.g., `matplotlib.use("Agg")`) and save figures instead of using `plt.show()`.

## Project Details

### 1. time-series-lstm
- **Script**: [time-series-lstm/dl1.py](time-series-lstm/dl1.py)
- **Data**: [time-series-lstm/airline-passengers.csv](time-series-lstm/airline-passengers.csv) (monthly passenger totals)
- **Model**: LSTM neural network for univariate time series forecasting
- **Steps**:
  - Scale data using `MinMaxScaler` (0-1 range)
  - Create look-back sequences (1-step by default)
  - Train LSTM with 4 units for 100 epochs
  - Compute train/test RMSE
  - Plot predictions vs. actual values
- **Customization**: Adjust `look_back`, LSTM units, epochs, or try stacked LSTMs with dropout for multi-step forecasting
- **Reproducibility**: Fixed random seed (`numpy.random.seed(7)`)

### 2. predict-weather-ml
- **Script**: [predict-weather-ml/dl2.py](predict-weather-ml/dl2.py)
- **Data**: [predict-weather-ml/GlobalTemperatures.csv](predict-weather-ml/GlobalTemperatures.csv) (Kaggle: Global Land and Ocean Temperature Time Series)
- **Model**: Random Forest Regressor for temperature prediction
- **Steps**:
  - Clean dataset (remove uncertainty columns)
  - Convert temperatures from °C to °F
  - Extract year from date column
  - Remove missing values
  - Plot correlation heatmaps
  - Train/validate `RandomForestRegressor`
  - Report baseline MSE and model accuracy (1 - MAPE)
- **Customization**: Modify `test_size`, forest hyperparameters, feature selection, or add model persistence
- **Reproducibility**: Fixed random states (`random_state=42/77`)

### 3. stock-prediction-prophet
- **Script**: [stock-prediction-prophet/dl3.py](stock-prediction-prophet/dl3.py)
- **Data**: Google stock prices (downloaded via KaggleHub)
- **Model**: Facebook Prophet for time series forecasting
- **Steps**:
  - Download Google stock data from Kaggle
  - Visualize historical closing prices
  - Prepare data for Prophet (rename to `ds` and `y` columns)
  - Train Prophet model
  - Generate forecasts
  - Plot predictions with confidence intervals
- **Use Case**: Stock market trend analysis and future price prediction
- **Customization**: Adjust Prophet parameters, add seasonality, or apply to different stocks

### 4. earthquake-prediction
- **Script**: [earthquake-prediction/dl4.py](earthquake-prediction/dl4.py)
- **Data**: [earthquake-prediction/database.csv](earthquake-prediction/database.csv)
- **Model**: Neural network for earthquake data analysis
- **Steps**:
  - Parse date/time into Unix timestamps
  - Extract key features (Latitude, Longitude, Depth, Magnitude)
  - Clean and preprocess data
  - Visualize earthquake locations using Cartopy (geographic projections)
  - Train neural network for earthquake prediction/classification
  - Generate geospatial visualizations
- **Features**: Geographic visualization, temporal analysis, magnitude prediction
- **Dependencies**: Cartopy for mapping capabilities

## Data Files

Each project includes its own dataset:
- `airline-passengers.csv` — Monthly airline passenger counts
- `GlobalTemperatures.csv` — Historical global temperature records
- Google stock data — Auto-downloaded via KaggleHub
- `database.csv` — Earthquake event database

## Notes

- **Data Paths**: Keep data files in their respective project folders; scripts use relative paths
- **Reproducibility**: Random seeds are set where applicable for consistent results
- **GPU Support**: For GPU-accelerated TensorFlow, install CUDA toolkit; CPU is sufficient for these projects
- **Visualization**: All scripts open matplotlib plots in GUI windows by default
- **Dependencies**: Some projects require specific packages (Prophet, Cartopy, KaggleHub) — install as needed

## Future Enhancements

- Add model persistence (save/load trained models)
- Implement cross-validation for better performance metrics
- Expand to multi-step forecasting
- Add hyperparameter tuning
- Create unified evaluation framework
- Add Jupyter notebooks for interactive exploration
