# Deep Learning Mini-Projects

Two small, runnable experiments:
- **predict-weather-ml/** — Random Forest regression on historical global land/ocean temperatures.
- **time-series-lstm/** — Univariate LSTM forecast on monthly airline passengers.

## Requirements
- Python 3.12
- Packages: pandas, numpy, seaborn, matplotlib, scikit-learn, tensorflow (CPU is fine for these scripts).

## Setup
```bash
python -m venv aiml_env
source aiml_env/bin/activate
pip install -U pip
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```
> On Apple Silicon, use `tensorflow-macos` if the standard wheel fails to install.

## How to Run
From the repo root:
```bash
# Random Forest temperature model
cd predict-weather-ml
python dl2.py

# LSTM time series forecast
cd ../time-series-lstm
python dl1.py
```
Both scripts print metrics to the console and open plots in a GUI window. If running headless, set a non-interactive matplotlib backend (e.g., `matplotlib.use("Agg")`) and save figures instead of `plt.show()`.

## Project Details
### predict-weather-ml
- Script: [predict-weather-ml/dl2.py](predict-weather-ml/dl2.py)
- Data: [predict-weather-ml/GlobalTemperatures.csv](predict-weather-ml/GlobalTemperatures.csv) (Kaggle: Global Land and Ocean Temperature Time Series).
- Steps: clean columns, convert °C→°F, extract year, drop missing rows, plot correlations, train/validate `RandomForestRegressor`, report baseline MSE and model accuracy (1 - MAPE).
- Adjust: tweak `test_size`, forest hyperparameters, or feature set; persist the model if desired.

### time-series-lstm
- Script: [time-series-lstm/dl1.py](time-series-lstm/dl1.py)
- Data: [time-series-lstm/airline-passengers.csv](time-series-lstm/airline-passengers.csv) (monthly totals).
- Steps: scale series with `MinMaxScaler`, build 1-step look-back sequences, fit a small LSTM (4 units, 100 epochs), compute train/test RMSE, plot predictions vs. original series.
- Adjust: increase `look_back`, units, or epochs; try stacked LSTMs, dropout, or multi-step forecasting.

## Notes
- Keep data files in place; scripts expect relative paths.
- Random seeds are set for reproducibility (`numpy.random.seed(7)` in LSTM; `random_state=42/77` in RF pipeline).
- For GPU TensorFlow, install the matching wheel and CUDA toolkit; otherwise CPU runs are sufficient here.
