import kagglehub

path = kagglehub.dataset_download("adarshraj321/googcsv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from prophet import Prophet

warnings.filterwarnings("ignore")

# Load data from KaggleHub cache
data = pd.read_csv(f"{path}/GOOG.csv")

# Plot closing price
plt.style.use("fivethirtyeight")
plt.figure(figsize=(16, 8))
plt.title("Google Closing Stock Price")
plt.plot(data["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

# Prepare data for Prophet
data = data[["Date", "Close"]]
data = data.rename(columns={"Date": "ds", "Close": "y"})
data["ds"] = pd.to_datetime(data["ds"])

# Train Prophet model
m = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True
)

m.fit(data)

# Forecast future
future = m.make_future_dataframe(periods=365)
predictions = m.predict(future)

# Plot forecast
m.plot(predictions)
plt.title("Prediction of Google Stock Price")
plt.xlabel("Date")
plt.ylabel("Closing Stock Price")
plt.show()

# Plot components
m.plot_components(predictions)
plt.show()