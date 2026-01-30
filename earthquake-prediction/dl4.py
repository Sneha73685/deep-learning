import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv("database.csv")
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

timestamps = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        timestamps.append(time.mktime(ts.timetuple()))
    except ValueError:
        timestamps.append(np.nan)

data['Timestamp'] = timestamps
data.dropna(inplace=True)
data = data.drop(['Date', 'Time'], axis=1)

# World map visualization
from mpl_toolkits.basemap import Basemap

m = Basemap(
    projection='mill',
    llcrnrlat=-80,
    urcrnrlat=80,
    llcrnrlon=-180,
    urcrnrlon=180,
    lat_ts=20,
    resolution='c'
)

longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
x, y = m(longitudes, latitudes)

plt.figure(figsize=(12, 10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize=2, color='blue')
m.drawcoastlines()
m.fillcontinents(color='coral', lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()

# Prepare ML data
X = data[['Timestamp', 'Latitude', 'Longitude']]
y = data[['Magnitude', 'Depth']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Deep Learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

def create_model(neurons=16, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2, activation='linear'))  # regression output

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

# -----------------------------
# Grid Search
# -----------------------------
from sklearn.model_selection import GridSearchCV

model = KerasRegressor(model=create_model, verbose=0)

param_grid = {
    "model__neurons": [16],
    "model__activation": ["relu", "sigmoid"],
    "model__optimizer": ["adam", "sgd"],
    "batch_size": [10],
    "epochs": [10]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

grid_result = grid.fit(X_train, y_train)

print("Best score:", grid_result.best_score_)
print("Best params:", grid_result.best_params_)

# Train final model
final_model = create_model(
    neurons=16,
    activation='relu',
    optimizer='adam'
)

final_model.fit(
    X_train,
    y_train,
    batch_size=10,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluation
loss, mae = final_model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")