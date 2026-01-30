import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import warnings

warnings.filterwarnings("ignore")

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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.scatter(
    data['Longitude'],
    data['Latitude'],
    s=5,
    color='red',
    alpha=0.6,
    transform=ccrs.PlateCarree()
)
plt.title("Global Earthquake Distribution")
plt.show()

X = data[['Timestamp', 'Latitude', 'Longitude']]
y = data[['Magnitude', 'Depth']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

def create_model(neurons=16, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

from sklearn.model_selection import GridSearchCV

model = KerasRegressor(model=create_model, verbose=0)

param_grid = {
    "model__neurons": [16],
    "model__activation": ["relu", "sigmoid"],
    "model__optimizer": ["adam", "sgd"],
    "batch_size": [10],
    "epochs": [10]
}

grid = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)

final_model = create_model(neurons=16, activation='relu', optimizer='adam')

final_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=10,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, mae = final_model.evaluate(X_test, y_test)
print(loss, mae)