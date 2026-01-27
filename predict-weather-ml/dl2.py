import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

global_temp = pd.read_csv("GlobalTemperatures.csv")

def wrangle(df):
    df = df.copy()
    df = df.drop(columns=[
        "LandAverageTemperatureUncertainty",
        "LandMaxTemperatureUncertainty",
        "LandMinTemperatureUncertainty",
        "LandAndOceanAverageTemperatureUncertainty"
    ])

    def converttemp(x):
        return (x * 1.8) + 32

    df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(converttemp)
    df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(converttemp)
    df["LandMinTemperature"] = df["LandMinTemperature"].apply(converttemp)
    df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(converttemp)

    df["dt"] = pd.to_datetime(df["dt"])
    df["Year"] = df["dt"].dt.year
    df = df.drop(columns=["dt"])

    df = df[df["Year"] >= 1850]
    df = df.set_index("Year")
    df = df.dropna()

    return df

global_temp = wrangle(global_temp)

corrMatrix = global_temp.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

target = "LandAndOceanAverageTemperature"
y = global_temp[target]
x = global_temp[["LandAverageTemperature", "LandMaxTemperature", "LandMinTemperature"]]

xtrain, xval, ytrain, yval = train_test_split(
    x, y, test_size=0.25, random_state=42
)

baseline_pred = [ytrain.mean()] * len(yval)
baseline_mse = mean_squared_error(yval, baseline_pred)
print("Baseline MSE:", round(baseline_mse, 5))

forest = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    )
)

forest.fit(xtrain, ytrain)
rf_preds = forest.predict(xval)

errors = abs(rf_preds - yval)
mape = 100 * (errors / yval)
accuracy = 100 - np.mean(mape)

print("Random Forest Model Accuracy:", round(accuracy, 2), "%")
