import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("adarshraj321/googcsv")
print(os.listdir(path))

data = pd.read_csv(f"{path}/goog.csv")