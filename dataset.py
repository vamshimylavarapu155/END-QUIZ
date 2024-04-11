import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("AirPassengers.csv")

# Preprocessing
# Handle missing values if any
data.dropna(inplace=True)

# Normalization
scaler = MinMaxScaler()
data["Passengers"] = scaler.fit_transform(data["Passengers"].values.reshape(-1, 1))

# Splitting into training and test sets
train_size = int(len(data) * 0.8)  # 80% for training
test_size = len(data) - train_size
train, test = data.iloc[:train_size], data.iloc[train_size:]

print("Training set size:", len(train))
print("Test set size:", len(test))
