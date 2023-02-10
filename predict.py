import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from binance import coinData
# get crypto data from binance api

#Load and prep data 
df = pd.DataFrame(coinData)
# X = data[['open', 'high', 'low', 'volume']]
X = df[[1, 2, 3, 5]]
y = df[4]
# Acquire data
# df = pd.read_csv("candles.csv")

# # Extract useful features
# X = df[['open', 'high', 'low', 'close']]

# # Define target variable
# y = df['price']


# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Plot prediction results
plt.scatter(y_test, predictions)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Candle Price Prediction")
plt.show()
