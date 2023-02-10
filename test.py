#Load libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from binance import coinData, coinDatalimited

# Load data
data = pd.DataFrame(coinData)

#Features and labels
features = data.drop(columns=[4])
labels = data[4]

#Train model
model = RandomForestRegressor()
model.fit(features, labels)

#Prediction
prediction = model.predict([[0,1,2,3,5,6,7,8,9,10,11]])
print(prediction)
