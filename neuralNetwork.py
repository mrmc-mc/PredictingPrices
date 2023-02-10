import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import datetime as dt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from binance import coinData, coinDatalimited


coin = "BTC-USD"

# st = dt.datetime(2022,1,1)
# et = dt.datetime.now()


data = pd.DataFrame(coinData)

# Prepare Data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[4].values.reshape(-1,1))


prediction_days = 60
future_days = 10
x_train , y_train = [], []

for x in range(prediction_days, len(scaled_data)
               #- future_days
               ):
    
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x #+ future_days
                               , 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create Neural Network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=30, batch_size=32)


# Testing

test_data = pd.DataFrame(coinDatalimited)

actual_prices = test_data[4].values

total_dataset = pd.concat((data[4], test_data[4]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])


x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)


# print(len(actual_prices))
# print(prediction_prices[:,0])
# exit()
# plot

# plt.plot([float(x) for x in actual_prices], color="black", label="Actual Prices")
# plt.plot(prediction_prices[:,0], color="red", label="Predicted Prices")
# plt.title(f"{coin} price prediction!")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend(loc="upper left")
# plt.show()


real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs)+1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(prediction)


