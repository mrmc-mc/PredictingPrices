#Import necessary libraries
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance import coinData
#Get Bitcoin OHLC data

btc_hist = pd.DataFrame(coinData)
btc_hist.head()


#Calculate exponential moving averages
ema12 = btc_hist[4].ewm(span=12, adjust=False).mean()
ema26 = btc_hist[4].ewm(span=26, adjust=False).mean()


#Calculate the MACD
macd = ema12 - ema26


#Plot the MACD
plt.figure(figsize=(15,7))
plt.plot(btc_hist[4], label='Price')
plt.plot(ema12, label='EMA 12')
plt.plot(ema26, label='EMA 26')
plt.plot(macd, label='MACD')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Bitcoin Moving Average Convergence Divergence', fontsize=18)
plt.legend(loc="upper right", fontsize=14)
plt.show()
