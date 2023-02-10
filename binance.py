import requests
import datetime as dt

endTime = int(dt.datetime.now().timestamp() * 1000)
print(endTime)
# Get crypto data from Binance API
url = f"https://api4.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&endTime={endTime}"
url_limited = f"https://api4.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=500&endTime={endTime}"
coinData = requests.api.get(url).json()
coinDatalimited = requests.api.get(url_limited).json()