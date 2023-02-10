import requests
import datetime as dt

endTime = int(dt.datetime.now().timestamp() * 1000)
# Get crypto data from Binance API
url = f"https://api4.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&endTime={endTime}"
url_limited = f"https://api4.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=500&endTime={endTime}"
coinData = requests.api.get(url).json()
coinDatalimited = requests.api.get(url_limited).json()