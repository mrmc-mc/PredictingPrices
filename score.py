import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from binance import coinData
# get crypto data from binance api

#Load and prep data 
data = pd.DataFrame(coinData)
# X = data[['open', 'high', 'low', 'volume']]
X = data[[1, 2, 3, 5]]
Y = data[4]

#Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split into train/test sets
kfold = KFold(n_splits=10)

for train_index, test_index in kfold.split(X):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    #Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #Make predictions
    predictions = model.predict(X_test)

    #Evaluate the performance of the model
    r2 = r2_score(y_test, predictions)
    print(f"R2 score: {r2}")
