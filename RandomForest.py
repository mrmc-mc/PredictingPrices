import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from binance import coinData, coinDatalimited

dataset = pd.DataFrame(coinData)

# X = dataset.iloc[: ,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,23]]
# X = dataset.iloc[: ,[0,1,2,3,5,6,7,8,9,10,11]]
X = dataset.iloc[: ,[0,1]]
#X = dataset.iloc[: ,:24].values
y = dataset.iloc[:, 4]
      

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_testTs = X_test.values[:,0]
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =100, criterion = 'gini', max_features= 10, random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# print(X_testTs[3])
# print(y_pred[3])
actual_data = []
for i in X_testTs:
    actual_data.append(dataset.loc[dataset[0]==i][4].values[0])
# print(sc.inverse_transform(X_test))
# print(actual_data)
# print(len(actual_data))
# exit()
# predicted_data = sc.inverse_transform(y_pred)
# print(y_pred)

# plt.plot([x for x in range(len(actual_data))],[float(x) for x in actual_data], color='blue', marker='x', label='Actual X')
# plt.plot([x for x in range(len(y_pred))],[float(x) for x in y_pred], color='red', marker='o', label='Train X')
# plt.legend(loc="upper left")
# plt.show()

real_data = pd.DataFrame(coinData[-2:])
# real_data = real_data.drop(4, axis=1)
real_data = real_data.iloc[: ,[0,1]]
print(real_data)
real_data = sc.transform(real_data)
print(classifier.predict(real_data))