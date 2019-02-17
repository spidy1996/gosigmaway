import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNR


data = pd.read_csv("data.csv")
train_data = data[:490]
test_data = data[490:]
print(data.head())

y_train = np.array((train_data.ix[train_data['Chance of Admit']]))
#x_train = np.array(train_data.drop(['Chance of Admi


"""
y_test = test_data[0]['Chance of Admit']
x_test = test_data[0].drop['Chance of Admit']


KNNR = KNR(5)
KNNR.fit(x_train)

test = np.array(x_test)
test1 = test.reshape(1, -1)

print('original is:', y_test)
print('predicted is:', KNNR.predict(test1))
"""
