import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm
from sklearn import preprocessing

n = input("Enter the dataset filename: ")
df = pd.read_csv(n)


df.drop('Serial No.', axis=1, inplace=True)
print(df.columns)
y = np.array((df['Chance of Admit ']))
x = np.array(df.drop('Chance of Admit ', axis=1))


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3)

x_train = preprocessing.scale(x_train)
x_test = preprocessing.scale(x_test)
y_train = preprocessing.scale(y_train)
y_test = preprocessing.scale(y_test)


print("Which algo would you like to use")
print("1. SVM")
print("2. KNN")
c = int(input("Enter your choice: "))

if c == 1:
    clf1 = svm.SVR(kernel='poly', degree=7)
    clf1.fit(x_train, y_train)
    print('Squared error Accuracy for SVR: ', clf1.score(x_test, y_test))
else:
    clf = KNR(10)
    clf.fit(x_train, y_train)
    print('Squared error Accuracy for KNN: ', clf.score(x_test, y_test))

