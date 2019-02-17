import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder as le

n = input("Enter the dataset filename: ")
df = pd.read_csv(n)

s = int(input("Is there a serial no. column, if yes then enter the column index else 0"))
if s != 0:
    df.drop(df.columns[s-1], axis=1, inplace=True)

s = int(input("Is there a date column, if yes then enter the column index else 0"))
if s != 0:
    df.drop(df.columns[s-1], axis=1, inplace=True)

no_of_rows, no_of_col = df.shape

s = input("enter the NaN character")

if int(s) != 0:
    for index, row in df.iterrows():
        for i in range(no_of_col):
            if row[i] == s:
                df.drop(index, axis='rows', inplace=True).reset_index(drop=True)
                break


no_of_rows, no_of_col = df.shape


def clean_float(st):
    st = str(st).replace(',','')
    st = float(st)
    return st


for column in df:
    if str(type(df[column][0])).replace('<class ', '').replace('>','') == '\'str\'':
        try:
            df[column] = df[column].apply(lambda i: clean_float(i))
        except ValueError:
            encoder = le()
            encoder.fit(df[column])
            df[column] = encoder.transform(df[column])

p = input("Enter the name of prediction column:")

y = np.array((df[p]))
x = np.array(df.drop(p, axis=1))


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
