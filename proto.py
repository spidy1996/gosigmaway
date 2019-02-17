import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
'''
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm
from sklearn import preprocessing

'''


df = pd.read_csv('Book1.csv')


s = int(input("Is there a date column, if yes then enter the column index else 0"))
if s != 0:
    df.drop(df.columns[s-1], axis=1, inplace=True)

no_of_rows, no_of_col = df.shape

s = input("enter the NaN character")

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

print(df.head)
