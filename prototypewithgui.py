import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder as le
from tkinter import *
import pickle


def clean_float(st):
    st = str(st).replace(',','')
    st = float(st)
    return st


def run():
    df = pd.read_csv(e1.get())
    s = int(e2.get())
    if s != 0:
        df.drop(df.columns[s-1], axis=1, inplace=True)

    s = int(e3.get())
    if s != 0:
        df.drop(df.columns[s-1], axis=1, inplace=True)

    no_of_rows, no_of_col = df.shape

    s = e4.get()

    if s != '0':
        for column in df:
            df = df[~df[column].isin([s, 'NaN'])]

    df.reset_index(drop=True, inplace=True)
    no_of_rows, no_of_col = df.shape


    for column in df:
        if str(type(df[column][0])).replace('<class ', '').replace('>','') == '\'str\'':
            try:
                df[column] = df[column].apply(lambda i: clean_float(i))
            except ValueError:
                encoder = le()
                encoder.fit(df[column])
                df[column] = encoder.transform(df[column])

    no_of_rows, no_of_col = df.shape
    p = e5.get()

    y = np.array((df[p]))
    x = np.array(df.drop(p, axis=1))


    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3)

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    c1 = var1.get()
    c2 = var2.get()

    if c1 == 1:
        clf1 = svm.SVR(kernel='rbf')
        clf1.fit(x_train, y_train)
        model_name1 = e6.get()
        with open(model_name1, 'wb') as f:
            pickle.dump(clf1, f)
        result.delete('1.0', END)
        result.insert(END, 'Squared error Accuracy for SVR: ' + str(clf1.score(x_test, y_test))+'\n')

    if c2 == 1:
        clf2 = KNR(3)
        clf2.fit(x_train, y_train)
        model_name2 = e7.get()
        with open(model_name2, 'wb') as f:
            pickle.dump(clf2, f)
        if c1 != 1:
            result.delete('1.0', END)
        result.insert(END, 'Squared error Accuracy for KNN: ' + str(clf2.score(x_test, y_test)))



root = Tk()
l1 = Label(root, text="Enter the dataset filename: ")
e1 = Entry(root)
l2 = Label(root, text="Is there a serial no. column, if yes then enter the column index else 0: ")
e2 = Entry(root)
l3 = Label(root, text="Is there a date column, if yes then enter the column index else 0: ")
e3 = Entry(root)
l4 = Label(root, text="enter the NaN character, if no missing data then enter 0: ")
e4 = Entry(root)
l5 = Label(root, text="Enter the name of prediction column: ")
e5 = Entry(root)
l6 = Label(root, text="Save Model As:")
e6 = Entry(root)
l7 = Label(root, text="Save Model As:")
e7 = Entry(root)
var1 = IntVar()
check1 = Checkbutton(root, text="SVR", variable=var1)
var2 = IntVar()
check2 = Checkbutton(root, text="KNR", variable=var2)
r = Button(root, text="RUN", command=run)
q = Button(root, text="QUIT", command=root.destroy)
result = Text(root, width=40, height=5)


l1.grid(row=0, column=0)
e1.grid(row=0, column=1)

l2.grid(row=1, column=0)
e2.grid(row=1, column=1)

l3.grid(row=2, column=0)
e3.grid(row=2, column=1)

l4.grid(row=3, column=0)
e4.grid(row=3, column=1)

l5.grid(row=4, column=0)
e5.grid(row=4, column=1)

l6.grid(row=5, column=0)
e6.grid(row=5, column=1)
check1.grid(row=5, column=2)


l7.grid(row=6, column=0)
e7.grid(row=6, column=1)
check2.grid(row=6, column=2)

r.grid(row=7, column=1)
q.grid(row=8, column=1)
result.grid(row=9,column=0)

root.mainloop()
