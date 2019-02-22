import pickle
from tkinter import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder as le


def clean_float(st):
    st = str(st).replace(',', '')
    st = float(st)
    return st


def predict():
    trained_model = e1.get()
    clf = pickle.load(open(trained_model, 'rb'))

    data_name = e2.get()
    df_test = pd.read_csv(data_name)

    s = int(e4.get())
    if s != 0:
        df_test.drop(df_test.columns[s-1], axis=1, inplace=True)

    s = int(e5.get())
    if s != 0:
        df_test.drop(df_test.columns[s-1], axis=1, inplace=True)

    no_of_rows, no_of_col = df_test.shape

    s = e6.get()

    if s != '0':
        for column in df_test:
            df_test = df_test[~df_test[column].isin([s, 'NaN'])]

    df_test.reset_index(drop=True, inplace=True)
    no_of_rows, no_of_col = df_test.shape


    for column in df_test:
        if str(type(df_test[column][0])).replace('<class ', '').replace('>','') == '\'str\'':
            try:
                df_test[column] = df_test[column].apply(lambda i: clean_float(i))
            except ValueError:
                encoder = le()
                encoder.fit(df_test[column])
                df_test[column] = encoder.transform(df_test[column])

    no_of_rows, no_of_col = df_test.shape

    x = np.array(df_test)
    x = preprocessing.scale(x)
    y = []
    for row in x:
        y.append(clf.predict([row]))
    print(y)
    pred_file = e3.get()
    with open(pred_file, 'w') as f:
        for item in y:
            f.write("%s\n" % item)
    f.close()

root = Tk()
l1 = Label(root, text="Enter Model Name:")
e1 = Entry(root)
l2 = Label(root, text="Enter dataset name:")
e2 = Entry(root)
l4 = Label(root, text="Is there a serial no. column, if yes then enter the column index else 0: ")
e4 = Entry(root)
l5 = Label(root, text="Is there a date column, if yes then enter the column index else 0: ")
e5 = Entry(root)
l6 = Label(root, text="enter the NaN character, if no missing data then enter 0: ")
e6 = Entry(root)
l3 = Label(root, text="Save Predicted Values As:")
e3 = Entry(root)
button1 = Button(root, text="Predict", command=predict)
button2 = Button(root, text="Quit", command=root.destroy)

l1.grid(row=0, column=0)
e1.grid(row=0, column=1)
l2.grid(row=1, column=0)
e2.grid(row=1, column=1)
l4.grid(row=2, column=0)
e4.grid(row=2, column=1)
l5.grid(row=3, column=0)
e5.grid(row=3, column=1)
l6.grid(row=4, column=0)
e6.grid(row=4, column=1)
l3.grid(row=5, column=0)
e3.grid(row=5, column=1)
button1.grid(row=6, column=0)
button2.grid(row=7, column=0)
root.mainloop()
