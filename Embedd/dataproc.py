import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def read_data():
    path = Path('/home/piotr/data/test/')
    files = [x.name for x in path.iterdir() if x.suffix == '.csv']


    columns = ['Age','Workclass','Id','Education','EducationNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss',
           'HoursWeek','Country','Salary']

    train = pd.read_csv(path/'adult-training.csv', names=columns)
    test = pd.read_csv(path/'adult-test.csv', header=0, names=columns)
    return train, test


def clean_data_categ(categorical, numerical):
    train, test = read_data()
    train['Salary'] = np.where(train['Salary'] == ' >50K', 1, 0)
    test['Salary'] = np.where(test['Salary'] == ' >50K.', 1, 0)

    y_train = train['Salary'].values
    y_test = test['Salary'].values

    del train['Id']
    del train['Salary']
    del test['Id']
    del test['Salary']

    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    for x in categorical:
        train[x] = encoder.fit_transform(train[x])
        test[x] = encoder.fit_transform(test[x])

    scaler.fit(train[numerical])
    train[numerical] = scaler.transform(train[numerical])

    scaler.fit(test[numerical])
    test[numerical] = scaler.transform(test[numerical])


    X_train = {col: np.array(train[col]) for col in categorical}
    X_train['Numerical'] = np.array(train[numerical])

    X_test = {col: np.array(test[col]) for col in categorical}
    X_test['Numerical'] = np.array(test[numerical])

    return X_train, y_train, X_test, y_test















