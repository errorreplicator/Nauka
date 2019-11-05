import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def read_data():
    path = Path('/home/piotr/data/test/')
    # files = [x.name for x in path.iterdir() if x.suffix == '.csv']

    columns = ['Age','Workclass','Id','Education','EducationNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss',
           'HoursWeek','Country','Salary']

    train = pd.read_csv(path/'adult-training.csv', names=columns)
    test = pd.read_csv(path/'adult-test.csv', header=0, names=columns)

    return train, test

def data_split(df, column_name):

    y_df = df[column_name]
    X_df = df.drop(column_name, axis=1)
    return X_df,y_df

def data_remove(df, column_name):

    del df[column_name]
    return df

def data_labelencoder(df,column_list):

    encoder = LabelEncoder()
    for x in column_list:
        df[x] = encoder.fit_transform(df.loc[:,x])
    return df

def data_minmax_column(df,column_list):
    scaler = MinMaxScaler()
    scaler.fit(df[column_list])
    df[column_list] = scaler.transform(df[column_list])
    return df

def data_minmax_row(df, columns_list):
    columns_list = df.columns
    df_numpy = df[columns_list].to_numpy()
    df_left = df[df.columns.difference(columns_list)]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_numpy.T)
    df = scaler.transform(df_numpy.T).T
    df = pd.DataFrame(df, columns=columns_list, index=range(df.shape[0]))
    df = df.join(df_left)
    df = df[columns_list]
    return df

def data_to_numpy(df,column_list):
    return np.array(df[column_list])


def swith_merge(df, numerical):
    columns_list = df.columns
    df_solid = df[df.columns.difference(numerical)]
    df_numer = df[numerical]
    numerical_rev = numerical[::-1]
    df_revers = df_numer[numerical_rev]
    df_joined = df_solid.join(df_revers)
    df_return = df.append(df_joined).reset_index()
    df_return.drop('index',axis=1,inplace=True)
    df_return = df_return[columns_list]
    return df_return

def data_tomodel(df, categorical, numerical):
    X_train = {col: data_to_numpy(df,[col]) for col in categorical}
    X_train['Numerical'] = data_to_numpy(df,numerical)
    return X_train

def data_seq_swithOFF():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = data_remove(train, 'Id')
    test = data_remove(test, 'Id')

    train = data_labelencoder(train, categorical)
    test = data_labelencoder(test, categorical)
    train = data_labelencoder(train, ['Salary'])
    test = data_labelencoder(test, ['Salary'])

    train = data_minmax_column(train, numerical)
    test = data_minmax_column(test, numerical)

    # train = dataproc.swith_merge(train, numerical)
    # test = dataproc.swith_merge(test,numerical)

    X_train, y_train = data_split(train, 'Salary')
    X_test, y_test = data_split(test, 'Salary')

    X_train = data_to_numpy(X_train, X_train.columns)
    X_test = data_to_numpy(X_test, X_test.columns)

    return X_train,y_train, X_test, y_test


def data_seq_swithON():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    # print(train.head())

    train = data_remove(train, 'Id')
    test = data_remove(test, 'Id')

    train = data_labelencoder(train, categorical)
    test = data_labelencoder(test, categorical)
    train = data_labelencoder(train, ['Salary'])
    test = data_labelencoder(test, ['Salary'])

    train = data_minmax_column(train, numerical)
    test = data_minmax_column(test, numerical)

    train = swith_merge(train, numerical)
    test = swith_merge(test,numerical)

    X_train, y_train = data_split(train, 'Salary')
    X_test, y_test = data_split(test, 'Salary')

    X_train = data_to_numpy(X_train, X_train.columns)
    X_test = data_to_numpy(X_test, X_test.columns)

    return X_train, y_train, X_test, y_test



def data_func_swithOFF():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = data_remove(train, 'Id')
    test = data_remove(test, 'Id')

    train = data_labelencoder(train, categorical)
    test = data_labelencoder(test, categorical)
    train = data_labelencoder(train, ['Salary'])
    test = data_labelencoder(test, ['Salary'])

    train = data_minmax_column(train, numerical)
    test = data_minmax_column(test, numerical)

    # train = dataproc.swith_merge(train, numerical)
    # test = dataproc.swith_merge(test,numerical)

    X_train, y_train = data_split(train, 'Salary')
    X_test, y_test = data_split(test, 'Salary')

    X_train_dict= {col: data_to_numpy(X_train, [col]) for col in categorical}
    X_train_dict['Numerical'] = data_to_numpy(X_train, numerical)

    X_test_dict = {col: data_to_numpy(X_test, [col]) for col in categorical}
    X_test_dict['Numerical'] = data_to_numpy(X_test, numerical)

    return X_train_dict, y_train, X_test_dict, y_test


def data_func_swithON():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = data_remove(train, 'Id')
    test = data_remove(test, 'Id')

    train = data_labelencoder(train, categorical)
    test = data_labelencoder(test, categorical)
    train = data_labelencoder(train, ['Salary'])
    test = data_labelencoder(test, ['Salary'])

    train = data_minmax_column(train, numerical)
    test = data_minmax_column(test, numerical)

    train = swith_merge(train, numerical)
    # test = swith_merge(test,numerical)

    X_train, y_train = data_split(train, 'Salary')
    X_test, y_test = data_split(test, 'Salary')

    X_train_dict= {col: data_to_numpy(X_train, [col]) for col in categorical}
    X_train_dict['Numerical'] = data_to_numpy(X_train, numerical)

    X_test_dict = {col: data_to_numpy(X_test, [col]) for col in categorical}
    X_test_dict['Numerical'] = data_to_numpy(X_test, numerical)

    return X_train_dict, y_train, X_test_dict, y_test

####################################################################################################################

# def data_categ_numpy(df, categorical, numerical,target_column,target_value):
#     df[target_column] = np.where(df[target_column] == target_value, 1, 0)
#
#     X_df, y_df = data_split(df,target_column)
#     y_df = df.values
#
#     X_df = data_labelencoder(X_df,categorical)
#     X_df = data_minmax_row(X_df,categorical)
#     X_dict = data_tomodel(X_df,categorical,numerical)
#     return X_dict, y_df

def data_categ(categorical, numerical,train,test):
    # train, test = read_data()
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

def data_categ_clean(categorical, numerical,train,test):
    train, test = train,test
    train['Salary'] = np.where(train['Salary'] == ' >50K', 1, 0)
    test['Salary'] = np.where(test['Salary'] == ' >50K.', 1, 0)

    # y_train = train['Salary'].values
    # y_test = test['Salary'].values

    del train['Id']
    # del train['Salary']
    del test['Id']
    # del test['Salary']

    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    for x in categorical:
        train[x] = encoder.fit_transform(train[x])
        test[x] = encoder.fit_transform(test[x])

    scaler.fit(train[numerical])
    train[numerical] = scaler.transform(train[numerical])

    scaler.fit(test[numerical])
    test[numerical] = scaler.transform(test[numerical])

    return train, train

def data_categ_normalized(categorical, numerical,train,test):

    y_train = train['Salary'].values
    y_test = test['Salary'].values

    del train['Salary']
    del test['Salary']

    X_train = {col: np.array(train[col]) for col in categorical}
    X_train['Numerical'] = np.array(train[numerical])

    X_test = {col: np.array(test[col]) for col in categorical}
    X_test['Numerical'] = np.array(test[numerical])

    return X_train, y_train, X_test, y_test








































