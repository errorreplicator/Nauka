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

def split_data(df, column_name):

    y_df = df[column_name]
    X_df = df.drop(column_name, axis=1)
    return X_df,y_df

def remove_data(df, column_name):

    del df[column_name]
    return df

def labelencoder(df, column_list):

    encoder = LabelEncoder()
    for x in column_list:
        df[x] = encoder.fit_transform(df.loc[:,x])
    return df

def minmax_column(df, column_list):
    scaler = MinMaxScaler()
    scaler.fit(df[column_list])
    df[column_list] = scaler.transform(df[column_list])
    return df

def minmax_row(df, columns_list):
    df_numpy = df[columns_list].to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_numpy.T)
    df_rescale = scaler.transform(df_numpy.T).T
    df_rescale = pd.DataFrame(df_rescale, columns=columns_list, index=range(df.shape[0]))
    df[columns_list] = df_rescale
    return df

def to_numpy_data(df, column_list):
    return np.array(df[column_list])


def swith_merge(df, numerical):
    numerical_rev = numerical[::-1]
    df2 = df.copy()
    df2[numerical] = df2[numerical_rev]
    df_return = df.append(df2).reset_index()
    df_return.drop('index',axis=1,inplace=True)
    return df_return

def data_tomodel(df, categorical, numerical):
    X_train = {col: to_numpy_data(df, [col]) for col in categorical}
    X_train['Numerical'] = to_numpy_data(df, numerical)
    return X_train

def data_seq_swithOFF():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    # train = dataproc.swith_merge(train, numerical)
    # test = dataproc.swith_merge(test,numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    X_train = to_numpy_data(X_train, X_train.columns)
    X_test = to_numpy_data(X_test, X_test.columns)

    return X_train,y_train, X_test, y_test


def data_seq_swithON():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    # print(train.head())

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    train = swith_merge(train, numerical)
    test = swith_merge(test,numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    X_train = to_numpy_data(X_train, X_train.columns)
    X_test = to_numpy_data(X_test, X_test.columns)

    return X_train, y_train, X_test, y_test



def data_func_swithOFF():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    # train = dataproc.swith_merge(train, numerical)
    # test = dataproc.swith_merge(test,numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    X_train_dict= {col: to_numpy_data(X_train, [col]) for col in categorical}
    X_train_dict['Numerical'] = to_numpy_data(X_train, numerical)

    X_test_dict = {col: to_numpy_data(X_test, [col]) for col in categorical}
    X_test_dict['Numerical'] = to_numpy_data(X_test, numerical)

    return X_train_dict, y_train, X_test_dict, y_test

def data_func_swithON():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    train = swith_merge(train, numerical)
    # test = swith_merge(test,numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    X_train_dict= {col: to_numpy_data(X_train, [col]) for col in categorical}
    X_train_dict['Numerical'] = to_numpy_data(X_train, numerical)

    X_test_dict = {col: to_numpy_data(X_test, [col]) for col in categorical}
    X_test_dict['Numerical'] = to_numpy_data(X_test, numerical)

    return X_train_dict, y_train, X_test_dict, y_test

def data_func_swithONscaleROW():
    categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])


    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    train = swith_merge(train, numerical)
    # test = swith_merge(test,numerical)
    train = minmax_row(train, numerical)
    test = minmax_row(test, numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    X_train_dict= {col: to_numpy_data(X_train, [col]) for col in categorical}
    X_train_dict['Numerical'] = to_numpy_data(X_train, numerical)

    X_test_dict = {col: to_numpy_data(X_test, [col]) for col in categorical}
    X_test_dict['Numerical'] = to_numpy_data(X_test, numerical)

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








































