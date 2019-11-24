import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import random
import tensorflow
def fix_seeds(seed):
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)
    tensorflow.random.set_random_seed(seed)
    tensorflow.random.experimental.Generator.from_seed(seed)


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
    return X_df,y_df.values

def remove_data(df, column_name):

    del df[column_name]
    return df

def labelencoder(df, column_list):

    encoder = LabelEncoder()
    for x in column_list:
        df[x] = encoder.fit_transform(df.loc[:,x])
    return df

def labelencoder_bycopy(df, column_list):

    encoder = LabelEncoder()
    for x in column_list:
        df[f'{x}_encode'] = encoder.fit_transform(df.loc[:,x])
    return df

def minmax_column(df_train,column_list):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[column_list])
    df_train[column_list] = scaler.transform(df_train[column_list])
    return df_train


def minmax_row(df,columns): # columns - target variable or list of columns to be excluded from switch
    columns_list = [col for col in df.columns if col not in columns]
    print(columns_list)
    df_numpy = df[columns_list].to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_numpy.T)
    df_rescale = scaler.transform(df_numpy.T).T
    df_rescale = pd.DataFrame(df_rescale, columns=columns_list, index=range(df.shape[0]))
    df[columns_list] = df_rescale
    return df

def to_numpy_data(df, column_list):
    return np.array(df[column_list])


def swith_merge(df, column_list): #column_list == list of columns to be excluded from switch !! - provide [] if all columns switch
    column_to_swith = [col for col in df.columns if col not in column_list]
    column_rev = column_to_swith[::-1]
    df2 = df.copy()
    df2[column_to_swith] = df2[column_rev]
    df_return = df.append(df2).reset_index()
    df_return.drop('index',axis=1,inplace=True)
    return df_return

def data_tomodel(df, categorical, numerical):
    X_train = {col: to_numpy_data(df, [col]) for col in categorical}
    X_train['Numerical'] = to_numpy_data(df, numerical)
    return X_train

def array2frame(dict_name, dict):
    names = [f'{dict_name}_{x}' for x in range(len(dict[0]))]
    df = pd.DataFrame(dict, columns=names)
    return df


def weights2df(dataFrame,model_path,layers,del_categ=True,normalize=False):
    from keras.models import load_model
    #layers - list of embedding layers to extract embeddings values  weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
    model = load_model(model_path)
    weight_dict = {}
    for layer in layers:
        weight_dict[layer] = model.get_layer(layer).get_weights()

    for key in weight_dict:
        df = array2frame(key, weight_dict[key][0])
        colnm = key[:-4]
        dataFrame = pd.merge(dataFrame, df, left_on=colnm, right_index=True)
        if normalize:
            dataFrame[colnm] += 1 ## Normalize categorical encoding
            dvd = dataFrame[colnm].max() ## as above
            dataFrame[colnm] /= dvd ## as above
        if del_categ == True: # Delete original category column or not?
            dataFrame.drop(colnm, axis=1, inplace=True)
    dataFrame.sort_index(inplace=True)
    return dataFrame

def dataload_stage1(categorical,numerical,onehot=False):
    train, test = read_data()
    train['type'] = 'train'
    test['type'] = 'test'

    train.loc[train['Salary'] == ' <=50K', 'Salary'] = '<=50K'
    train.loc[train['Salary'] == ' >50K', 'Salary'] = '>50K'

    test.loc[test['Salary'] == ' <=50K.', 'Salary'] = '<=50K'
    test.loc[test['Salary'] == ' >50K.', 'Salary'] = '>50K'
    # test = test.append(train.iloc[1]).reset_index() # spy from train to test dataset to check embeddings reset_index creates 'index' column !!!!!!!!!!!!!!!!
    big_df = train.append(test)
    big_df = remove_data(big_df, 'Id')
    big_df = labelencoder(big_df, categorical)
    big_df = labelencoder(big_df, ['Salary'])
    big_df = minmax_column(big_df, numerical)
    if onehot:
        for col_name in categorical:
            big_df = pd.concat([big_df, pd.get_dummies(big_df[col_name], prefix=col_name, dummy_na=False)], axis=1).drop([col_name],axis=1)
    X_train = big_df.loc[big_df['type'] == 'train']
    X_test = big_df.loc[big_df['type'] == 'test']
    X_train.drop('type', axis=1, inplace=True)
    X_test.drop('type', axis=1, inplace=True)

    return X_train, X_test
    # return big_df
def dataload_minmaxall(categorical,embedding_model,weights):
    train, test = read_data()
    train['type'] = 'train'
    test['type'] = 'test'

    train.loc[train['Salary'] == ' <=50K', 'Salary'] = '<=50K'
    train.loc[train['Salary'] == ' >50K', 'Salary'] = '>50K'

    test.loc[test['Salary'] == ' <=50K.', 'Salary'] = '<=50K'
    test.loc[test['Salary'] == ' >50K.', 'Salary'] = '>50K'
    # test = test.append(train.iloc[1]).reset_index() # spy from train to test dataset to check embeddings reset_index creates 'index' column !!!!!!!!!!!!!!!!
    big_df = train.append(test)
    big_df = remove_data(big_df, 'Id')
    big_df = labelencoder(big_df, categorical)
    big_df = labelencoder(big_df, ['Salary'])
    big_df = weights2df(big_df, embedding_model, weights, del_categ=True, normalize=False)
    minmax_columns = [col for col in big_df.columns if col not in ['type','Salary']]
    big_df = minmax_column(big_df, minmax_columns)
    X_train = big_df.loc[big_df['type'] == 'train']
    X_test = big_df.loc[big_df['type'] == 'test']
    X_train = remove_data(X_train, 'type')
    X_test = remove_data(X_test, 'type')

    return X_train, X_test


def fun_swithOFF(categorical, numerical, to_dict=True):

    train,test = dataload_stage1(categorical,numerical,onehot=False)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    if to_dict == True:
        X_train_dict = {col: to_numpy_data(X_train, col) for col in categorical}
        X_train_dict['Numerical'] = to_numpy_data(X_train, numerical)

        X_test_dict = {col: to_numpy_data(X_test, col) for col in categorical}
        X_test_dict['Numerical'] = to_numpy_data(X_test, numerical)

        return X_train_dict, y_train, X_test_dict, y_test
    else:
        return X_train, y_train, X_test, y_test

def data_seq_swithOFF(categorical,numerical,numpyON = True,expandY = False):
    # categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    # numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train, test = minmax_column(train, test, numerical)
    # test = minmax_column(test, numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')
    if numpyON:
        X_train = to_numpy_data(X_train, X_train.columns)
        X_test = to_numpy_data(X_test, X_test.columns)
    if expandY:
        y_train = y_train.append(y_train)
        y_test = y_test.append(y_test)
    return X_train,y_train.values, X_test, y_test.values

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

    return X_train, y_train.values, X_test, y_test.values

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

def data_func_swithOFF(categorical,numerical,to_dict = True):
    # categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    # numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()
    train = train.drop(axis=0, index=19609)  # delete Holand one row wich breaks encodes as it does not live in test
    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train, test = minmax_column(train,test, numerical)
    # test = minmax_column(test, numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    if to_dict == True:
        X_train_dict = {col: to_numpy_data(X_train, col) for col in categorical}
        X_train_dict['Numerical'] = to_numpy_data(X_train, numerical)

        X_test_dict = {col: to_numpy_data(X_test, col) for col in categorical}
        X_test_dict['Numerical'] = to_numpy_data(X_test, numerical)

        return X_train_dict, y_train.values, X_test_dict, y_test.values
    else:
        return X_train, y_train.values, X_test, y_test.values


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

def dataframe_seq_swithOFF(categorical,numerical):
    # categorical = ['Workclass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
    # numerical = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek']

    train, test = read_data()
    train = train.drop(axis=0, index=19609) # delete Holand one row wich breaks encodes as it does not live in test

    train = remove_data(train, 'Id')
    test = remove_data(test, 'Id')

    train = labelencoder(train, categorical)
    test = labelencoder(test, categorical)
    train = labelencoder(train, ['Salary'])
    test = labelencoder(test, ['Salary'])

    train = minmax_column(train, numerical)
    test = minmax_column(test, numerical)

    X_train, y_train = split_data(train, 'Salary')
    X_test, y_test = split_data(test, 'Salary')

    return X_train,y_train, X_test, y_test








































