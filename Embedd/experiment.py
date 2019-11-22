from sklearn.preprocessing import MinMaxScaler
from Embedd import dataproc
from PIL import Image
def minmax_column(df_train,column_list):
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaler.fit(df_train[column_list])
    df_train[column_list] = scaler.transform(df_train[column_list])
    return df_train


def dataload_minmaxall(categorical,embedding_model,weights):
    train, test = dataproc.read_data()
    train['type'] = 'train'
    test['type'] = 'test'

    train.loc[train['Salary'] == ' <=50K', 'Salary'] = '<=50K'
    train.loc[train['Salary'] == ' >50K', 'Salary'] = '>50K'

    test.loc[test['Salary'] == ' <=50K.', 'Salary'] = '<=50K'
    test.loc[test['Salary'] == ' >50K.', 'Salary'] = '>50K'
    # test = test.append(train.iloc[1]).reset_index() # spy from train to test dataset to check embeddings reset_index creates 'index' column !!!!!!!!!!!!!!!!
    big_df = train.append(test)
    big_df = dataproc.remove_data(big_df, 'Id')
    big_df = dataproc.labelencoder(big_df, categorical)
    big_df = dataproc.labelencoder(big_df, ['Salary'])
    big_df = dataproc.weights2df(big_df, embedding_model, weights, del_categ=True, normalize=False)
    minmax_columns = [col for col in big_df.columns if col not in ['type','Salary']]

    big_df = minmax_column(big_df, minmax_columns)
    X_train = big_df.loc[big_df['type'] == 'train']
    X_test = big_df.loc[big_df['type'] == 'test']
    X_train = dataproc.remove_data(X_train, 'type')
    X_test = dataproc.remove_data(X_test, 'type')

    return X_train, X_test

def img_save(numpy_array,ratio=1,stop=None):
    for index,single in enumerate(numpy_array):
        im = Image.fromarray(single)
        im = im.convert('RGB')
        im = im.resize((single.shape[0] * ratio, single.shape[1] * ratio))
        im.save(f'/home/piotr/data/test/img/{index}.png')
        if stop:
            if stop == index:
                break