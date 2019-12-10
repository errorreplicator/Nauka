from sklearn.preprocessing import MinMaxScaler
from Embedd import dataproc
from PIL import Image
import numpy as np

def minmax_column(df_train,column_list):
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaler.fit(df_train[column_list])
    df_train[column_list] = scaler.transform(df_train[column_list])
    return df_train


def dataload_minmaxall(categorical,numerical,embedding_model,weights):
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
    exclude_cols = ['type','Salary','Sex'] + numerical # columns that are not use for 0,255 minmax normalization
    minmax_columns = [col for col in big_df.columns if col not in exclude_cols]

    big_df = minmax_column(big_df, minmax_columns) # for 0,255 norm
    big_df = dataproc.minmax_column(big_df,numerical) # for 0,1 norm
    # big_df = dataproc.swith_merge(big_df,numerical+exclude_cols) # DATA SWITH ##################################
    X_train = big_df.loc[big_df['type'] == 'train']
    X_test = big_df.loc[big_df['type'] == 'test']
    X_train = dataproc.remove_data(X_train, 'type')
    X_test = dataproc.remove_data(X_test, 'type')

    return X_train, X_test

def make_vgg_pic(numpy_array, howmany_z, a, b):
    # from keras.applications.vgg16 import  preprocess_input
    # from keras.preprocessing.image import img_to_array
    # from keras.preprocessing.image import save_img
    # zeros = np.zeros((numpy_array.shape[0], howmany_z)) # fill in with zeros to make picture
    fill = np.full((numpy_array.shape[0],howmany_z),255.)
    image = np.concatenate((numpy_array, fill),1)
    image = image.reshape(-1,a,b,1)
    image = np.tile(A=image, reps=[1, 3])
    # image = np.repeat(image, 3, -1)
    # print(image[0])
    # imgs = []
    # for im in image:
    #     img = img_to_array(im)
    #     img = np.repeat(img,3,-1)
    #     # img = np.expand_dims(img,axis=0)
    #     # img = preprocess_input(img)
    #     # save_img('/home/piotr/Pictures/xcx.png',img[0])
    #     imgs.append(img)
    # image_array = np.concatenate(imgs,axis=0)
    # img_save(image,10)
    # image_array = np.array(image_array)
    # image_array = np.repeat(image_array,3,-1)#####????????
    # image = preprocess_input(image)
    # print(image_array[0].shape)
    return image/255.

def img_save(numpy_array,y,catalog,ratio=1,stop=None):
    for index,single in enumerate(numpy_array):
        # print(single.astype(np.uint8))
        if index%1000==0:
            print(index)
        im = Image.fromarray(single.astype(np.uint8))
        im = im.convert('RGB')
        im = im.resize((single.shape[0] * ratio, single.shape[1] * ratio))
        folder_flag = '0'
        if y[index] == 0:
            folder_flag = '0'
        else:
            folder_flag = '1'
        im.save(f'/home/piotr/data/test/img/{catalog}/{folder_flag}/{index}.png')

        if stop:
            if stop == index:
                break
























