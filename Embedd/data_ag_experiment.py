import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from Embedd import modeler, dataproc,experiment
import numpy as np
from keras.models import load_model
import numpy as np
import pandas as pd
from Embedd import dataproc,modeler
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image
dataproc.fix_seeds(1)
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)


dataproc.fix_seeds(1)

embedding_model = '/home/piotr/data/test/models/fun_300_EmbeddSource.h5'
categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Country_emb']

###############Train Embeddings 50 baseline ######################################################
#[0.31105740891888395, 0.8560899207335887]
# [0.44659990448325987, 0.8505005834692929]
# epochs = 50
# model_name = f'fun_{epochs}_EmbeddSource_15x15'
#
# X_train, X_test = dataproc.dataload_stage1(categorical,numerical,onehot=False)
#
# X_train, y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train = dataproc.data_tomodel(X_train,categorical,numerical)
# X_test = dataproc.data_tomodel(X_test,categorical,numerical)
#
# model = modeler.exp_model_Emb1DropoutBIG_corr()
# #
# model.fit(X_train,y_train,epochs=epochs,batch_size=128)
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

################### CNN2D + Dense on Embeddings swith ON MinMax ALL#############################################
# import numpy as np
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_15x15.h5'
# epochs = 100
# model_name = f'picture_baseline_{epochs}_Embeding_CNN2DandDense_2'
# batch_size = 1024
# X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)
#
#
# numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
# pixels = [col for col in X_train.columns if col not in numerical_col]
# # X_train = dataproc.swith_merge(X_train, numerical_col)
# X_train_pixels = X_train[pixels]
# X_train_numerical = X_train[numerical_col]
# X_test_pixels = X_test[pixels]
# X_test_numerical = X_test[numerical_col]
#
# X_test_numerical.drop('Salary',axis=1,inplace=True)
# X_train_numerical.drop('Salary',axis=1,inplace=True)
#
# _,y_train = dataproc.split_data(X_train,'Salary')
# _, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
# X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
# X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
# X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)
# # print(X_train_pixels.shape)
# train_zeros = np.zeros((len(X_train),11))
# test_zeros = np.zeros((len(X_test),11))
# X_train_pixels = np.concatenate((X_train_pixels,train_zeros),1)
# X_test_pixels = np.concatenate((X_test_pixels,test_zeros),1)
#
# # [0.4194015270048362, 0.8439899268757176]
# # [0.41646632663532657, 0.8435599778553872]
# # print(X_train_pixels[0])
# # print(X_train_pixels.shape)
# # print(X_train_pixels[0].shape)
#
# X_train_pixels =X_train_pixels.reshape(-1,14,14,1)
# X_test_pixels = X_test_pixels.reshape(-1,14,14,1)
# #
# model = modeler.model_CNN_Dense(CNN_shape=(14,14,1),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)

#########################VGG16 + Dense Numerical || minmax (0,255)#################################3
# from keras.preprocessing.image import ImageDataGenerator
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource.h5'
# epochs = 200
# model_name = f'picture_baseline_{epochs}_Embeding_CNNandDense_7'
# batch_size = 32
# X_train,X_test = experiment.dataload_minmaxall(categorical,numerical,embedding_model,weights)
#
# numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
# pixels = [col for col in X_train.columns if col not in numerical_col]
# X_train = dataproc.swith_merge(X_train, numerical_col)
# X_train_pixels = X_train[pixels]
# X_train_numerical = X_train[numerical_col]
# X_test_pixels = X_test[pixels]
# X_test_numerical = X_test[numerical_col]
#
# X_train_numerical,y_train = dataproc.split_data(X_train_numerical,'Salary')
# X_test_numerical, y_test = dataproc.split_data(X_test_numerical,'Salary')
#
# X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
# X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
# X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
# X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)
#
# sh = 50
# count_zeros = sh*sh-49#2500-49
# X_train_pixels = experiment.make_vgg_pic(X_train_pixels, count_zeros, sh, sh)
# X_test_pixels = experiment.make_vgg_pic(X_test_pixels, count_zeros, sh, sh)
#
#
# transform = ImageDataGenerator(
#     rotation_range=20
#     ,zoom_range=0.2
#     ,width_shift_range=0.2
#     ,height_shift_range=0.2
#     ,shear_range=0.2
#     ,horizontal_flip=True
#     ,vertical_flip=True
# )
# itr = imageGen = transform.flow(X_train_pixels,batch_size=65122)
# # print(X_train_pixels.shape)
# # print(imageGen)
# X_train_pixels = itr.next()
# # model = modeler.model_VGG16_Dense(CNN_shape=(sh,sh,3),Dense_shape=(6,))
# model = modeler.model_CNN_Dense_antyoverfeet(CNN_shape=(sh,sh,3),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# # model.fit_generator(
# #     transform.flow([imageGen,X_train_numerical],y_train,batch_size=1024)
# #     ,steps_per_epoch=len(X_train_pixels)//1024
# #     ,epochs=epochs
# # )
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)
##################################################################################################################
# check CNN1D ??????
# check data agum on simpler CNN model
# vGG6 freez layer - not train?
# higher zoom range ? tune in ImageDataGEn hyperparamiters ?
# check if adding corelation numbers into model would work ???
# KFold / Stratified KFold




################### CNN2D + Dense on Embeddings + multiply categorical embeddings #############################################
# pix_shape = 49
# epochs = 100
# model_name = f'picture_baseline_{epochs}_Embeding_CNN2DandDense_multiply'
# batch_size = 1024
# X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)
# #
# #
# numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
# pixels = [col for col in X_train.columns if col not in numerical_col]
#
# # print(pixels)
#
# # # X_train = dataproc.swith_merge(X_train, numerical_col)
# X_train_pixels = X_train[pixels]
# X_train_numerical = X_train[numerical_col]
# X_test_pixels = X_test[pixels]
# X_test_numerical = X_test[numerical_col]
# #
# X_test_numerical.drop('Salary',axis=1,inplace=True)
# X_train_numerical.drop('Salary',axis=1,inplace=True)
# #
# _,y_train = dataproc.split_data(X_train,'Salary')
# _, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train_pixels = dataproc.duplic_col(X_train_pixels, pix_shape)
# X_test_pixels = dataproc.duplic_col(X_test_pixels,pix_shape)
#
# X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
# X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
# X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
# X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)
#
# #
# X_train_pixels =X_train_pixels.reshape(-1,pix_shape,pix_shape,1)
# X_test_pixels = X_test_pixels.reshape(-1,pix_shape,pix_shape,1)
# # #
# model = modeler.model_CNN_Dense(CNN_shape=(pix_shape,pix_shape,1),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)



################### VGG16 + Dense on Embeddings + multiply categorical embeddings #############################################
from keras.applications.vgg16 import  preprocess_input
from keras.preprocessing.image import ImageDataGenerator
pix_shape = 49
epochs = 100
model_name = f'picture_baseline_{epochs}_Embeding_VGG16andDense_multiply'
batch_size = 1024
X_train,X_test = experiment.dataload_minmaxall(categorical,numerical,embedding_model,weights)
#
#
numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
pixels = [col for col in X_train.columns if col not in numerical_col]

X_train_pixels = X_train[pixels]
X_train_numerical = X_train[numerical_col]
X_test_pixels = X_test[pixels]
X_test_numerical = X_test[numerical_col]
#
X_test_numerical.drop('Salary',axis=1,inplace=True)
X_train_numerical.drop('Salary',axis=1,inplace=True)
#
_,y_train = dataproc.split_data(X_train,'Salary')
_, y_test = dataproc.split_data(X_test,'Salary')

X_train_pixels = dataproc.duplic_col(X_train_pixels, pix_shape)
X_test_pixels = dataproc.duplic_col(X_test_pixels,pix_shape)

# X_train_pixels = X_train_pixels/255
# X_test_pixels = X_test_pixels/255

# X_train_pixels = X_train_pixels.iloc[:2,:2]
# X_test_pixels = X_test_pixels.iloc[:2,:2]

X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)

X_train_pixels =X_train_pixels.reshape(-1,pix_shape,pix_shape,1)
X_test_pixels = X_test_pixels.reshape(-1,pix_shape,pix_shape,1)
experiment.img_save(X_train_pixels,1,10) # to make it work numpy array needs to be INT only - check how to convert to int or perform MinMAx as int
# than save as pictures oth train and test
# than run VGG16 on saved using Image Data Gen with rotation and so on

X_train_pixels = np.tile(A=X_train_pixels, reps=[1, 3])
X_test_pixels = np.tile(A=X_test_pixels, reps=[1, 3])
# X_train_pixels = preprocess_input(X_train_pixels)

# model = modeler.model_VGG16_Dense(CNN_shape=(pix_shape,pix_shape,3),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)


# transform = ImageDataGenerator(
#     rotation_range=20
#     ,zoom_range=0.2
#     ,width_shift_range=0.2
#     ,height_shift_range=0.2
#     ,shear_range=0.2
#     ,horizontal_flip=True
#     ,vertical_flip=True
# )
# itr = imageGen = transform.flow(X_train_pixels,batch_size=32561) #65122
#
# X_train_pixels = itr.next()
# model = modeler.model_VGG16_Dense(CNN_shape=(pix_shape,pix_shape,3),Dense_shape=(6,))
# # model = modeler.model_CNN_Dense_antyoverfeet(CNN_shape=(sh,sh,3),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# # model.fit_generator(
# #     transform.flow([imageGen,X_train_numerical],y_train,batch_size=1024)
# #     ,steps_per_epoch=len(X_train_pixels)//1024
# #     ,epochs=epochs
# # )
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)
















