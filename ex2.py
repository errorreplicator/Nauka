from Embedd import dataproc, modeler
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)
dataproc.fix_seeds(1)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
# weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Country_emb']


###################SEQ Embedding to DF ###########################################
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
# epochs = 100
# model_name = f'seq_{epochs}_Embeding_picture_baseline_2'
# train, test = dataproc.dataload_stage1(categorical,numerical)
#
# # print(train.head())
# X_train = dataproc.weights2df(train,embedding_model,weights,del_categ=True,normalize=False)
# X_test = dataproc.weights2df(test,embedding_model,weights,del_categ=True,normalize=False)
#
# X_train,y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
#
# model = modeler.get_model_Seq((73,))
# model.fit(X_train,y_train,batch_size=1024,epochs=epochs)
# modeler.evaluateSeqModel(X_test,y_test,model,model_name)
###################FUN Embedding to DF ###########################################
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
# epochs = 200
# model_name = f'picture_baseline_{epochs}_Embeding_fun'
# train, test = dataproc.dataload_stage1(categorical,numerical,onehot=False)
#
# # print(train.head())
# X_train = dataproc.weights2df(train,embedding_model,weights,del_categ=True,normalize=False)
# X_test = dataproc.weights2df(test,embedding_model,weights,del_categ=True,normalize=False)
# non_swith = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek','Salary','Sex']
# pic_col = [col for col in X_train.columns if col not in non_swith]
#
# X_train = dataproc.swith_merge(X_train,non_swith)
#
# # print(len(pic_col))# 7 + 67
# numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek','Sex']
# X_train,y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
# #
# # X_train = dataproc.data_tomodel(X_train,pic_col,numerical)
# # X_test = dataproc.data_tomodel(X_test,pic_col,numerical)
# X_train_dict = {}
# X_test_dict = {}
#
#
# X_train_dict['Categorical'] = dataproc.to_numpy_data(X_train, pic_col)
# X_train_dict['Numerical'] = dataproc.to_numpy_data(X_train, numerical)
#
# X_test_dict['Categorical'] = dataproc.to_numpy_data(X_test, pic_col)
# X_test_dict['Numerical'] = dataproc.to_numpy_data(X_test, numerical)
# #
# # print(X_train)
# model = modeler.tester()
# model.fit(X_train_dict,y_train,batch_size=1024,epochs=epochs)
# modeler.evaluateFunModel(X_test_dict,y_test,model,model_name)

###################CNN Embedding to DF ###########################################
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
# epochs = 100
# model_name = f'picture_baseline_{epochs}_Embeding_CNN1D_2'
#
# X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)
#
# non_swith = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek','Salary','Sex']
# pic_col = [col for col in X_train.columns if col not in non_swith]
#
# X_train = dataproc.swith_merge(X_train,non_swith)
# #
# X_train,y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
# #
# #
# X_train = dataproc.to_numpy_data(X_train,X_train.columns)
# X_test = dataproc.to_numpy_data(X_test,X_test.columns)
# X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
# #
# model = modeler.model_Fun_CNN1((X_train.shape[1],1))
# model.fit(X_train,y_train,batch_size=1024,epochs=epochs) # ADAM was better [0.3315874114509048, 0.8546772310953601]
# modeler.evaluateFunModel(X_test,y_test,model,model_name)


################### CNN2D on Embeddings swith ON MinMax ALL#############################################
# import numpy as np
# from PIL import Image
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
# epochs = 100
# model_name = f'picture_baseline_{epochs}_Embeding_CNN2D_2'
# batch_size = 1024
# non_swith = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek','Salary','Sex']
# X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)
#
# X_train = dataproc.swith_merge(X_train,non_swith)
#
# X_train,y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train = dataproc.to_numpy_data(X_train,X_train.columns)
# X_test = dataproc.to_numpy_data(X_test,X_test.columns)
#
# train_zeros = np.zeros((len(X_train),7))
# test_zeros = np.zeros((len(X_test),7))
# X_train = np.concatenate((X_train,train_zeros),1)
# X_test = np.concatenate((X_test,test_zeros),1)
#
#
# X_train =X_train.reshape(-1,8,10,1)
# X_test = X_test.reshape(-1,8,10,1)
# # print(X_train.shape)
# # print(X_train[0].shape)
# # # print(X_train[0][0])
# model = modeler.model_Fun_CNN2((8,10,1))
# model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs)
# modeler.evaluateFunModel(X_test,y_test,model,model_name)

################### CNN2D + Dense on Embeddings swith ON MinMax ALL#############################################
# import numpy as np
# embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
# epochs = 300
# model_name = f'picture_baseline_{epochs}_Embeding_CNN2DandDense'
# batch_size = 1024
# X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)
#
#
# numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
# pixels = [col for col in X_train.columns if col not in numerical_col]
# X_train = dataproc.swith_merge(X_train, numerical_col)
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
#
# train_zeros = np.zeros((len(X_train),3))
# test_zeros = np.zeros((len(X_test),3))
# X_train_pixels = np.concatenate((X_train_pixels,train_zeros),1)
# X_test_pixels = np.concatenate((X_test_pixels,test_zeros),1)
#
# X_train_pixels =X_train_pixels.reshape(-1,7,10,1)
# X_test_pixels = X_test_pixels.reshape(-1,7,10,1)
#
# model = modeler.model_CNN_Dense(CNN_shape=(7,10,1),Dense_shape=(6,))
# model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
# modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)

#################### CNN2D + Dense on Embeddings swith ON MinMax ALL#############################################
import numpy as np
embedding_model = '/home/piotr/data/test/models/fun_50_EmbeddSource_picture.h5'
epochs = 100
model_name = f'picture_baseline_{epochs}_Embeding_VGG16andDense'
batch_size = 32
X_train,X_test = dataproc.dataload_minmaxall(categorical,embedding_model,weights)


numerical_col = ['Age', 'EducationNum', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Salary', 'Sex']
pixels = [col for col in X_train.columns if col not in numerical_col]
# X_train = dataproc.swith_merge(X_train, numerical_col)
X_train_pixels = X_train[pixels]
X_train_numerical = X_train[numerical_col]
X_test_pixels = X_test[pixels]
X_test_numerical = X_test[numerical_col]

X_test_numerical.drop('Salary',axis=1,inplace=True)
X_train_numerical.drop('Salary',axis=1,inplace=True)

_,y_train = dataproc.split_data(X_train,'Salary')
_, y_test = dataproc.split_data(X_test,'Salary')

X_train_pixels = dataproc.to_numpy_data(X_train_pixels,X_train_pixels.columns)
X_train_numerical = dataproc.to_numpy_data(X_train_numerical,X_train_numerical.columns)
X_test_pixels = dataproc.to_numpy_data(X_test_pixels,X_test_pixels.columns)
X_test_numerical = dataproc.to_numpy_data(X_test_numerical,X_test_numerical.columns)

# train_zeros = np.zeros((len(X_train),3))
# test_zeros = np.zeros((len(X_test),3))
ten = 2500
train_zeros = np.zeros((len(X_train),ten-67))
test_zeros = np.zeros((len(X_test),ten-67))
print(X_train_pixels.shape)
X_train_pixels = np.concatenate((X_train_pixels,train_zeros),1)
X_test_pixels = np.concatenate((X_test_pixels,test_zeros),1)
print(X_train_pixels.shape)
X_train_pixels = np.concatenate((X_train_pixels,X_train_pixels,X_train_pixels))
X_test_pixels = np.concatenate((X_test_pixels,X_test_pixels,X_test_pixels))
print(X_train_pixels.shape)
sh = 50
X_train_pixels =X_train_pixels.reshape(-1,sh,sh,3)
X_test_pixels = X_test_pixels.reshape(-1,sh,sh,3)

model = modeler.model_VGG16_Dense(CNN_shape=(sh,sh,3),Dense_shape=(6,))
model.fit([X_train_pixels,X_train_numerical],y_train,batch_size=batch_size,epochs=epochs)
modeler.evaluateFunModel([X_test_pixels,X_test_numerical],y_test,model,model_name)