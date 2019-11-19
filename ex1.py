import Embedd.dataproc as dp
import Embedd.modeler as mod
import numpy as np
import pandas as pd
from keras.models import load_model
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)
#
# train, test = dp.read_data()
#
# categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
# numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

############TEST CHECK###########################
# print(test.shape)
# print(test.head())

# test = dp.labelencoder(test, categorical)
# test = dp.labelencoder(test,['Salary'])
# test = dp.minmax_column(test, numerical)
# test = dp.remove_data(test,'Id')
# X_test, y_test = dp.split_data(test,'Salary')
#
# X_train,y_train,X_test,y_test = dp.data_seq_swithOFF(categorical,numerical,numpyON=False)
#
# weights_list = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']
# weights_path ='/home/piotr/data/test/models/fun_300_EmbeddSource_flatten_names.h5'
# model = load_model(weights_path)
# X_train_weight = dp.weights2df(X_train, weights_path, weights_list, del_categ=False)
# X_test_weight = dp.weights2df(X_test, weights_path, weights_list, del_categ=False)
#
# for col in categorical:
#     X_train_weight[col] += 1
#     dvd = X_train_weight[col].max()
#     X_train_weight[col] /= dvd
#
# for col in categorical:
#     X_test_weight[col] += 1
#     dvd = X_test_weight[col].max()
#     X_test_weight[col] /= dvd
#
# # y_train = y_train.append(y_train)
# X_train_weight = dp.minmax_column(X_train_weight,X_train_weight.columns)
# X_train_weight = dp.swith_merge(X_train_weight,numerical)
# X_test_weight = dp.minmax_column(X_test_weight,X_train_weight.columns)
# X_test_weight = dp.swith_merge(X_test_weight,X_test_weight.columns)
#
#
# epochs = 100
# model_name = f'seq_{epochs}_EmbeddONSwithON_tester'
#
# # train_model = mod.get_model_Seq((64,))
# # train_model.fit(X_train_weight, y_train, epochs=epochs, batch_size=1024,validation_split=0.2)
# # train_model.save(f'/home/piotr/data/test/models/{model_name}.h5')
# # mod.evaluateSeqModel(X_test_weight, y_test, train_model, model_name)
#
# X_train_weight = dp.to_numpy_data(X_train_weight,X_train_weight.columns)
# X_test_weight = dp.to_numpy_data(X_test_weight,X_test_weight.columns)
# X_train_weight =X_train_weight.reshape(X_train_weight.shape[0],X_train_weight.shape[1],1)
# X_test_weight =X_test_weight.reshape(X_test_weight.shape[0],X_test_weight.shape[1], 1)
#
# print(X_train_weight.shape)
#
# train_model = mod.model_Fun_CNN1((X_train_weight.shape[1],1))
# train_model.fit(X_train_weight,y_train,epochs=epochs,batch_size=1024,validation_split=0.2)
# # model.save(f'/home/piotr/data/test/models/{model_name}.h5')
#
# # mod.evaluateSeqModel(X_test_weight,y_test,train_model,model_name)

from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler(feature_range=(0, 1))

X_train = [[2,5,2,7],[1,9,8,4],[6,8,3,1]]
X_test = [[2,5,2,7],[1,9,8,4]]
# print(scaler.fit_transform(X_train))

scaler.fit(X_train)
X_train_trans = scaler.transform(X_train)

X_test_trains = scaler.transform(X_test)
print(X_train_trans)
print(X_test_trains)

X_train = np.array(X_train)
X_test = np.array(X_test)
scaler.fit(X_train.T)
X_train = scaler.transform(X_train.T).T
print(X_train)
X_test = scaler.transform(X_test.T).T


























