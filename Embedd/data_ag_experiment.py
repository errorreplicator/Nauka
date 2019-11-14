
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from Embedd import modeler, dataproc

desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']


########################### Sequential Embeddings OFF########################################################
# model_name = 'seq_200_EmbeddOFF_2run'
# X_train,y_train,X_test,y_test = dataproc.data_seq_swithOFF(categorical,numerical)
#
# model = modeler.get_model_Seq((13,))
# model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test))
# model.save(f'/home/piotr/data/test/{model_name}.h5')
#
# actual, classes, pred = modeler.evaluateModel(X_test,y_test,model,model_name)

########################### Functional Embeddings ON########################################################

model_name = 'fun_100_EmbeddON_tmp'


# train, test = dataproc.read_data()
# print(train.head())
# train = dataproc.remove_data(train,'Id')
# train = dataproc.labelencoder(train,categorical)
# train = dataproc.labelencoder(train,['Salary'])
# train = dataproc.minmax_column(train,numerical)
# X_train, y_train = dataproc.split_data(train, 'Salary')
# X_train_dict= {col: dataproc.to_numpy_data(X_train, col) for col in categorical}
# X_train_dict['Numerical'] = dataproc.to_numpy_data(X_train, numerical)
# print(X_train.head())
# print(X_train_dict)


X_train, y_train,X_test,y_test = dataproc.data_func_swithOFF(categorical,numerical)

model = modeler.get_model_Emb1Dropout()
model.fit(X_train,y_train,epochs=30,batch_size=1024)
model.save(f'/home/piotr/data/test/{model_name}.h5')
modeler.evaluateFunModel(X_test,y_test,model,model_name)
