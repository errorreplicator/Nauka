from Embedd import modeler, dataproc
import numpy as np
import pandas as pd
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
np.set_printoptions(threshold=np.inf)
from keras.models import Sequential, load_model, Model
from sklearn.metrics import confusion_matrix
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']
weights = ['Workclass_emb','Education_emb','MaritalStatus_emb','Occupation_emb','Relationship_emb','Race_emb','Sex_emb','Country_emb']

dataproc.fix_seeds(1)

###############Train Embeddings 50 baseline ######################################################

# epochs = 50
# model_name = f'fun_{epochs}_EmbeddSource'
#
# X_train, X_test = dataproc.dataload_stage1(categorical,numerical,onehot=False)
#
# X_train, y_train = dataproc.split_data(X_train,'Salary')
# X_test, y_test = dataproc.split_data(X_test,'Salary')
#
# X_train = dataproc.data_tomodel(X_train,categorical,numerical)
# X_test = dataproc.data_tomodel(X_test,categorical,numerical)
#
# model = modeler.get_model_Emb1DropoutBIG()
# #
# model.fit(X_train,y_train,epochs=epochs,batch_size=128)
# modeler.evaluateFunModel(X_test, y_test, model, model_name)

###############Train Embeddings 50 baseline with correction######################################################

epochs = 50
model_name = f'fun_{epochs}_EmbeddSource_picture'

X_train, X_test = dataproc.dataload_stage1(categorical,numerical,onehot=False)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Country','Sex']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

X_train, y_train = dataproc.split_data(X_train,'Salary')
X_test, y_test = dataproc.split_data(X_test,'Salary')

X_train = dataproc.data_tomodel(X_train,categorical,numerical)
X_test = dataproc.data_tomodel(X_test,categorical,numerical)

model = modeler.get_model_Emb1DropoutBIG_corr()
#
model.fit(X_train,y_train,epochs=epochs,batch_size=512)
modeler.evaluateFunModel(X_test, y_test, model, model_name)