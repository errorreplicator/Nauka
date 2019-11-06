import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Embedd import modeler, dataproc
from sklearn.preprocessing import MinMaxScaler

desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']


train, test = dataproc.read_data()


X_train,y_train, X_test, y_test = dataproc.data_func_swithONscaleROW()


model = modeler.get_model_Emb1DropoutBIG()
model.fit(X_train, y_train.values, epochs=300, batch_size=256, validation_split=0.2)
model.save('/home/piotr/data/test/model_300_swithON_EmbeddBIGDrop.h5')




# train, test = read_data()
# train, test = data_categ_clean(categorical,numerical,train,test)
# train_split = scale_row(train,numerical)
# train_split = Embedd.dataproc.swith_merge(train_split,numerical)
# X_train, y_train, X_test, y_test = Embedd.dataproc.data_categ_normalized(categorical, numerical, train_split, test)
# # model = modeler.get_model_Emb1DropoutBIG()
# model = modeler.get_model_Emb1()
# model.fit(X_train,y_train,epochs=100, batch_size=256, validation_split=0.2)
# # model.save('/home/piotr/data/test/modelBIG_300_bs256.h5')