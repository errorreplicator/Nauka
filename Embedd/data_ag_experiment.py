import numpy as np
import pandas as pd
from Embedd import modeler
from Embedd.dataproc import data_categ, read_data, data_categ_numpy, data_categ_clean, scale_row
import Embedd.dataproc
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

# X_train, y_train, X_test, y_test = data_categ_numpy(categorical,numerical)
# train, test = data_categ(categorical,numerical)


train, test = read_data()
# train2 = train[['EducationNum', 'Workclass', 'Education', 'Age', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Country', 'Salary']]
# train = train.append(train2,sort=False)

# ??X_train, y_train, X_test, y_test = data_categ(categorical, numerical, train, test)
# ??model = get_model_Emb1Dropout()
# ??model.fit(X_train,y_train,epochs=300, batch_size=256, validation_split=0.2)

train, test = data_categ_clean(categorical,numerical,train,test)
# print(train.shape)
# train = train[train.columns.difference(numerical)]
# print(train)

train_split = scale_row(train,numerical)

train_split = Embedd.dataproc.swith_merge(train_split,numerical)

# print(train_split.shape)
# print(train_split.head().append(train_split.tail()))


X_train, y_train, X_test, y_test = Embedd.dataproc.data_categ_normalized(categorical, numerical, train_split, test)
# model = modeler.get_model_Emb1DropoutBIG()
model = modeler.get_model_Emb1()
model.fit(X_train,y_train,epochs=100, batch_size=256, validation_split=0.2)
# model.save('/home/piotr/data/test/modelBIG_300_bs256.h5')