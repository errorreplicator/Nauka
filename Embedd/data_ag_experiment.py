import numpy as np
import pandas as pd
from Embedd.modeler import get_model_Emb1Dropout
from Embedd.dataproc import data_categ, read_data, data_categ_numpy, data_categ_clean, scale_row
desired_width = 320
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 200)

categorical = ['Workclass', 'Education', 'MaritalStatus','Occupation','Relationship','Race','Sex','Country']
numerical = ['Age','EducationNum','CapitalGain', 'CapitalLoss','HoursWeek']

# X_train, y_train, X_test, y_test = data_categ_numpy(categorical,numerical)
# train, test = data_categ(categorical,numerical)

# print(train['Age'].describe())
# print(train['EducationNum'].describe())

# print(train.columns)

train, test = read_data()
train2 = train[['EducationNum', 'Workclass', 'Education', 'Age', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HoursWeek', 'Country', 'Salary']]
# print(train2.head(20))
print(train.shape)
train = train.append(train2,sort=False)
print(train.shape)
# ??X_train, y_train, X_test, y_test = data_categ(categorical, numerical, train, test)
# ??model = get_model_Emb1Dropout()
# ??model.fit(X_train,y_train,epochs=300, batch_size=256, validation_split=0.2)

train, test = data_categ_clean(categorical,numerical,train,test)

# train = train[train.columns.difference(numerical)]
# print(train)

print(train.head(1))
print(train.iloc[:,:-1].head(5))
train_split = scale_row(train.iloc[:,:-1].head(5),numerical)
print(train_split)
